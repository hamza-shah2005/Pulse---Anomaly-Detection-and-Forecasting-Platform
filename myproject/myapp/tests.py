import io
import datetime
from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone

from .models import User, Dataset, DataPoint, Forecast, Machine, MachineReading, DowntimeEvent, EnergyForecast
from .tasks import (
    ingest_csv_task, preprocess_dataset_task, detect_anomalies_task, generate_forecast_task,
    ingest_machine_readings_task, detect_machine_anomalies_task,
    root_cause_analysis_task, machine_energy_forecast_task, energy_optimization_task,
)


class AuthenticationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="testuser",
            email="test@example.com",
            password="StrongPassword123!"
        )

    def test_login_success(self):
        response = self.client.post(reverse("login"), {
            "email": "test@example.com",
            "password": "StrongPassword123!"
        })
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, reverse("landing_page"))

    def test_login_invalid_password(self):
        response = self.client.post(reverse("login"), {
            "email": "test@example.com",
            "password": "WrongPassword!"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Invalid email or password.")

    def test_register_success(self):
        response = self.client.post(reverse("register"), {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "SecurePassword456!",
            "confirm_password": "SecurePassword456!"
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(User.objects.filter(email="newuser@example.com").exists())

    def test_register_password_mismatch(self):
        response = self.client.post(reverse("register"), {
            "username": "mismatchuser",
            "email": "mismatch@example.com",
            "password": "SecurePassword456!",
            "confirm_password": "DifferentPassword456!"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Passwords do not match.")
        self.assertFalse(User.objects.filter(email="mismatch@example.com").exists())

    def test_register_duplicate_email(self):
        response = self.client.post(reverse("register"), {
            "username": "anotheruser",
            "email": "test@example.com",
            "password": "SecurePassword456!",
            "confirm_password": "SecurePassword456!"
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "An account with this email already exists.")


class DataPipelineTasksTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username="pipelineuser",
            email="pipeline@example.com",
            password="StrongPassword123!"
        )

    def _create_sample_csv(self, rows=50, with_duplicates=False, with_nans=False):
        content = "timestamp,power_usage\n"
        base_time = timezone.now() - datetime.timedelta(hours=rows)
        for i in range(rows):
            ts = (base_time + datetime.timedelta(minutes=i * 10)).isoformat()
            val = 20.0 + (i % 5) * 2.0
            if with_nans and i in (10, 25):
                content += f"{ts},\n"
            else:
                content += f"{ts},{val}\n"

        if with_duplicates:
            # Duplicate the 5th timestamp
            dup_ts = (base_time + datetime.timedelta(minutes=5 * 10)).isoformat()
            content += f"{dup_ts},99.0\n"

        return SimpleUploadedFile(
            "sample.csv",
            content.encode("utf-8"),
            content_type="text/csv"
        )

    def test_csv_ingestion_and_auto_target_detection(self):
        csv_file = self._create_sample_csv(rows=40)
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Test Series",
            source_file=csv_file,
            source_type="csv",
            status="pending"
        )

        ingest_csv_task(str(dataset.id))
        dataset.refresh_from_db()

        self.assertEqual(dataset.status, "ready")
        self.assertEqual(dataset.row_count, 40)
        self.assertEqual(dataset.target_column, "power_usage")
        self.assertEqual(DataPoint.objects.filter(dataset=dataset).count(), 40)

    def test_preprocessing_task(self):
        csv_file = self._create_sample_csv(rows=40, with_duplicates=False, with_nans=True)
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Preprocessing Test",
            source_file=csv_file,
            source_type="csv",
            status="pending"
        )

        ingest_csv_task(str(dataset.id))
        preprocess_dataset_task(str(dataset.id))
        dataset.refresh_from_db()

        self.assertEqual(dataset.preprocessing_status, "ready")
        self.assertGreater(dataset.missing_values_filled, 0)
        # Ensure no NaN points remain in DB
        points = DataPoint.objects.filter(dataset=dataset)
        for p in points:
            self.assertIsNotNone(p.value)

    def test_anomaly_detection_isolation_forest(self):
        # Create dataset with normal points and 2 extreme anomalies
        csv_file = self._create_sample_csv(rows=60)
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Anomaly Test",
            source_file=csv_file,
            source_type="csv",
            status="pending"
        )
        ingest_csv_task(str(dataset.id))

        # Inject extreme anomalies
        pts = list(DataPoint.objects.filter(dataset=dataset).order_by("timestamp"))
        pts[15].value = 500.0
        pts[15].save()
        pts[45].value = -200.0
        pts[45].save()

        detect_anomalies_task(str(dataset.id), model_name="isolation_forest", contamination=0.05)
        dataset.refresh_from_db()

        self.assertEqual(dataset.anomaly_status, "ready")
        self.assertGreater(dataset.anomaly_count, 0)
        self.assertTrue(DataPoint.objects.filter(dataset=dataset, is_anomaly=True).exists())

    def test_forecasting_arima(self):
        csv_file = self._create_sample_csv(rows=50)
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Forecast Test ARIMA",
            source_file=csv_file,
            source_type="csv",
            status="pending"
        )
        ingest_csv_task(str(dataset.id))

        generate_forecast_task(str(dataset.id), model_name="arima", steps=12)
        dataset.refresh_from_db()

        self.assertEqual(dataset.forecast_status, "ready")
        self.assertEqual(dataset.forecast_model, "arima")
        forecasts = Forecast.objects.filter(dataset=dataset)
        self.assertEqual(forecasts.count(), 12)
        self.assertIsNotNone(forecasts.first().predicted_value)

    def test_forecasting_prophet(self):
        csv_file = self._create_sample_csv(rows=50)
        dataset = Dataset.objects.create(
            owner=self.user,
            name="Forecast Test Prophet",
            source_file=csv_file,
            source_type="csv",
            status="pending"
        )
        ingest_csv_task(str(dataset.id))

        generate_forecast_task(str(dataset.id), model_name="prophet", steps=12)
        dataset.refresh_from_db()

        self.assertEqual(dataset.forecast_status, "ready")
        self.assertEqual(dataset.forecast_model, "prophet")
        forecasts = Forecast.objects.filter(dataset=dataset)
        self.assertEqual(forecasts.count(), 12)
        self.assertIsNotNone(forecasts.first().predicted_value)


class ViewsAndAPITests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(
            username="viewuser",
            email="view@example.com",
            password="StrongPassword123!"
        )
        self.client.force_login(self.user)

        self.dataset = Dataset.objects.create(
            owner=self.user,
            name="API Test Dataset",
            status="ready",
            row_count=10
        )
        base_time = timezone.now() - datetime.timedelta(hours=10)
        for i in range(10):
            DataPoint.objects.create(
                dataset=self.dataset,
                timestamp=base_time + datetime.timedelta(minutes=i * 10),
                value=50.0 + i,
                is_anomaly=(i == 5)
            )

    def test_dataset_status_api(self):
        url = reverse("dataset_status", kwargs={"dataset_id": self.dataset.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "API Test Dataset")
        self.assertEqual(data["row_count"], 10)

    def test_update_model_settings_api(self):
        url = reverse("model_settings", kwargs={"dataset_id": self.dataset.id})
        response = self.client.post(url, {
            "anomaly_model": "lof",
            "forecast_model": "arima",
            "contamination": "0.05"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["anomaly_model"], "lof")
        self.assertEqual(data["forecast_model"], "arima")
        self.assertEqual(data["contamination"], 0.05)

    def test_anomaly_data_api(self):
        url = reverse("anomaly_data_api", kwargs={"dataset_id": self.dataset.id})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_points"], 10)
        self.assertEqual(data["anomaly_count"], 1)
        self.assertIn("chart", data)
        self.assertIn("anomalies", data)

    def test_upload_view_rejects_non_csv(self):
        bad_file = SimpleUploadedFile("bad.txt", b"some text content", content_type="text/plain")
        response = self.client.post(reverse("upload_data_view"), {
            "file": bad_file
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only CSV files are supported", response.json().get("error", ""))


# ═══════════════════════════════════════════════════════════════
# IoT / PC Sensor Pipeline Tests
# ═══════════════════════════════════════════════════════════════

class MachinePipelineTests(TestCase):
    """
    End-to-end tests for the industrial IoT / PC-sensor pipeline:
    Machine CRUD → reading ingestion → anomaly detection → RCA → energy forecast → optimization.
    """

    def setUp(self):
        self.user = User.objects.create_user(
            username="iotuser",
            email="iot@example.com",
            password="StrongIoT123!"
        )
        self.machine = Machine.objects.create(
            owner=self.user,
            name="Test PC",
            machine_type="pc",
            location="Lab",
        )

    def _make_readings(self, n=60, inject_anomaly_at=None):
        """
        Generate n synthetic PC sensor readings at 1-minute intervals.
        Optionally inject anomalous readings at the given index list.
        """
        import random
        readings = []
        base_ts = timezone.now() - timezone.timedelta(minutes=n)
        for i in range(n):
            ts = base_ts + timezone.timedelta(minutes=i)
            cpu = random.uniform(30, 60)
            temp = 45 + cpu * 0.3 + random.uniform(-2, 2)
            ram = random.uniform(40, 70)
            power = 10.0 + (cpu / 100.0) * 55.0 + ram * 0.02

            if inject_anomaly_at and i in inject_anomaly_at:
                cpu = 99.0
                temp = 95.0
                ram = 95.0
                power = 65.0

            readings.append({
                "timestamp": ts.isoformat(),
                "cpu_usage_percent": cpu,
                "cpu_temp_celsius": temp,
                "ram_usage_percent": ram,
                "disk_read_mbps": random.uniform(0, 10),
                "disk_write_mbps": random.uniform(0, 5),
                "net_sent_mbps": random.uniform(0, 1),
                "net_recv_mbps": random.uniform(0, 2),
                "process_count": random.randint(100, 200),
                "battery_percent": None,
                "estimated_power_watts": round(power, 2),
            })
        return readings

    def test_machine_creation(self):
        """Machine record is created and accessible."""
        self.assertEqual(self.machine.name, "Test PC")
        self.assertEqual(self.machine.machine_type, "pc")
        self.assertEqual(self.machine.owner, self.user)

    def test_ingest_readings(self):
        """ingest_machine_readings_task inserts readings into MachineReading."""
        readings = self._make_readings(n=30)
        result = ingest_machine_readings_task(str(self.machine.id), readings)
        self.assertEqual(result["inserted"], 30)
        self.assertEqual(MachineReading.objects.filter(machine=self.machine).count(), 30)

    def test_ingest_readings_dedup(self):
        """Duplicate timestamps are silently ignored (ignore_conflicts=True)."""
        readings = self._make_readings(n=20)
        ingest_machine_readings_task(str(self.machine.id), readings)
        result2 = ingest_machine_readings_task(str(self.machine.id), readings)
        # Second insert should insert 0 new rows
        self.assertEqual(result2["inserted"], 20)  # bulk_create returns all, DB ignores dupes
        self.assertEqual(MachineReading.objects.filter(machine=self.machine).count(), 20)

    def test_detect_machine_anomalies(self):
        """detect_machine_anomalies_task flags anomalous readings and creates DowntimeEvent."""
        readings = self._make_readings(n=60, inject_anomaly_at=[55, 56, 57, 58, 59])
        ingest_machine_readings_task(str(self.machine.id), readings)

        result = detect_machine_anomalies_task(str(self.machine.id), contamination=0.08, lookback_hours=2)

        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["anomalies_found"], 0)

        # At least one anomaly flag set in DB
        self.assertTrue(MachineReading.objects.filter(machine=self.machine, is_anomaly=True).exists())

        # DowntimeEvent should have been auto-created for the cluster of 5 anomalies
        events = DowntimeEvent.objects.filter(machine=self.machine)
        self.assertGreater(events.count(), 0)

    def test_root_cause_analysis(self):
        """root_cause_analysis_task produces a structured RCA result for a DowntimeEvent."""
        # Inject anomalies and create a known DowntimeEvent
        readings = self._make_readings(n=60, inject_anomaly_at=[50, 51, 52, 53, 54])
        ingest_machine_readings_task(str(self.machine.id), readings)
        detect_machine_anomalies_task(str(self.machine.id), contamination=0.08, lookback_hours=2)

        event = DowntimeEvent.objects.filter(machine=self.machine).first()
        self.assertIsNotNone(event)

        rca = root_cause_analysis_task(str(self.machine.id), str(event.id), pre_event_minutes=30)

        self.assertIn("summary", rca)
        self.assertIn("top_causes", rca)
        self.assertIn("recommendations", rca)
        self.assertIsInstance(rca["top_causes"], list)
        self.assertIsInstance(rca["recommendations"], list)

        event.refresh_from_db()
        self.assertEqual(event.rca_status, "ready")
        self.assertIsNotNone(event.rca_results)

    def test_energy_forecast(self):
        """machine_energy_forecast_task produces EnergyForecast records."""
        readings = self._make_readings(n=60)
        ingest_machine_readings_task(str(self.machine.id), readings)

        result = machine_energy_forecast_task(str(self.machine.id), steps=12, model_name="arima")
        self.assertEqual(result["status"], "ok")

        forecasts = EnergyForecast.objects.filter(machine=self.machine)
        self.assertEqual(forecasts.count(), 12)
        self.assertIsNotNone(forecasts.first().predicted_watts)

    def test_energy_optimization(self):
        """energy_optimization_task returns efficiency curve and recommendations."""
        readings = self._make_readings(n=60)
        ingest_machine_readings_task(str(self.machine.id), readings)

        result = energy_optimization_task(str(self.machine.id), lookback_hours=2)
        self.assertEqual(result["status"], "ok")
        self.assertIn("efficiency_curve", result)
        self.assertIn("recommendations", result)
        self.assertIn("optimal_cpu_range", result)
        self.assertGreater(len(result["efficiency_curve"]), 0)

    def test_machine_list_api(self):
        """GET /api/machines/ returns machines for authenticated user."""
        self.client = __import__("django.test", fromlist=["Client"]).Client()
        self.client.login(username="iotuser", password="StrongIoT123!")
        response = self.client.get(reverse("machine_list"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("machines", data)
        self.assertEqual(len(data["machines"]), 1)

    def test_machine_readings_api_post(self):
        """POST /api/machines/<id>/readings/ accepts JSON readings."""
        import json
        self.client = __import__("django.test", fromlist=["Client"]).Client()
        self.client.login(username="iotuser", password="StrongIoT123!")
        readings = self._make_readings(n=5)
        response = self.client.post(
            reverse("machine_readings", args=[str(self.machine.id)]),
            data=json.dumps({"readings": readings}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["inserted"], 5)

    def test_machine_dashboard_api(self):
        """GET /api/machines/<id>/dashboard/ returns chart and health data."""
        import json
        self.client = __import__("django.test", fromlist=["Client"]).Client()
        self.client.login(username="iotuser", password="StrongIoT123!")
        readings = self._make_readings(n=10)
        ingest_machine_readings_task(str(self.machine.id), readings)

        response = self.client.get(reverse("machine_dashboard", args=[str(self.machine.id)]))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("machine", data)
        self.assertIn("chart", data)
        self.assertIn("latest_reading", data)

