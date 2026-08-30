import statistics
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from rest_framework import generics
from rest_framework.permissions import AllowAny

from .models import DataPoint, Dataset, Forecast, User, Machine, MachineReading, DowntimeEvent, EnergyForecast
from .serializers import RegisterSerializer, UserSerializer
from .tasks import (
    detect_anomalies_task, generate_forecast_task, ingest_csv_task, preprocess_dataset_task,
    ingest_machine_readings_task, detect_machine_anomalies_task,
    root_cause_analysis_task, machine_energy_forecast_task, energy_optimization_task,
)

MAX_UPLOAD_SIZE = 500 * 1024 * 1024   # 500 MB


# ─────────────────────────────────────────────
# DRF API VIEWS
# ─────────────────────────────────────────────

class RegisterView(generics.CreateAPIView):
    serializer_class = RegisterSerializer
    permission_classes = [AllowAny]


class ProfileView(generics.RetrieveAPIView):
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user


# ─────────────────────────────────────────────
# AUTHENTICATION & TEMPLATE VIEWS
# ─────────────────────────────────────────────

def landing_page(request):
    return render(request, "landing_page.html")


def login_page(request):
    if request.user.is_authenticated:
        return redirect("landing_page")

    if request.method == "POST":
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")

        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            return redirect("landing_page")

        messages.error(request, "Invalid email or password.")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


def register(request):
    if request.user.is_authenticated:
        return redirect("landing_page")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password", "")
        confirm_password = request.POST.get("confirm_password", "")

        if not username or not email or not password:
            messages.error(request, "All fields are required.")
            return render(request, "register.html")

        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, "register.html")

        try:
            validate_password(password)
        except ValidationError as e:
            for error in e.messages:
                messages.error(request, error)
            return render(request, "register.html")

        if User.objects.filter(username=username).exists():
            messages.error(request, "That username is already taken.")
            return render(request, "register.html")

        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return render(request, "register.html")

        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        return redirect("landing_page")

    return render(request, "register.html")


# ─────────────────────────────────────────────
# DATASET MANAGEMENT VIEWS
# ─────────────────────────────────────────────

@login_required
def upload_data_view(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("file")
        target_column = request.POST.get("target_column", "").strip()

        if not uploaded_file:
            return JsonResponse({"error": "Please choose a file to upload."}, status=400)
        if not uploaded_file.name.lower().endswith(".csv"):
            return JsonResponse({"error": "Only CSV files are supported."}, status=400)
        if uploaded_file.size > MAX_UPLOAD_SIZE:
            return JsonResponse({"error": "File is too large (max 500MB)."}, status=400)

        estimate = max(1, uploaded_file.size // 40)

        dataset = Dataset.objects.create(
            owner=request.user,
            name=request.POST.get("name") or uploaded_file.name,
            source_file=uploaded_file,
            source_type="csv",
            target_column=target_column or "value",
            status="pending",
            total_rows_estimate=estimate,
        )

        ingest_csv_task.delay(str(dataset.id), value_column=target_column or None)
        dataset.refresh_from_db()

        return JsonResponse({
            "dataset_id": str(dataset.id),
            "name": dataset.name,
            "status": dataset.status,
            "target_column": dataset.target_column,
        })

    return JsonResponse({"error": "POST required"}, status=405)


@login_required
def dataset_status_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    return JsonResponse({
        "dataset_id": str(dataset.id),
        "name": dataset.name,
        "status": dataset.status,
        "row_count": dataset.row_count,
        "target_column": dataset.target_column,
        "preprocessing_status": dataset.preprocessing_status,
        "duplicate_rows_removed": dataset.duplicate_rows_removed,
        "missing_values_filled": dataset.missing_values_filled,
        "anomaly_status": dataset.anomaly_status,
        "anomaly_count": dataset.anomaly_count,
        "anomaly_model": dataset.anomaly_model,
        "contamination": dataset.contamination,
        "forecast_status": dataset.forecast_status,
        "forecast_model": dataset.forecast_model,
        "forecast_mae": dataset.forecast_mae,
        "forecast_rmse": dataset.forecast_rmse,
        "forecast_mape": dataset.forecast_mape,
        "error_message": dataset.error_message,
    })


@login_required
def run_preprocessing_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    preprocess_dataset_task.delay(str(dataset.id))
    dataset.refresh_from_db()

    return JsonResponse({
        "preprocessing_status": dataset.preprocessing_status,
        "duplicate_rows_removed": dataset.duplicate_rows_removed,
        "missing_values_filled": dataset.missing_values_filled,
        "row_count": dataset.row_count,
    })


@login_required
def update_model_settings_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)

    if request.method == "POST":
        anomaly_model = request.POST.get("anomaly_model")
        forecast_model = request.POST.get("forecast_model")
        contamination_raw = request.POST.get("contamination")

        valid_anomaly = dict(Dataset.ANOMALY_MODEL_CHOICES)
        valid_forecast = dict(Dataset.FORECAST_MODEL_CHOICES)

        if anomaly_model and anomaly_model not in valid_anomaly:
            return JsonResponse({"error": "Invalid anomaly model."}, status=400)
        if forecast_model and forecast_model not in valid_forecast:
            return JsonResponse({"error": "Invalid forecast model."}, status=400)

        update_fields = []
        if anomaly_model:
            dataset.anomaly_model = anomaly_model
            update_fields.append("anomaly_model")
        if forecast_model:
            dataset.forecast_model = forecast_model
            update_fields.append("forecast_model")
        if contamination_raw is not None:
            try:
                c_val = float(contamination_raw)
                dataset.contamination = max(0.001, min(0.5, c_val))
                update_fields.append("contamination")
            except ValueError:
                pass

        if update_fields:
            dataset.save(update_fields=update_fields)

        return JsonResponse({
            "anomaly_model": dataset.anomaly_model,
            "forecast_model": dataset.forecast_model,
            "contamination": dataset.contamination,
        })

    return JsonResponse({
        "anomaly_model": dataset.anomaly_model,
        "forecast_model": dataset.forecast_model,
        "contamination": dataset.contamination,
        "anomaly_choices": Dataset.ANOMALY_MODEL_CHOICES,
        "forecast_choices": Dataset.FORECAST_MODEL_CHOICES,
    })


@login_required
def run_anomaly_detection_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    model_name = request.POST.get("model") or dataset.anomaly_model
    contamination = request.POST.get("contamination")
    contam_val = float(contamination) if contamination else dataset.contamination

    detect_anomalies_task.delay(str(dataset.id), model_name=model_name, contamination=contam_val)
    dataset.refresh_from_db()
    return JsonResponse({
        "anomaly_status": dataset.anomaly_status,
        "anomaly_count": dataset.anomaly_count,
        "anomaly_model": dataset.anomaly_model,
        "contamination": dataset.contamination,
    })


@login_required
def run_forecast_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    model_name = request.POST.get("model") or dataset.forecast_model
    steps = int(request.POST.get("steps", 24))
    generate_forecast_task.delay(str(dataset.id), model_name=model_name, steps=steps)
    dataset.refresh_from_db()
    return JsonResponse({
        "forecast_status": dataset.forecast_status,
        "forecast_model": dataset.forecast_model,
    })


# ─────────────────────────────────────────────
# DETAIL AND API DATA VIEWS
# ─────────────────────────────────────────────

@login_required
def anomaly_detail_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    return render(request, "anomaly_detail.html", {"dataset": dataset})


@login_required
def anomaly_data_api(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)

    all_points_qs = DataPoint.objects.filter(dataset=dataset).order_by("timestamp")
    total_count = all_points_qs.count()

    if total_count == 0:
        return JsonResponse({"error": "No data points found for this dataset."}, status=404)

    normal_points = list(all_points_qs.filter(is_anomaly=False).values("timestamp", "value"))
    anomaly_points = list(all_points_qs.filter(is_anomaly=True).values("id", "timestamp", "value"))

    normal_values = [p["value"] for p in normal_points]
    mean_val = statistics.mean(normal_values) if normal_values else 0
    std_val = statistics.pstdev(normal_values) if len(normal_values) > 1 else 1
    if std_val == 0:
        std_val = 1

    MAX_CHART_POINTS = 1500
    stride = max(1, len(normal_points) // MAX_CHART_POINTS)
    sampled_normal = normal_points[::stride]

    chart_points = sampled_normal + [{"timestamp": p["timestamp"], "value": p["value"]} for p in anomaly_points]
    chart_points.sort(key=lambda p: p["timestamp"])

    anomaly_set = {p["timestamp"].isoformat() for p in anomaly_points}

    chart_data = {
        "timestamps": [p["timestamp"].isoformat() for p in chart_points],
        "values": [p["value"] for p in chart_points],
        "is_anomaly": [p["timestamp"].isoformat() in anomaly_set for p in chart_points],
    }

    anomaly_details = []
    for p in anomaly_points:
        z_score = round((p["value"] - mean_val) / std_val, 2)
        abs_z = abs(z_score)
        if abs_z >= 3:
            severity = "critical"
        elif abs_z >= 2:
            severity = "high"
        else:
            severity = "moderate"

        anomaly_details.append({
            "id": p["id"],
            "timestamp": p["timestamp"].isoformat(),
            "value": p["value"],
            "z_score": z_score,
            "severity": severity,
            "direction": "above expected range" if p["value"] > mean_val else "below expected range",
        })

    anomaly_details.sort(key=lambda a: abs(a["z_score"]), reverse=True)

    return JsonResponse({
        "dataset_name": dataset.name,
        "model_used": dataset.anomaly_model,
        "total_points": total_count,
        "anomaly_count": len(anomaly_points),
        "anomaly_percent": round((len(anomaly_points) / total_count) * 100, 2) if total_count else 0,
        "mean_value": round(mean_val, 2),
        "std_value": round(std_val, 2),
        "chart": chart_data,
        "anomalies": anomaly_details,
    })


@login_required
def forecast_detail_view(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    return render(request, "forecast_detail.html", {"dataset": dataset})


@login_required
def forecast_data_api(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)

    history = list(
        DataPoint.objects.filter(dataset=dataset).order_by("-timestamp")[:100]
        .values("timestamp", "value")
    )
    history.reverse()

    forecasts = list(
        Forecast.objects.filter(dataset=dataset).order_by("timestamp")
        .values("timestamp", "predicted_value", "lower_bound", "upper_bound")
    )

    if not forecasts:
        return JsonResponse({"error": "No forecast has been generated yet for this dataset."}, status=404)

    return JsonResponse({
        "dataset_name": dataset.name,
        "model_used": dataset.forecast_model,
        "forecast_count": len(forecasts),
        "mae": dataset.forecast_mae,
        "rmse": dataset.forecast_rmse,
        "mape": dataset.forecast_mape,
        "history": [{"timestamp": p["timestamp"].isoformat(), "value": p["value"]} for p in history],
        "forecasts": [
            {
                "timestamp": f["timestamp"].isoformat(),
                "predicted_value": round(f["predicted_value"], 3),
                "lower_bound": round(f["lower_bound"], 3) if f["lower_bound"] is not None else None,
                "upper_bound": round(f["upper_bound"], 3) if f["upper_bound"] is not None else None,
            }
            for f in forecasts
        ],
    })


# ═══════════════════════════════════════════════════════════════
# MACHINE / IoT API VIEWS
# ═══════════════════════════════════════════════════════════════

import json
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone


# ─────────────────────────────────────────────
# Machine CRUD
# ─────────────────────────────────────────────

@login_required
def machine_list_view(request):
    """
    GET  — list all machines owned by the user
    POST — create a new machine (PC, etc.)
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
        except Exception:
            data = request.POST

        name = data.get("name", "").strip()
        if not name:
            return JsonResponse({"error": "Machine name is required."}, status=400)

        machine = Machine.objects.create(
            owner=request.user,
            name=name,
            location=data.get("location", ""),
            machine_type=data.get("machine_type", "pc"),
            description=data.get("description", ""),
        )
        return JsonResponse(_machine_to_dict(machine), status=201)

    machines = Machine.objects.filter(owner=request.user)
    return JsonResponse({"machines": [_machine_to_dict(m) for m in machines]})


@login_required
def machine_detail_view(request, machine_id):
    """GET — get machine details, latest reading, and health summary."""
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)
    latest = MachineReading.objects.filter(machine=machine).order_by("-timestamp").first()
    event_count = DowntimeEvent.objects.filter(machine=machine).count()
    open_events = DowntimeEvent.objects.filter(machine=machine, end_time__isnull=True).count()

    data = _machine_to_dict(machine)
    data["latest_reading"] = _reading_to_dict(latest) if latest else None
    data["total_downtime_events"] = event_count
    data["open_downtime_events"] = open_events
    data["total_readings"] = MachineReading.objects.filter(machine=machine).count()
    return JsonResponse(data)


def _machine_to_dict(m):
    return {
        "id": str(m.id),
        "name": m.name,
        "location": m.location,
        "machine_type": m.machine_type,
        "description": m.description,
        "is_active": m.is_active,
        "created_at": m.created_at.isoformat(),
    }


def _reading_to_dict(r):
    return {
        "id": r.id,
        "timestamp": r.timestamp.isoformat(),
        "cpu_usage_percent": r.cpu_usage_percent,
        "cpu_temp_celsius": r.cpu_temp_celsius,
        "ram_usage_percent": r.ram_usage_percent,
        "disk_read_mbps": r.disk_read_mbps,
        "disk_write_mbps": r.disk_write_mbps,
        "net_sent_mbps": r.net_sent_mbps,
        "net_recv_mbps": r.net_recv_mbps,
        "process_count": r.process_count,
        "battery_percent": r.battery_percent,
        "estimated_power_watts": r.estimated_power_watts,
        "is_anomaly": r.is_anomaly,
        "anomaly_score": r.anomaly_score,
    }


# ─────────────────────────────────────────────
# Live Reading Ingestion (used by collector script)
# ─────────────────────────────────────────────

@csrf_exempt
@login_required
def machine_readings_view(request, machine_id):
    """
    POST — ingest one or more sensor readings (JSON body).
            Body: {"readings": [{...}, ...]}  or single dict {timestamp, cpu_usage_percent, ...}
    GET  — retrieve the last N readings (default 100) for charting.
    """
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)

    if request.method == "POST":
        try:
            body = json.loads(request.body)
        except Exception:
            return JsonResponse({"error": "Invalid JSON body."}, status=400)

        # Accept single reading or list
        if isinstance(body, list):
            readings = body
        elif "readings" in body:
            readings = body["readings"]
        else:
            readings = [body]

        result = ingest_machine_readings_task(str(machine.id), readings)
        return JsonResponse({"status": "ok", "inserted": result.get("inserted", 0)})

    # GET — return recent readings
    limit = min(int(request.GET.get("limit", 200)), 2000)
    qs = MachineReading.objects.filter(machine=machine).order_by("-timestamp")[:limit]
    readings_list = [_reading_to_dict(r) for r in reversed(list(qs))]
    return JsonResponse({"machine_id": str(machine.id), "readings": readings_list})


# ─────────────────────────────────────────────
# Anomaly Detection
# ─────────────────────────────────────────────

@login_required
def machine_detect_anomalies_view(request, machine_id):
    """POST — trigger anomaly detection on the last 24h of readings."""
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)

    contamination = float(request.POST.get("contamination", 0.05))
    lookback_hours = int(request.POST.get("lookback_hours", 24))

    result = detect_machine_anomalies_task.delay(
        str(machine.id), contamination=contamination, lookback_hours=lookback_hours
    )
    return JsonResponse({
        "status": "queued",
        "machine_id": str(machine.id),
        "contamination": contamination,
        "lookback_hours": lookback_hours,
    })


# ─────────────────────────────────────────────
# Downtime Events
# ─────────────────────────────────────────────

@login_required
def machine_downtime_list_view(request, machine_id):
    """GET — list downtime events for a machine."""
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)
    events = DowntimeEvent.objects.filter(machine=machine).order_by("-start_time")

    return JsonResponse({
        "machine_id": str(machine.id),
        "total": events.count(),
        "events": [_event_to_dict(e) for e in events[:50]],
    })


def _event_to_dict(e):
    return {
        "id": str(e.id),
        "start_time": e.start_time.isoformat(),
        "end_time": e.end_time.isoformat() if e.end_time else None,
        "severity": e.severity,
        "rca_status": e.rca_status,
        "has_rca": e.rca_results is not None,
        "detected_at": e.detected_at.isoformat(),
    }


# ─────────────────────────────────────────────
# Root Cause Analysis
# ─────────────────────────────────────────────

@login_required
def machine_rca_view(request, machine_id, event_id):
    """
    GET  — return RCA results for an event (if ready).
    POST — trigger RCA for an event.
    """
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)
    event = get_object_or_404(DowntimeEvent, id=event_id, machine=machine)

    if request.method == "POST":
        pre_event_minutes = int(request.POST.get("pre_event_minutes", 30))
        root_cause_analysis_task.delay(str(machine.id), str(event.id), pre_event_minutes=pre_event_minutes)
        return JsonResponse({
            "status": "queued",
            "event_id": str(event.id),
            "rca_status": "processing",
        })

    # GET
    event.refresh_from_db()
    return JsonResponse({
        "event_id": str(event.id),
        "machine_id": str(machine.id),
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat() if event.end_time else None,
        "severity": event.severity,
        "rca_status": event.rca_status,
        "rca_results": event.rca_results,
    })


# ─────────────────────────────────────────────
# Energy Forecasting
# ─────────────────────────────────────────────

@login_required
def machine_energy_forecast_view(request, machine_id):
    """
    POST — trigger energy forecast (steps=24 by default).
    GET  — return the latest energy forecast results.
    """
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)

    if request.method == "POST":
        steps = int(request.POST.get("steps", 24))
        model_name = request.POST.get("model", "prophet")
        machine_energy_forecast_task.delay(str(machine.id), steps=steps, model_name=model_name)
        return JsonResponse({"status": "queued", "steps": steps, "model": model_name})

    # GET — return stored forecast
    forecasts = list(
        EnergyForecast.objects.filter(machine=machine).order_by("timestamp")
        .values("timestamp", "predicted_watts", "lower_bound", "upper_bound", "mae", "rmse", "mape")
    )
    if not forecasts:
        return JsonResponse({"error": "No energy forecast generated yet."}, status=404)

    # Also include the last 50 actual readings for chart overlay
    history = list(
        MachineReading.objects.filter(machine=machine)
        .order_by("-timestamp")[:50]
        .values("timestamp", "estimated_power_watts")
    )
    history.reverse()

    return JsonResponse({
        "machine_id": str(machine.id),
        "machine_name": machine.name,
        "mae": forecasts[0]["mae"],
        "rmse": forecasts[0]["rmse"],
        "mape": forecasts[0]["mape"],
        "forecast_count": len(forecasts),
        "forecasts": [
            {
                "timestamp": f["timestamp"].isoformat(),
                "predicted_watts": f["predicted_watts"],
                "lower_bound": f["lower_bound"],
                "upper_bound": f["upper_bound"],
            }
            for f in forecasts
        ],
        "history": [
            {"timestamp": h["timestamp"].isoformat(), "watts": h["estimated_power_watts"]}
            for h in history
        ],
    })


# ─────────────────────────────────────────────
# Energy Optimization
# ─────────────────────────────────────────────

@login_required
def machine_optimization_view(request, machine_id):
    """
    POST — run optimization analysis (lookback_hours default 72).
    GET  — run optimization on the spot (synchronous, since it's fast).
    """
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)
    lookback_hours = int(request.GET.get("lookback_hours", request.POST.get("lookback_hours", 72)))

    result = energy_optimization_task(str(machine.id), lookback_hours=lookback_hours)
    result["machine_id"] = str(machine.id)
    result["machine_name"] = machine.name
    return JsonResponse(result)


# ─────────────────────────────────────────────
# Live Dashboard Summary
# ─────────────────────────────────────────────

@login_required
def machine_dashboard_view(request, machine_id):
    """
    GET — one-shot endpoint returning everything needed for the live dashboard:
          latest reading, last 60 readings chart, open downtime events,
          and basic health indicators.
    """
    machine = get_object_or_404(Machine, id=machine_id, owner=request.user)

    # Latest reading
    latest = MachineReading.objects.filter(machine=machine).order_by("-timestamp").first()

    # Last 60 readings for sparkline charts
    recent_qs = list(
        MachineReading.objects.filter(machine=machine).order_by("-timestamp")[:60]
        .values("timestamp", "cpu_usage_percent", "estimated_power_watts",
                "cpu_temp_celsius", "ram_usage_percent", "is_anomaly")
    )
    recent_qs.reverse()

    # Open downtime events (no end_time yet)
    open_events = list(
        DowntimeEvent.objects.filter(machine=machine, end_time__isnull=True)
        .order_by("-start_time")[:5]
        .values("id", "start_time", "severity", "rca_status")
    )

    # Latest energy forecast summary
    latest_forecast = EnergyForecast.objects.filter(machine=machine).order_by("timestamp").first()

    # Anomaly count in last 24h
    since_24h = timezone.now() - timezone.timedelta(hours=24)
    anomaly_count_24h = MachineReading.objects.filter(
        machine=machine, timestamp__gte=since_24h, is_anomaly=True
    ).count()

    return JsonResponse({
        "machine": _machine_to_dict(machine),
        "latest_reading": _reading_to_dict(latest) if latest else None,
        "anomaly_count_24h": anomaly_count_24h,
        "open_downtime_events": [
            {
                "id": str(e["id"]),
                "start_time": e["start_time"].isoformat(),
                "severity": e["severity"],
                "rca_status": e["rca_status"],
            }
            for e in open_events
        ],
        "next_forecast_watts": latest_forecast.predicted_watts if latest_forecast else None,
        "chart": {
            "timestamps": [r["timestamp"].isoformat() for r in recent_qs],
            "cpu_usage": [r["cpu_usage_percent"] for r in recent_qs],
            "power_watts": [r["estimated_power_watts"] for r in recent_qs],
            "cpu_temp": [r["cpu_temp_celsius"] for r in recent_qs],
            "ram_usage": [r["ram_usage_percent"] for r in recent_qs],
            "is_anomaly": [r["is_anomaly"] for r in recent_qs],
        },
    })