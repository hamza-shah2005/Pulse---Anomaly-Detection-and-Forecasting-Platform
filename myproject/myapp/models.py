import uuid
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=100, blank=True)
    last_name = models.CharField(max_length=100, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email


class Dataset(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("ready", "Ready"),
        ("failed", "Failed"),
    ]
    SOURCE_CHOICES = [
        ("csv", "CSV Upload"),
        ("feed", "Live Feed"),
    ]
    ANOMALY_MODEL_CHOICES = [
        ("isolation_forest", "Isolation Forest"),
        ("lof", "Local Outlier Factor"),
    ]
    FORECAST_MODEL_CHOICES = [
        ("prophet", "Prophet"),
        ("arima", "ARIMA"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="datasets",
    )
    name = models.CharField(max_length=255)
    source_file = models.FileField(upload_to="datasets/%Y/%m/%d/", blank=True, null=True)
    source_type = models.CharField(max_length=10, choices=SOURCE_CHOICES, default="csv")
    target_column = models.CharField(max_length=100, default="value", blank=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="pending")
    row_count = models.PositiveIntegerField(default=0)
    total_rows_estimate = models.PositiveIntegerField(default=0)
    error_message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Preprocessing
    preprocessing_status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="pending")
    duplicate_rows_removed = models.PositiveIntegerField(default=0)
    missing_values_filled = models.PositiveIntegerField(default=0)

    # Anomaly Detection
    anomaly_model = models.CharField(max_length=20, choices=ANOMALY_MODEL_CHOICES, default="isolation_forest")
    anomaly_status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="pending")
    anomaly_count = models.PositiveIntegerField(default=0)
    contamination = models.FloatField(default=0.02)

    # Forecasting
    forecast_model = models.CharField(max_length=20, choices=FORECAST_MODEL_CHOICES, default="prophet")
    forecast_status = models.CharField(max_length=12, choices=STATUS_CHOICES, default="pending")
    forecast_mae = models.FloatField(null=True, blank=True)
    forecast_rmse = models.FloatField(null=True, blank=True)
    forecast_mape = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["owner", "status"])]

    @property
    def progress_percent(self):
        if not self.total_rows_estimate:
            return 0
        return min(100, int((self.row_count / self.total_rows_estimate) * 100))

    def __str__(self):
        return f"{self.name} ({self.status})"


class DataPoint(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="points")
    timestamp = models.DateTimeField()
    value = models.FloatField()
    is_anomaly = models.BooleanField(default=False, db_index=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [models.Index(fields=["dataset", "timestamp"])]
        constraints = [
            models.UniqueConstraint(fields=["dataset", "timestamp"], name="unique_dataset_timestamp")
        ]

    def __str__(self):
        return f"{self.dataset_id} @ {self.timestamp} = {self.value}"


class Forecast(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="forecasts")
    timestamp = models.DateTimeField()
    predicted_value = models.FloatField()
    lower_bound = models.FloatField(null=True, blank=True)
    upper_bound = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [models.Index(fields=["dataset", "timestamp"])]

    def __str__(self):
        return f"{self.dataset_id} forecast @ {self.timestamp} = {self.predicted_value}"


# ─────────────────────────────────────────────
# INDUSTRIAL IoT / PC SENSOR MODELS
# ─────────────────────────────────────────────

class Machine(models.Model):
    """
    Represents a monitored machine — in this project, the user's own PC.
    Can be generalised to any machine with sensors.
    """
    MACHINE_TYPES = [
        ("pc", "Personal Computer"),
        ("cnc", "CNC Machine"),
        ("compressor", "Compressor"),
        ("pump", "Pump"),
        ("motor", "Electric Motor"),
        ("other", "Other"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="machines",
    )
    name = models.CharField(max_length=255)
    location = models.CharField(max_length=255, blank=True, default="")
    machine_type = models.CharField(max_length=20, choices=MACHINE_TYPES, default="pc")
    description = models.TextField(blank=True, default="")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.machine_type})"


class MachineReading(models.Model):
    """
    One sensor snapshot per timestamp for a machine.
    For a PC this maps to: CPU%, CPU temp, RAM%, disk I/O, network I/O, estimated watts.
    For an industrial machine it maps to real sensor channels.
    """
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="readings")
    timestamp = models.DateTimeField(db_index=True)

    # Core performance / load sensors
    cpu_usage_percent = models.FloatField(null=True, blank=True)
    cpu_temp_celsius = models.FloatField(null=True, blank=True)
    ram_usage_percent = models.FloatField(null=True, blank=True)

    # I/O throughput sensors
    disk_read_mbps = models.FloatField(null=True, blank=True)
    disk_write_mbps = models.FloatField(null=True, blank=True)
    net_sent_mbps = models.FloatField(null=True, blank=True)
    net_recv_mbps = models.FloatField(null=True, blank=True)

    # System-level sensors
    process_count = models.IntegerField(null=True, blank=True)
    battery_percent = models.FloatField(null=True, blank=True)   # if laptop

    # Energy (key metric)
    estimated_power_watts = models.FloatField(null=True, blank=True)

    # Anomaly flag (set by detect_machine_anomalies_task)
    is_anomaly = models.BooleanField(default=False, db_index=True)
    anomaly_score = models.FloatField(null=True, blank=True)   # raw IF/LOF score

    class Meta:
        ordering = ["timestamp"]
        indexes = [models.Index(fields=["machine", "timestamp"])]
        constraints = [
            models.UniqueConstraint(fields=["machine", "timestamp"], name="unique_machine_timestamp")
        ]

    def __str__(self):
        return f"{self.machine.name} @ {self.timestamp} | {self.estimated_power_watts}W"


class DowntimeEvent(models.Model):
    """
    A detected anomaly window — equivalent to a machine 'failure' event.
    Root Cause Analysis results are stored as JSON in rca_results.
    """
    SEVERITY_CHOICES = [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("critical", "Critical"),
    ]
    RCA_STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("ready", "Ready"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="downtime_events")
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES, default="medium")

    # Root Cause Analysis
    rca_status = models.CharField(max_length=12, choices=RCA_STATUS_CHOICES, default="pending")
    rca_results = models.JSONField(null=True, blank=True)
    # rca_results schema:
    # {
    #   "summary": "CPU temperature rose 18°C in 10 minutes before event",
    #   "top_causes": [
    #     {"sensor": "cpu_temp_celsius", "contribution_pct": 67, "direction": "rising", "description": "..."},
    #     ...
    #   ],
    #   "recommendations": ["Improve ventilation", ...]
    # }

    detected_at = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-start_time"]
        indexes = [models.Index(fields=["machine", "start_time"])]

    def __str__(self):
        return f"Downtime @ {self.machine.name} {self.start_time} [{self.severity}]"


class EnergyForecast(models.Model):
    """
    Predicted energy consumption (watts) for a machine at a future timestamp.
    Generated by energy_forecast_task.
    """
    machine = models.ForeignKey(Machine, on_delete=models.CASCADE, related_name="energy_forecasts")
    timestamp = models.DateTimeField()
    predicted_watts = models.FloatField()
    lower_bound = models.FloatField(null=True, blank=True)
    upper_bound = models.FloatField(null=True, blank=True)

    # Accuracy metrics from training evaluation
    mae = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [models.Index(fields=["machine", "timestamp"])]

    def __str__(self):
        return f"{self.machine.name} forecast @ {self.timestamp} = {self.predicted_watts}W"