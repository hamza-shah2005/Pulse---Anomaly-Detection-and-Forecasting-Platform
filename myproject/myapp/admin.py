from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import User, Dataset, DataPoint, Forecast


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    model = User

    list_display = (
        "email",
        "username",
        "first_name",
        "last_name",
        "is_staff",
        "is_active",
    )
    ordering = ("email",)

    fieldsets = UserAdmin.fieldsets + (
        (
            "Additional Information",
            {
                "fields": (
                    "created_at",
                    "updated_at",
                )
            },
        ),
    )
    readonly_fields = (
        "created_at",
        "updated_at",
    )


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "owner",
        "status",
        "row_count",
        "preprocessing_status",
        "anomaly_status",
        "anomaly_count",
        "forecast_status",
        "created_at",
    )
    list_filter = ("status", "preprocessing_status", "anomaly_status", "forecast_status", "anomaly_model", "forecast_model")
    search_fields = ("name", "owner__email", "owner__username")
    readonly_fields = ("created_at", "updated_at", "row_count", "total_rows_estimate")


@admin.register(DataPoint)
class DataPointAdmin(admin.ModelAdmin):
    list_display = ("dataset", "timestamp", "value", "is_anomaly")
    list_filter = ("is_anomaly", "dataset")
    search_fields = ("dataset__name",)
    ordering = ("-timestamp",)


@admin.register(Forecast)
class ForecastAdmin(admin.ModelAdmin):
    list_display = ("dataset", "timestamp", "predicted_value", "lower_bound", "upper_bound", "created_at")
    list_filter = ("dataset",)
    search_fields = ("dataset__name",)
    ordering = ("-timestamp",)


from .models import Machine, MachineReading, DowntimeEvent, EnergyForecast


@admin.register(Machine)
class MachineAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "machine_type", "location", "is_active", "created_at")
    list_filter = ("machine_type", "is_active")
    search_fields = ("name", "owner__email", "location")
    readonly_fields = ("created_at", "updated_at")


@admin.register(MachineReading)
class MachineReadingAdmin(admin.ModelAdmin):
    list_display = (
        "machine", "timestamp", "cpu_usage_percent", "cpu_temp_celsius",
        "estimated_power_watts", "is_anomaly"
    )
    list_filter = ("machine", "is_anomaly")
    search_fields = ("machine__name",)
    ordering = ("-timestamp",)
    readonly_fields = ("timestamp",)


@admin.register(DowntimeEvent)
class DowntimeEventAdmin(admin.ModelAdmin):
    list_display = ("machine", "start_time", "end_time", "severity", "rca_status", "detected_at")
    list_filter = ("machine", "severity", "rca_status")
    search_fields = ("machine__name",)
    ordering = ("-start_time",)
    readonly_fields = ("detected_at", "created_at")


@admin.register(EnergyForecast)
class EnergyForecastAdmin(admin.ModelAdmin):
    list_display = ("machine", "timestamp", "predicted_watts", "lower_bound", "upper_bound", "mae", "created_at")
    list_filter = ("machine",)
    search_fields = ("machine__name",)
    ordering = ("timestamp",)