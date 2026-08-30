from django.urls import path
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from myapp import views

urlpatterns = [
    # JWT Auth (API)
    path("api/token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("api/register/", views.RegisterView.as_view(), name="api_register"),
    path("api/profile/", views.ProfileView.as_view(), name="api_profile"),

    # Legacy routes for JWT
    path("register/", views.RegisterView.as_view()),
    path("login/", TokenObtainPairView.as_view()),
    path("refresh/", TokenRefreshView.as_view()),
    path("profile/", views.ProfileView.as_view()),

    # Template Auth & Navigation
    path("", views.login_page, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("landing_page/", views.landing_page, name="landing_page"),
    path("register_page/", views.register, name="register"),

    # Dataset Processing Pipeline
    path("upload/", views.upload_data_view, name="upload_data_view"),
    path("dataset/<uuid:dataset_id>/status/", views.dataset_status_view, name="dataset_status"),
    path("dataset/<uuid:dataset_id>/preprocess/", views.run_preprocessing_view, name="run_preprocessing"),
    path("dataset/<uuid:dataset_id>/model-settings/", views.update_model_settings_view, name="model_settings"),
    path("dataset/<uuid:dataset_id>/detect-anomalies/", views.run_anomaly_detection_view, name="run_anomaly_detection"),
    path("dataset/<uuid:dataset_id>/forecast/", views.run_forecast_view, name="run_forecast"),

    # Results & Visualization APIs
    path("dataset/<uuid:dataset_id>/anomalies/", views.anomaly_detail_view, name="anomaly_detail"),
    path("dataset/<uuid:dataset_id>/anomalies/data/", views.anomaly_data_api, name="anomaly_data_api"),
    path("dataset/<uuid:dataset_id>/forecast/results/", views.forecast_detail_view, name="forecast_detail"),
    path("dataset/<uuid:dataset_id>/forecast/data/", views.forecast_data_api, name="forecast_data_api"),

    # ── Machine / IoT API ──────────────────────────────────────────────
    path("api/machines/", views.machine_list_view, name="machine_list"),
    path("api/machines/<uuid:machine_id>/", views.machine_detail_view, name="machine_detail"),
    path("api/machines/<uuid:machine_id>/readings/", views.machine_readings_view, name="machine_readings"),
    path("api/machines/<uuid:machine_id>/detect-anomalies/", views.machine_detect_anomalies_view, name="machine_detect_anomalies"),
    path("api/machines/<uuid:machine_id>/downtimes/", views.machine_downtime_list_view, name="machine_downtime_list"),
    path("api/machines/<uuid:machine_id>/downtimes/<uuid:event_id>/rca/", views.machine_rca_view, name="machine_rca"),
    path("api/machines/<uuid:machine_id>/energy-forecast/", views.machine_energy_forecast_view, name="machine_energy_forecast"),
    path("api/machines/<uuid:machine_id>/optimization/", views.machine_optimization_view, name="machine_optimization"),
    path("api/machines/<uuid:machine_id>/dashboard/", views.machine_dashboard_view, name="machine_dashboard"),
]