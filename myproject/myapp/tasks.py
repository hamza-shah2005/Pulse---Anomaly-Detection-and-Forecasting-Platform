import numpy as np
import pandas as pd
from celery import shared_task
from django.db import transaction

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

from .models import Dataset, DataPoint, Forecast, Machine, MachineReading, DowntimeEvent, EnergyForecast

CHUNK_SIZE = 50_000     # rows read from disk per pandas chunk
BATCH_SIZE = 10_000     # rows per bulk_create call


# ─────────────────────────────────────────────
# CSV INGESTION
# ─────────────────────────────────────────────

@shared_task(bind=True)
def ingest_csv_task(self, dataset_id, value_column=None):
    """
    Ingests time-series data from an uploaded CSV file.
    Supports flexible timestamp column naming ('timestamp', 'datetime', 'Date'+'Time')
    and dynamic numeric target column detection.
    """
    dataset = Dataset.objects.get(id=dataset_id)
    dataset.status = "processing"
    dataset.save(update_fields=["status"])

    try:
        total_inserted = 0
        target_col = value_column or dataset.target_column or None

        # Check if source_file exists
        if not dataset.source_file:
            raise ValueError("No source file attached to dataset.")

        file_path = dataset.source_file.path

        for chunk in pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            sep=None,              # auto-detect delimiter (comma, semicolon, tab)
            engine="python",
        ):
            # Normalize column names for comparison
            col_map = {c: str(c).strip() for c in chunk.columns}
            chunk = chunk.rename(columns=col_map)
            lower_cols = {str(c).lower(): c for c in chunk.columns}

            # 1. Detect and parse timestamp
            if "date" in lower_cols and "time" in lower_cols:
                d_col, t_col = lower_cols["date"], lower_cols["time"]
                chunk["_parsed_ts"] = pd.to_datetime(
                    chunk[d_col].astype(str) + " " + chunk[t_col].astype(str),
                    dayfirst=True, errors="coerce"
                )
            elif "timestamp" in lower_cols:
                chunk["_parsed_ts"] = pd.to_datetime(chunk[lower_cols["timestamp"]], errors="coerce")
            elif "datetime" in lower_cols:
                chunk["_parsed_ts"] = pd.to_datetime(chunk[lower_cols["datetime"]], errors="coerce")
            elif "date" in lower_cols:
                chunk["_parsed_ts"] = pd.to_datetime(chunk[lower_cols["date"]], errors="coerce")
            elif "time" in lower_cols:
                chunk["_parsed_ts"] = pd.to_datetime(chunk[lower_cols["time"]], errors="coerce")
            else:
                # Try parsing first column as datetime
                first_col = chunk.columns[0]
                chunk["_parsed_ts"] = pd.to_datetime(chunk[first_col], errors="coerce")
                if chunk["_parsed_ts"].isna().all():
                    raise ValueError(
                        f"Could not identify a timestamp column in CSV. Found columns: {list(chunk.columns)}"
                    )

            # Localize timezone to UTC
            if chunk["_parsed_ts"].dt.tz is None:
                chunk["_parsed_ts"] = chunk["_parsed_ts"].dt.tz_localize("UTC", ambiguous="coerce", nonexistent="shift_forward")
            else:
                chunk["_parsed_ts"] = chunk["_parsed_ts"].dt.tz_convert("UTC")

            # 2. Detect target value column
            if not target_col or target_col not in chunk.columns:
                # Check known common columns or first numeric column
                common_names = ["value", "Global_active_power", "target", "close", "price", "power", "measurement"]
                found_target = None
                for name in common_names:
                    if name in chunk.columns:
                        found_target = name
                        break
                    elif name.lower() in lower_cols:
                        found_target = lower_cols[name.lower()]
                        break

                if not found_target:
                    # Find first numeric column that is not the timestamp
                    for c in chunk.columns:
                        if c != "_parsed_ts" and c not in [lower_cols.get("date"), lower_cols.get("time"), lower_cols.get("timestamp")]:
                            converted = pd.to_numeric(chunk[c], errors="coerce")
                            if converted.notna().sum() > 0:
                                found_target = c
                                break

                if not found_target:
                    raise ValueError(
                        f"No numeric value column found in CSV. Available columns: {list(chunk.columns)}"
                    )

                target_col = found_target
                dataset.target_column = target_col
                dataset.save(update_fields=["target_column"])

            chunk["_parsed_val"] = pd.to_numeric(chunk[target_col], errors="coerce")

            # 3. Create DataPoint records
            batch = []
            for row in chunk[["_parsed_ts", "_parsed_val"]].itertuples(index=False):
                ts, val = row[0], row[1]
                if pd.isna(ts) or pd.isna(val):
                    continue
                batch.append(DataPoint(dataset=dataset, timestamp=ts, value=float(val)))

                if len(batch) >= BATCH_SIZE:
                    with transaction.atomic():
                        DataPoint.objects.bulk_create(batch, ignore_conflicts=True, batch_size=BATCH_SIZE)
                    total_inserted += len(batch)
                    batch = []

            if batch:
                with transaction.atomic():
                    DataPoint.objects.bulk_create(batch, ignore_conflicts=True, batch_size=BATCH_SIZE)
                total_inserted += len(batch)

        dataset.row_count = total_inserted
        dataset.status = "ready"
        dataset.error_message = None
        dataset.save(update_fields=["row_count", "status", "error_message"])

    except Exception as e:
        dataset.status = "failed"
        dataset.error_message = str(e)
        dataset.save(update_fields=["status", "error_message"])
        raise


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

@shared_task(bind=True)
def preprocess_dataset_task(self, dataset_id):
    """
    Cleans ingested time series data:
    1. Removes duplicate timestamps (keeps first).
    2. Identifies regular time steps and interpolates missing values/gaps.
    3. Saves summary metrics to Dataset model.
    """
    dataset = Dataset.objects.get(id=dataset_id)
    dataset.preprocessing_status = "processing"
    dataset.save(update_fields=["preprocessing_status"])

    try:
        qs = DataPoint.objects.filter(dataset=dataset).order_by("timestamp").values("id", "timestamp", "value")
        df = pd.DataFrame(list(qs))

        if len(df) < 2:
            raise ValueError("Not enough data points to preprocess (minimum 2 required).")

        # 1. Remove duplicate timestamps
        before_dedup = len(df)
        dup_mask = df.duplicated(subset="timestamp", keep="first")
        duplicate_ids = df.loc[dup_mask, "id"].tolist()
        df = df.loc[~dup_mask].copy()
        duplicates_removed = before_dedup - len(df)

        if duplicate_ids:
            with transaction.atomic():
                for i in range(0, len(duplicate_ids), 5000):
                    DataPoint.objects.filter(id__in=duplicate_ids[i:i+5000]).delete()

        # 2. Resample / Interpolate missing time gaps
        df = df.sort_values("timestamp")
        time_diffs = df["timestamp"].diff().dropna()
        median_step = time_diffs.median() if len(time_diffs) and pd.notna(time_diffs.median()) else None

        missing_filled = 0
        if median_step and median_step > pd.Timedelta(0):
            # Check for gaps between min and max timestamp
            min_ts = df["timestamp"].min()
            max_ts = df["timestamp"].max()
            full_index = pd.date_range(start=min_ts, end=max_ts, freq=median_step)

            if len(full_index) > len(df):
                df_indexed = df.set_index("timestamp").reindex(full_index)
                missing_before = int(df_indexed["value"].isna().sum())
                df_indexed["value"] = df_indexed["value"].interpolate(method="linear", limit_direction="both").bfill().ffill()

                existing_timestamps = set(df["timestamp"])
                new_points = [
                    DataPoint(dataset=dataset, timestamp=ts, value=float(val))
                    for ts, val in zip(df_indexed.index, df_indexed["value"])
                    if ts not in existing_timestamps
                ]

                if new_points:
                    with transaction.atomic():
                        DataPoint.objects.bulk_create(new_points, ignore_conflicts=True, batch_size=5000)
                    missing_filled = len(new_points)

                df = df_indexed.reset_index().rename(columns={"index": "timestamp"})

        dataset.duplicate_rows_removed = duplicates_removed
        dataset.missing_values_filled = missing_filled
        dataset.row_count = DataPoint.objects.filter(dataset=dataset).count()
        dataset.preprocessing_status = "ready"
        dataset.error_message = None
        dataset.save(update_fields=[
            "duplicate_rows_removed", "missing_values_filled", "row_count", "preprocessing_status", "error_message"
        ])

    except Exception as e:
        dataset.preprocessing_status = "failed"
        dataset.error_message = str(e)
        dataset.save(update_fields=["preprocessing_status", "error_message"])
        raise


# ─────────────────────────────────────────────
# ANOMALY DETECTION
# ─────────────────────────────────────────────

@shared_task(bind=True)
def detect_anomalies_task(self, dataset_id, model_name=None, contamination=None):
    """
    Detects anomalous points using Isolation Forest or Local Outlier Factor.
    Uses multi-feature temporal representations (raw value, rolling mean, rolling std, lag diff)
    when sufficient data points are available.
    """
    dataset = Dataset.objects.get(id=dataset_id)
    model_name = model_name or dataset.anomaly_model
    contam = contamination if contamination is not None else (dataset.contamination or 0.02)
    contam = max(0.001, min(0.5, float(contam)))

    dataset.anomaly_status = "processing"
    dataset.save(update_fields=["anomaly_status"])

    try:
        points = list(
            DataPoint.objects.filter(dataset=dataset).order_by("timestamp").values("id", "value")
        )
        if len(points) < 10:
            raise ValueError("Not enough data points to run anomaly detection (need at least 10).")

        ids = [p["id"] for p in points]
        raw_values = np.array([p["value"] for p in points], dtype=float)

        # Build feature representation
        if len(raw_values) >= 30:
            s = pd.Series(raw_values)
            window_size = max(3, min(20, len(raw_values) // 10))
            rolling_mean = s.rolling(window=window_size, min_periods=1).mean().values
            rolling_std = s.rolling(window=window_size, min_periods=1).std().fillna(0).values
            diff = s.diff().fillna(0).values

            feature_matrix = np.column_stack([raw_values, rolling_mean, rolling_std, diff])
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_matrix)
        else:
            scaled_features = raw_values.reshape(-1, 1)

        # Train model & predict
        if model_name == "isolation_forest":
            model = IsolationForest(contamination=contam, random_state=42)
            predictions = model.fit_predict(scaled_features)          # -1 = anomaly
        elif model_name == "lof":
            n_neighbors = min(20, max(2, len(points) - 1))
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contam)
            predictions = model.fit_predict(scaled_features)           # -1 = anomaly
        else:
            raise ValueError(f"Unknown anomaly model: {model_name}")

        anomaly_ids = [ids[i] for i, pred in enumerate(predictions) if pred == -1]
        anomaly_id_set = set(anomaly_ids)

        # Update database points in atomic transaction
        with transaction.atomic():
            DataPoint.objects.filter(dataset=dataset).update(is_anomaly=False)
            if anomaly_ids:
                for i in range(0, len(anomaly_ids), 5000):
                    DataPoint.objects.filter(id__in=anomaly_ids[i:i+5000]).update(is_anomaly=True)

        dataset.anomaly_count = len(anomaly_ids)
        dataset.anomaly_model = model_name
        dataset.contamination = contam
        dataset.anomaly_status = "ready"
        dataset.error_message = None
        dataset.save(update_fields=["anomaly_count", "anomaly_model", "contamination", "anomaly_status", "error_message"])

    except Exception as e:
        dataset.anomaly_status = "failed"
        dataset.error_message = str(e)
        dataset.save(update_fields=["anomaly_status", "error_message"])
        raise


# ─────────────────────────────────────────────
# FORECASTING
# ─────────────────────────────────────────────

MAX_PROPHET_HISTORY = 2000   # Cap historical points for Prophet to maintain sub-second fit
MAX_ARIMA_HISTORY = 800      # Cap historical points for ARIMA for optimal AR/MA memory and speed


@shared_task(bind=True)
def generate_forecast_task(self, dataset_id, model_name=None, steps=24):
    """
    Generates future time series forecasts with confidence bounds using Prophet or ARIMA.
    Fast single-pass execution: fits once on optimized historical context and computes
    both future forecasts and evaluation metrics (MAE, RMSE, MAPE) in sub-second time.
    """
    dataset = Dataset.objects.get(id=dataset_id)
    model_name = model_name or dataset.forecast_model
    steps = max(1, min(500, int(steps)))

    dataset.forecast_status = "processing"
    dataset.save(update_fields=["forecast_status"])

    try:
        # Load only recent necessary points from DB for high-speed inference
        max_needed = max(MAX_PROPHET_HISTORY, MAX_ARIMA_HISTORY) + steps
        qs = (
            DataPoint.objects.filter(dataset=dataset)
            .order_by("-timestamp")[:max_needed]
            .values("timestamp", "value")
        )
        points_list = list(qs)
        points_list.reverse()

        if len(points_list) < steps + 5:
            raise ValueError(f"Not enough data points for forecasting. Need at least {steps + 5}, have {len(points_list)}.")

        df = pd.DataFrame(points_list).set_index("timestamp")
        full_series = df["value"]

        time_diffs = full_series.index.to_series().diff().dropna()
        step = time_diffs.median() if len(time_diffs) and pd.notna(time_diffs.median()) else pd.Timedelta(minutes=1)

        # ── Single-pass Fit & Forecast ──
        real_predictions, metrics = _run_forecast_model(model_name, full_series, steps, step)

        forecast_objs = [
            Forecast(
                dataset=dataset,
                timestamp=p["timestamp"],
                predicted_value=p["value"],
                lower_bound=p["lower"],
                upper_bound=p["upper"],
            )
            for p in real_predictions
        ]

        with transaction.atomic():
            Forecast.objects.filter(dataset=dataset).delete()
            Forecast.objects.bulk_create(forecast_objs)

        dataset.forecast_model = model_name
        dataset.forecast_mae = metrics.get("mae")
        dataset.forecast_rmse = metrics.get("rmse")
        dataset.forecast_mape = metrics.get("mape")
        dataset.forecast_status = "ready"
        dataset.error_message = None
        dataset.save(update_fields=[
            "forecast_model", "forecast_mae", "forecast_rmse", "forecast_mape", "forecast_status", "error_message"
        ])

    except Exception as e:
        dataset.forecast_status = "failed"
        dataset.error_message = str(e)
        dataset.save(update_fields=["forecast_status", "error_message"])
        raise


def _run_forecast_model(model_name, series, steps, step):
    """
    Shared helper: fits `model_name` ('prophet' or 'arima') on `series`
    and returns `(predictions_list, metrics_dict)`.
    """
    last_timestamp = series.index[-1]
    metrics = {"mae": None, "rmse": None, "mape": None}

    if model_name == "prophet":
        from prophet import Prophet
        import logging

        # Suppress verbose prophet / cmdstanpy output
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        # Use recent context for fast and accurate trend fitting
        train_series = series.iloc[-MAX_PROPHET_HISTORY:]
        prophet_df = train_series.reset_index()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

        model = Prophet(
            daily_seasonality=len(train_series) > 50,
            weekly_seasonality=len(train_series) > 200,
            yearly_seasonality=False,
            uncertainty_samples=100,  # Fast uncertainty sampling (10x speedup)
            n_changepoints=min(15, max(5, len(train_series) // 20)),
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=steps, freq=step)
        forecast_df = model.predict(future)

        # ── Fast in-sample evaluation ──
        eval_window = min(steps, len(train_series) // 4)
        if eval_window >= 2:
            historical_forecast = forecast_df.iloc[:len(train_series)].tail(eval_window)
            actual_y = prophet_df["y"].tail(eval_window).values
            pred_y = historical_forecast["yhat"].values
            metrics = _calculate_metrics(actual_y, pred_y)

        out_forecast = forecast_df.tail(steps)
        results = [
            {
                "timestamp": row["ds"].tz_localize("UTC"),
                "value": round(float(row["yhat"]), 4),
                "lower": round(float(row["yhat_lower"]), 4),
                "upper": round(float(row["yhat_upper"]), 4),
            }
            for _, row in out_forecast.iterrows()
        ]
        return results, metrics

    elif model_name == "arima":
        import warnings

        # Use optimal recent context for autoregressive fitting
        train_series = series.iloc[-MAX_ARIMA_HISTORY:].astype(float).dropna()
        p, d, q = (1, 1, 1) if len(train_series) < 100 else (2, 1, 1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ARIMA(train_series.values, order=(p, d, q)).fit(method_kwargs={"maxiter": 50, "disp": False})
            except Exception:
                model = ARIMA(train_series.values, order=(1, 0, 0)).fit(method_kwargs={"maxiter": 30, "disp": False})

        forecast_result = model.get_forecast(steps=steps)
        mean_forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.05)

        # ── Fast in-sample evaluation from fitted values ──
        eval_window = min(steps, len(train_series) // 4)
        if eval_window >= 2 and hasattr(model, "fittedvalues"):
            actual_y = train_series.values[-eval_window:]
            pred_y = model.fittedvalues[-eval_window:]
            metrics = _calculate_metrics(actual_y, pred_y)

        results = []
        for i in range(steps):
            ts = last_timestamp + (step * (i + 1))
            mean_val = float(mean_forecast[i]) if hasattr(mean_forecast, "__getitem__") else float(mean_forecast.iloc[i])
            lower_val = float(conf_int[i, 0]) if isinstance(conf_int, np.ndarray) else float(conf_int.iloc[i, 0])
            upper_val = float(conf_int[i, 1]) if isinstance(conf_int, np.ndarray) else float(conf_int.iloc[i, 1])

            results.append({
                "timestamp": ts,
                "value": round(mean_val, 4),
                "lower": round(lower_val, 4),
                "upper": round(upper_val, 4),
            })
        return results, metrics

    else:
        raise ValueError(f"Unknown forecast model: {model_name}")


def _calculate_metrics(actual_y, pred_y):
    """Calculates MAE, RMSE, and MAPE from actual and predicted numpy arrays."""
    try:
        actual_y = np.array(actual_y, dtype=float)
        pred_y = np.array(pred_y, dtype=float)
        diff = actual_y - pred_y

        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        nonzero = actual_y != 0
        mape = float(np.mean(np.abs(diff[nonzero] / actual_y[nonzero])) * 100) if nonzero.any() else None

        return {
            "mae": round(mae, 4) if not np.isnan(mae) else None,
            "rmse": round(rmse, 4) if not np.isnan(rmse) else None,
            "mape": round(mape, 2) if (mape is not None and not np.isnan(mape)) else None,
        }
    except Exception:
        return {"mae": None, "rmse": None, "mape": None}


# ═══════════════════════════════════════════════════════════════
# INDUSTRIAL IoT / PC SENSOR PIPELINE
# ═══════════════════════════════════════════════════════════════

SENSOR_FIELDS = [
    "cpu_usage_percent",
    "cpu_temp_celsius",
    "ram_usage_percent",
    "disk_read_mbps",
    "disk_write_mbps",
    "net_sent_mbps",
    "net_recv_mbps",
    "process_count",
    "estimated_power_watts",
]

SENSOR_LABELS = {
    "cpu_usage_percent":   "CPU Usage (%)",
    "cpu_temp_celsius":    "CPU Temperature (°C)",
    "ram_usage_percent":   "RAM Usage (%)",
    "disk_read_mbps":      "Disk Read (MB/s)",
    "disk_write_mbps":     "Disk Write (MB/s)",
    "net_sent_mbps":       "Network Upload (MB/s)",
    "net_recv_mbps":       "Network Download (MB/s)",
    "process_count":       "Process Count",
    "estimated_power_watts": "Estimated Power (W)",
}


# ─────────────────────────────────────────────
# 1. INGEST MACHINE READINGS
# ─────────────────────────────────────────────

@shared_task(bind=True)
def ingest_machine_readings_task(self, machine_id, readings):
    """
    Bulk-inserts a list of sensor reading dicts into MachineReading.
    Called by the PC collector script (or the live-feed API endpoint).

    Each reading dict should contain:
        timestamp (ISO 8601 string), cpu_usage_percent, cpu_temp_celsius,
        ram_usage_percent, disk_read_mbps, disk_write_mbps, net_sent_mbps,
        net_recv_mbps, process_count, estimated_power_watts, battery_percent (optional)
    """
    machine = Machine.objects.get(id=machine_id)
    objs = []
    for r in readings:
        ts = pd.to_datetime(r.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        objs.append(MachineReading(
            machine=machine,
            timestamp=ts.to_pydatetime(),
            cpu_usage_percent=r.get("cpu_usage_percent"),
            cpu_temp_celsius=r.get("cpu_temp_celsius"),
            ram_usage_percent=r.get("ram_usage_percent"),
            disk_read_mbps=r.get("disk_read_mbps"),
            disk_write_mbps=r.get("disk_write_mbps"),
            net_sent_mbps=r.get("net_sent_mbps"),
            net_recv_mbps=r.get("net_recv_mbps"),
            process_count=r.get("process_count"),
            battery_percent=r.get("battery_percent"),
            estimated_power_watts=r.get("estimated_power_watts"),
        ))

    if objs:
        with transaction.atomic():
            MachineReading.objects.bulk_create(objs, ignore_conflicts=True, batch_size=5000)

    return {"inserted": len(objs)}


# ─────────────────────────────────────────────
# 2. DETECT MACHINE ANOMALIES
# ─────────────────────────────────────────────

@shared_task(bind=True)
def detect_machine_anomalies_task(self, machine_id, contamination=0.05, lookback_hours=24):
    """
    Runs IsolationForest on the last `lookback_hours` of MachineReadings.
    Flags anomalous rows (is_anomaly=True, anomaly_score) and auto-creates
    DowntimeEvent records for clusters of consecutive anomalies.
    """
    import warnings
    from django.utils import timezone

    machine = Machine.objects.get(id=machine_id)
    since = timezone.now() - pd.Timedelta(hours=lookback_hours)

    qs = MachineReading.objects.filter(
        machine=machine, timestamp__gte=since
    ).order_by("timestamp").values("id", "timestamp", *SENSOR_FIELDS)

    rows = list(qs)
    if len(rows) < 10:
        return {"status": "skipped", "reason": "Not enough readings (need ≥10)."}

    df = pd.DataFrame(rows)
    df = df.set_index("id")

    # Build feature matrix — drop columns with all-null values
    feature_df = df[SENSOR_FIELDS].copy().fillna(df[SENSOR_FIELDS].median())
    available_cols = [c for c in SENSOR_FIELDS if feature_df[c].notna().any()]
    X = feature_df[available_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    contam = max(0.01, min(0.5, float(contamination)))
    clf = IsolationForest(contamination=contam, random_state=42, n_estimators=100)
    preds = clf.fit_predict(X_scaled)            # -1 = anomaly
    scores = clf.decision_function(X_scaled)     # more negative = more anomalous

    anomaly_ids = [int(rid) for rid, pred in zip(df.index, preds) if pred == -1]
    anomaly_id_set = set(anomaly_ids)

    # Update is_anomaly and anomaly_score on each row
    with transaction.atomic():
        MachineReading.objects.filter(machine=machine, timestamp__gte=since).update(
            is_anomaly=False, anomaly_score=None
        )
        for rid, score in zip(df.index, scores):
            MachineReading.objects.filter(id=rid).update(
                is_anomaly=(rid in anomaly_id_set),
                anomaly_score=round(float(score), 4),
            )

    # Auto-create DowntimeEvent for consecutive anomaly clusters
    df["is_anomaly"] = [rid in anomaly_id_set for rid in df.index]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    _create_downtime_events(machine, df)

    return {
        "status": "ok",
        "total_readings": len(rows),
        "anomalies_found": len(anomaly_ids),
        "anomaly_pct": round(len(anomaly_ids) / len(rows) * 100, 2),
    }


def _create_downtime_events(machine, df):
    """
    Groups consecutive anomalous readings into DowntimeEvent windows.
    Skips windows already covered by existing DowntimeEvent records.
    """
    from django.utils import timezone

    anom_df = df[df["is_anomaly"]].copy()
    if anom_df.empty:
        return

    # Cluster consecutive anomalies (gap > 10 min → new event)
    anom_df = anom_df.sort_values("timestamp")
    anom_df["gap"] = anom_df["timestamp"].diff() > pd.Timedelta(minutes=10)
    anom_df["cluster"] = anom_df["gap"].cumsum()

    existing = set(
        DowntimeEvent.objects.filter(machine=machine).values_list("start_time", flat=True)
    )

    for _, cluster in anom_df.groupby("cluster"):
        start = cluster["timestamp"].iloc[0].to_pydatetime()
        end = cluster["timestamp"].iloc[-1].to_pydatetime()

        if start in existing:
            continue

        # Determine severity by anomaly cluster size
        n = len(cluster)
        if n >= 10:
            severity = "critical"
        elif n >= 5:
            severity = "high"
        elif n >= 2:
            severity = "medium"
        else:
            severity = "low"

        DowntimeEvent.objects.create(
            machine=machine,
            start_time=start,
            end_time=end,
            severity=severity,
            rca_status="pending",
        )


# ─────────────────────────────────────────────
# 3. ROOT CAUSE ANALYSIS
# ─────────────────────────────────────────────

@shared_task(bind=True)
def root_cause_analysis_task(self, machine_id, event_id, pre_event_minutes=30):
    """
    Analyses sensor data in the `pre_event_minutes` window before a DowntimeEvent
    to identify which sensors drifted most strongly toward the failure.

    Method:
      - Computes Pearson correlation of each sensor with 'anomaly_score' (closeness to failure)
      - Computes drift rate (slope of linear regression) for each sensor in the pre-event window
      - Combines correlation + drift into a ranked list of root causes
      - Generates human-readable recommendations
    """
    machine = Machine.objects.get(id=machine_id)
    event = DowntimeEvent.objects.get(id=event_id, machine=machine)

    event.rca_status = "processing"
    event.save(update_fields=["rca_status"])

    try:
        window_start = event.start_time - pd.Timedelta(minutes=pre_event_minutes)

        qs = MachineReading.objects.filter(
            machine=machine,
            timestamp__gte=window_start,
            timestamp__lte=event.start_time,
        ).order_by("timestamp").values("timestamp", "anomaly_score", *SENSOR_FIELDS)

        rows = list(qs)
        if len(rows) < 5:
            raise ValueError(f"Not enough readings in pre-event window ({len(rows)} found, need ≥5).")

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["t_seconds"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

        # Anomaly score: higher = more anomalous (invert IF score so -1→most anomalous)
        if "anomaly_score" in df.columns and df["anomaly_score"].notna().any():
            # IF score: more negative = more anomalous → invert
            df["risk_score"] = -df["anomaly_score"].fillna(0)
        else:
            df["risk_score"] = np.arange(len(df), dtype=float)  # fallback: time index

        causes = []
        for col in SENSOR_FIELDS:
            col_data = df[col].dropna()
            if len(col_data) < 3:
                continue

            col_filled = df[col].fillna(df[col].median())

            # Correlation with risk score
            try:
                corr = float(np.corrcoef(col_filled.values, df["risk_score"].values)[0, 1])
            except Exception:
                corr = 0.0

            # Drift slope (linear regression over time)
            try:
                t = df["t_seconds"].values
                y = col_filled.values
                slope = float(np.polyfit(t, y, 1)[0])
                # Normalise slope by the column's std so they're comparable
                col_std = float(col_filled.std()) or 1.0
                normalised_slope = slope / col_std
            except Exception:
                normalised_slope = 0.0

            # Combined score: high |corr| AND strong drift = likely cause
            combined = 0.6 * abs(corr) + 0.4 * min(1.0, abs(normalised_slope) * 10)

            # Direction of drift
            if normalised_slope > 0.001:
                direction = "rising"
            elif normalised_slope < -0.001:
                direction = "falling"
            else:
                direction = "stable"

            start_val = float(col_filled.iloc[0])
            end_val = float(col_filled.iloc[-1])
            delta = end_val - start_val

            causes.append({
                "sensor": col,
                "label": SENSOR_LABELS.get(col, col),
                "correlation_with_risk": round(corr, 3),
                "drift_direction": direction,
                "start_value": round(start_val, 2),
                "end_value": round(end_val, 2),
                "delta": round(delta, 2),
                "combined_score": round(combined, 4),
            })

        # Rank by combined score
        causes.sort(key=lambda x: x["combined_score"], reverse=True)

        # Compute contribution percentages from top-3
        top3 = causes[:3]
        total_score = sum(c["combined_score"] for c in top3) or 1.0
        for c in top3:
            c["contribution_pct"] = round(c["combined_score"] / total_score * 100, 1)

        # Generate human-readable description for each top cause
        for c in top3:
            label = c["label"]
            direction = c["drift_direction"]
            delta = c["delta"]
            c["description"] = (
                f"{label} {direction} by {abs(delta):.1f} units "
                f"in the {pre_event_minutes} minutes before the event "
                f"(correlation with risk: {c['correlation_with_risk']:.2f})."
            )

        # Generate recommendations based on which sensors are top causes
        recommendations = _generate_recommendations(top3, event.severity)

        # Build summary sentence
        if top3:
            primary = top3[0]
            summary = (
                f"Primary cause: {primary['label']} was {primary['drift_direction']} "
                f"({'+' if primary['delta']>0 else ''}{primary['delta']:.1f} units) "
                f"in the {pre_event_minutes} min before the event "
                f"with {primary['contribution_pct']}% contribution."
            )
        else:
            summary = "Insufficient sensor data to determine root cause."

        rca_results = {
            "summary": summary,
            "pre_event_window_minutes": pre_event_minutes,
            "readings_analysed": len(rows),
            "top_causes": top3,
            "all_sensor_scores": causes,
            "recommendations": recommendations,
        }

        event.rca_status = "ready"
        event.rca_results = rca_results
        event.save(update_fields=["rca_status", "rca_results"])

        return rca_results

    except Exception as e:
        event.rca_status = "failed"
        event.rca_results = {"error": str(e)}
        event.save(update_fields=["rca_status", "rca_results"])
        raise


def _generate_recommendations(top_causes, severity):
    """Generates actionable recommendations based on top sensor causes."""
    recs = []
    sensor_names = [c["sensor"] for c in top_causes]

    if "cpu_temp_celsius" in sensor_names:
        recs.append("Check CPU cooling — clean dust from heatsink/fans and reapply thermal paste if needed.")
        recs.append("Review active background processes that may be causing thermal load.")
    if "cpu_usage_percent" in sensor_names:
        recs.append("Identify and terminate high-CPU background tasks (antivirus scans, updates).")
        recs.append("Consider scheduling heavy workloads during off-peak hours.")
    if "ram_usage_percent" in sensor_names:
        recs.append("Close unused applications to free RAM. Consider upgrading RAM if consistently >85%.")
    if "disk_read_mbps" in sensor_names or "disk_write_mbps" in sensor_names:
        recs.append("High disk I/O detected — check for disk fragmentation or failing drive health (S.M.A.R.T).")
    if "estimated_power_watts" in sensor_names:
        recs.append("Reduce power draw by enabling power-saving CPU profiles during non-critical tasks.")
    if "net_sent_mbps" in sensor_names or "net_recv_mbps" in sensor_names:
        recs.append("Unusual network activity detected — check for background downloads or malware.")

    if not recs:
        recs.append("Monitor all sensor channels over the next 24 hours for recurring patterns.")

    if severity in ("high", "critical"):
        recs.insert(0, "⚠️  Severity is HIGH — consider restarting the machine or reducing workload immediately.")

    return recs


# ─────────────────────────────────────────────
# 4. ENERGY CONSUMPTION FORECASTING
# ─────────────────────────────────────────────

@shared_task(bind=True)
def machine_energy_forecast_task(self, machine_id, steps=24, model_name="prophet"):
    """
    Forecasts future energy consumption (estimated_power_watts) for a machine.
    Uses Prophet with cpu_usage_percent as an additional regressor for higher accuracy.
    Falls back to ARIMA if Prophet fit fails.

    Stores results in EnergyForecast. Previous forecasts for this machine are replaced.
    """
    import warnings
    machine = Machine.objects.get(id=machine_id)
    steps = max(1, min(500, int(steps)))

    # Fetch recent readings — need at least steps+10 points
    MAX_HISTORY = 2000
    qs = (
        MachineReading.objects.filter(machine=machine)
        .order_by("-timestamp")[:MAX_HISTORY]
        .values("timestamp", "estimated_power_watts", "cpu_usage_percent", "ram_usage_percent")
    )
    rows = list(qs)
    rows.reverse()

    if len(rows) < steps + 5:
        return {"status": "failed", "reason": f"Need at least {steps+5} readings, have {len(rows)}."}

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["estimated_power_watts"])

    if len(df) < 10:
        return {"status": "failed", "reason": "Not enough readings with valid power data."}

    time_diffs = df["timestamp"].diff().dropna()
    step = time_diffs.median() if len(time_diffs) and pd.notna(time_diffs.median()) else pd.Timedelta(minutes=1)
    last_ts = df["timestamp"].iloc[-1]

    forecast_rows = []
    metrics = {"mae": None, "rmse": None, "mape": None}

    if model_name == "prophet":
        try:
            from prophet import Prophet
            import logging
            logging.getLogger("prophet").setLevel(logging.ERROR)
            logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

            prophet_df = df[["timestamp", "estimated_power_watts", "cpu_usage_percent"]].copy()
            prophet_df.columns = ["ds", "y", "cpu_usage"]
            prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
            prophet_df["cpu_usage"] = prophet_df["cpu_usage"].fillna(prophet_df["cpu_usage"].median())

            model = Prophet(
                daily_seasonality=len(prophet_df) > 50,
                weekly_seasonality=len(prophet_df) > 200,
                yearly_seasonality=False,
                uncertainty_samples=100,
                n_changepoints=min(15, max(5, len(prophet_df) // 20)),
            )
            model.add_regressor("cpu_usage")
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=steps, freq=step)
            # Fill cpu_usage for future rows using the rolling mean of the last 10 readings
            last_cpu = prophet_df["cpu_usage"].tail(10).mean()
            future["cpu_usage"] = future["cpu_usage"].fillna(last_cpu) if "cpu_usage" in future.columns else last_cpu

            forecast_df = model.predict(future)

            # In-sample metrics
            eval_window = min(steps, len(prophet_df) // 4)
            if eval_window >= 2:
                hist_fc = forecast_df.iloc[:len(prophet_df)].tail(eval_window)
                actual = prophet_df["y"].tail(eval_window).values
                pred = hist_fc["yhat"].values
                metrics = _calculate_metrics(actual, pred)

            out_fc = forecast_df.tail(steps)
            for _, row in out_fc.iterrows():
                forecast_rows.append({
                    "timestamp": pd.Timestamp(row["ds"]).tz_localize("UTC").to_pydatetime(),
                    "predicted_watts": round(float(row["yhat"]), 2),
                    "lower_bound": round(float(row["yhat_lower"]), 2),
                    "upper_bound": round(float(row["yhat_upper"]), 2),
                })

        except Exception as prophet_err:
            # Fallback to ARIMA
            model_name = "arima"

    if model_name == "arima" or not forecast_rows:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = df["estimated_power_watts"].astype(float).dropna()
            series_arr = series.iloc[-800:].values
            p, d, q = (2, 1, 1) if len(series_arr) >= 100 else (1, 1, 0)
            try:
                arima_model = ARIMA(series_arr, order=(p, d, q)).fit(
                    method_kwargs={"maxiter": 50, "disp": False}
                )
            except Exception:
                arima_model = ARIMA(series_arr, order=(1, 0, 0)).fit(
                    method_kwargs={"maxiter": 30, "disp": False}
                )

            fc_result = arima_model.get_forecast(steps=steps)
            mean_fc = fc_result.predicted_mean
            conf = fc_result.conf_int(alpha=0.05)

            eval_window = min(steps, len(series_arr) // 4)
            if eval_window >= 2 and hasattr(arima_model, "fittedvalues"):
                actual = series_arr[-eval_window:]
                pred = arima_model.fittedvalues[-eval_window:]
                metrics = _calculate_metrics(actual, pred)

            forecast_rows = []
            for i in range(steps):
                ts = last_ts + step * (i + 1)
                mean_val = float(mean_fc[i]) if hasattr(mean_fc, "__getitem__") else float(mean_fc.iloc[i])
                lo = float(conf[i, 0]) if isinstance(conf, np.ndarray) else float(conf.iloc[i, 0])
                hi = float(conf[i, 1]) if isinstance(conf, np.ndarray) else float(conf.iloc[i, 1])
                forecast_rows.append({
                    "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                    "predicted_watts": round(mean_val, 2),
                    "lower_bound": round(lo, 2),
                    "upper_bound": round(hi, 2),
                })

    # Save forecast objects — replace old ones
    ef_objs = [
        EnergyForecast(
            machine=machine,
            timestamp=r["timestamp"],
            predicted_watts=r["predicted_watts"],
            lower_bound=r["lower_bound"],
            upper_bound=r["upper_bound"],
            mae=metrics.get("mae"),
            rmse=metrics.get("rmse"),
            mape=metrics.get("mape"),
        )
        for r in forecast_rows
    ]

    with transaction.atomic():
        EnergyForecast.objects.filter(machine=machine).delete()
        EnergyForecast.objects.bulk_create(ef_objs)

    return {
        "status": "ok",
        "model": model_name,
        "steps": steps,
        "mae": metrics.get("mae"),
        "rmse": metrics.get("rmse"),
        "mape": metrics.get("mape"),
    }


# ─────────────────────────────────────────────
# 5. ENERGY OPTIMIZATION
# ─────────────────────────────────────────────

@shared_task(bind=True)
def energy_optimization_task(self, machine_id, lookback_hours=72):
    """
    Analyses the relationship between CPU load (throughput proxy) and power draw.
    Finds the optimal CPU operating point where efficiency (output/watt) is maximised.

    Returns:
        - Efficiency curve data points (cpu% vs watts/cpu%)
        - Recommended optimal CPU load range
        - Estimated energy savings at recommended vs current average operating point
        - Pareto frontier data for throughput vs energy tradeoff chart
    """
    from django.utils import timezone
    machine = Machine.objects.get(id=machine_id)
    since = timezone.now() - pd.Timedelta(hours=lookback_hours)

    qs = MachineReading.objects.filter(
        machine=machine, timestamp__gte=since
    ).order_by("timestamp").values(
        "timestamp", "cpu_usage_percent", "estimated_power_watts",
        "cpu_temp_celsius", "ram_usage_percent"
    )
    rows = list(qs)

    if len(rows) < 20:
        return {"status": "failed", "reason": f"Need ≥20 readings for optimization. Have {len(rows)}."}

    df = pd.DataFrame(rows).dropna(subset=["cpu_usage_percent", "estimated_power_watts"])
    if len(df) < 20:
        return {"status": "failed", "reason": "Not enough readings with both CPU and power data."}

    cpu = df["cpu_usage_percent"].values
    watts = df["estimated_power_watts"].values

    # ── Efficiency curve: efficiency = cpu_output / watts ──
    # Avoid division by zero; filter near-zero watt readings
    valid = watts > 1.0
    cpu_v = cpu[valid]
    watts_v = watts[valid]

    # Bin CPU usage into 10%-wide buckets
    bins = np.arange(0, 101, 10)
    bucket_labels = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    efficiency_curve = []
    pareto_points = []

    for lo, hi in bucket_labels:
        mask = (cpu_v >= lo) & (cpu_v < hi)
        if mask.sum() < 3:
            continue
        avg_cpu = float(np.mean(cpu_v[mask]))
        avg_watts = float(np.mean(watts_v[mask]))
        efficiency = avg_cpu / avg_watts  # CPU% per watt
        efficiency_curve.append({
            "cpu_range": f"{lo}-{hi}%",
            "avg_cpu_percent": round(avg_cpu, 1),
            "avg_watts": round(avg_watts, 1),
            "efficiency_cpu_per_watt": round(efficiency, 4),
            "sample_count": int(mask.sum()),
        })
        pareto_points.append({"cpu": round(avg_cpu, 1), "watts": round(avg_watts, 1)})

    if not efficiency_curve:
        return {"status": "failed", "reason": "Could not build efficiency curve — data too uniform."}

    # ── Find optimal point: max efficiency ──
    best = max(efficiency_curve, key=lambda x: x["efficiency_cpu_per_watt"])
    current_avg_cpu = float(np.mean(cpu_v))
    current_avg_watts = float(np.mean(watts_v))

    # Estimate savings: if user moved to optimal CPU range
    best_watts = best["avg_watts"]
    savings_watts = current_avg_watts - best_watts
    savings_pct = (savings_watts / current_avg_watts * 100) if current_avg_watts > 0 else 0

    # Estimate daily kWh savings (assuming 24h operation)
    daily_kwh_savings = (savings_watts * 24) / 1000.0

    # ── Overheat risk assessment ──
    temp_data = df["cpu_temp_celsius"].dropna()
    thermal_risk = "low"
    if len(temp_data) > 0:
        p95_temp = float(np.percentile(temp_data, 95))
        if p95_temp > 90:
            thermal_risk = "critical"
        elif p95_temp > 80:
            thermal_risk = "high"
        elif p95_temp > 70:
            thermal_risk = "medium"

    # ── Correlations summary ──
    corr_cpu_watts = float(np.corrcoef(cpu_v, watts_v)[0, 1]) if len(cpu_v) > 2 else 0.0

    # ── Recommendations ──
    recommendations = []
    if savings_pct > 10:
        recommendations.append(
            f"Operating at {best['cpu_range']} CPU load gives the best efficiency "
            f"({best['efficiency_cpu_per_watt']:.4f} CPU%/W). "
            f"Reducing from current average {current_avg_cpu:.0f}% could save "
            f"~{savings_pct:.1f}% energy ({daily_kwh_savings:.3f} kWh/day)."
        )
    else:
        recommendations.append(
            f"Current operating point ({current_avg_cpu:.0f}% avg CPU) is already near optimal efficiency."
        )

    if thermal_risk in ("high", "critical"):
        recommendations.append(
            f"⚠️ CPU temperatures at or above 80°C (95th percentile: {p95_temp:.0f}°C). "
            "Thermal throttling may be reducing throughput — improve cooling before increasing load."
        )

    if corr_cpu_watts > 0.85:
        recommendations.append(
            "Strong linear relationship between CPU load and power draw (R={:.2f}). "
            "Scheduling batch workloads during off-peak hours will directly reduce peak energy cost.".format(corr_cpu_watts)
        )

    return {
        "status": "ok",
        "lookback_hours": lookback_hours,
        "readings_analysed": len(df),
        "current_avg_cpu_percent": round(current_avg_cpu, 1),
        "current_avg_watts": round(current_avg_watts, 1),
        "optimal_cpu_range": best["cpu_range"],
        "optimal_avg_watts": best["avg_watts"],
        "efficiency_at_optimal": best["efficiency_cpu_per_watt"],
        "estimated_savings_percent": round(savings_pct, 1),
        "estimated_daily_kwh_savings": round(daily_kwh_savings, 4),
        "thermal_risk": thermal_risk,
        "cpu_watts_correlation": round(corr_cpu_watts, 3),
        "efficiency_curve": efficiency_curve,
        "pareto_points": pareto_points,
        "recommendations": recommendations,
    }