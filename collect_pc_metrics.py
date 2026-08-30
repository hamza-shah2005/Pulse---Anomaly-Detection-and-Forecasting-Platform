"""
collect_pc_metrics.py
─────────────────────────────────────────────────────────────────────────────
PC Sensor Collector — posts real system metrics to the Pulse Analytics backend.

Usage (run from repo root):
    python collect_pc_metrics.py --url http://127.0.0.1:8000 \
                                  --machine-id <UUID> \
                                  --email user@example.com --password secret \
                                  --interval 30

Requirements:
    pip install psutil requests

Optional (Windows CPU temperature via OpenHardwareMonitor):
    pip install wmi pywin32
    (also need to run OpenHardwareMonitor.exe as administrator first)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import psutil
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("collector")

# Power estimation constants — adjust for your CPU
IDLE_POWER_WATTS = 10.0   # approximate idle system power (W)
CPU_TDP_WATTS    = 65.0   # your CPU TDP (W) — change this


def estimate_power(cpu_pct: float, ram_pct: float) -> float:
    cpu_fraction = max(0.0, min(1.0, cpu_pct / 100.0))
    ram_overhead = ram_pct * 0.02
    return round(IDLE_POWER_WATTS + cpu_fraction * (CPU_TDP_WATTS - IDLE_POWER_WATTS) + ram_overhead, 2)


def get_cpu_temperature():
    # Linux / macOS
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for key in ("coretemp", "cpu_thermal", "k10temp", "acpitz"):
                if key in temps and temps[key]:
                    return round(temps[key][0].current, 1)
            first_key = next(iter(temps))
            if temps[first_key]:
                return round(temps[first_key][0].current, 1)
    except (AttributeError, NotImplementedError):
        pass

    # Windows via WMI + OpenHardwareMonitor (run OHM as admin first)
    try:
        import wmi
        w = wmi.WMI(namespace="root\\OpenHardwareMonitor")
        sensors = w.Sensor()
        cpu_temps = [float(s.Value) for s in sensors if s.SensorType == "Temperature" and "CPU" in s.Name]
        if cpu_temps:
            return round(max(cpu_temps), 1)
    except Exception:
        pass

    return None


def collect_reading() -> dict:
    now_utc = datetime.now(timezone.utc).isoformat()
    cpu_pct = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    ram_pct = ram.percent

    disk_io  = psutil.disk_io_counters()
    net_io   = psutil.net_io_counters()
    time.sleep(1)
    disk_io2 = psutil.disk_io_counters()
    net_io2  = psutil.net_io_counters()

    disk_read_mbps  = round((disk_io2.read_bytes  - disk_io.read_bytes)  / 1_000_000, 4)
    disk_write_mbps = round((disk_io2.write_bytes - disk_io.write_bytes) / 1_000_000, 4)
    net_sent_mbps   = round((net_io2.bytes_sent   - net_io.bytes_sent)   / 1_000_000, 4)
    net_recv_mbps   = round((net_io2.bytes_recv   - net_io.bytes_recv)   / 1_000_000, 4)

    battery_pct = None
    try:
        batt = psutil.sensors_battery()
        if batt:
            battery_pct = round(batt.percent, 1)
    except Exception:
        pass

    return {
        "timestamp": now_utc,
        "cpu_usage_percent": cpu_pct,
        "cpu_temp_celsius": get_cpu_temperature(),
        "ram_usage_percent": ram_pct,
        "disk_read_mbps": disk_read_mbps,
        "disk_write_mbps": disk_write_mbps,
        "net_sent_mbps": net_sent_mbps,
        "net_recv_mbps": net_recv_mbps,
        "process_count": len(psutil.pids()),
        "battery_percent": battery_pct,
        "estimated_power_watts": estimate_power(cpu_pct, ram_pct),
    }


def post_readings(session, base_url, machine_id, readings):
    url = f"{base_url.rstrip('/')}/api/machines/{machine_id}/readings/"
    try:
        resp = session.post(url, json={"readings": readings}, timeout=15)
        if resp.status_code == 200:
            log.info(f"Sent {resp.json().get('inserted', '?')} readings.")
            return True
        log.warning(f"Server returned {resp.status_code}: {resp.text[:200]}")
    except requests.RequestException as e:
        log.error(f"Network error: {e}")
    return False


def get_jwt_session(base_url, email, password):
    session = requests.Session()
    resp = session.post(f"{base_url.rstrip('/')}/api/token/", json={"email": email, "password": password}, timeout=10)
    if resp.status_code != 200:
        log.error(f"Login failed ({resp.status_code}): {resp.text}")
        sys.exit(1)
    token = resp.json().get("access")
    session.headers.update({"Authorization": f"Bearer {token}"})
    log.info("Authenticated with JWT.")
    return session


def main():
    parser = argparse.ArgumentParser(description="Pulse PC Sensor Collector")
    parser.add_argument("--url",        default="http://127.0.0.1:8000")
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--email",      default=None)
    parser.add_argument("--password",   default=None)
    parser.add_argument("--token",      default=None)
    parser.add_argument("--interval",   type=int,   default=30)
    parser.add_argument("--batch",      type=int,   default=1)
    parser.add_argument("--idle-power", type=float, default=IDLE_POWER_WATTS)
    parser.add_argument("--tdp",        type=float, default=CPU_TDP_WATTS)
    args = parser.parse_args()

    global IDLE_POWER_WATTS, CPU_TDP_WATTS
    IDLE_POWER_WATTS = args.idle_power
    CPU_TDP_WATTS    = args.tdp

    session = requests.Session()
    if args.token:
        session.headers.update({"Authorization": f"Bearer {args.token}"})
        log.info("Using provided JWT token.")
    elif args.email and args.password:
        session = get_jwt_session(args.url, args.email, args.password)
    else:
        log.error("Provide --token OR (--email AND --password).")
        sys.exit(1)

    log.info(f"Collecting for machine {args.machine_id} every {args.interval}s. Press Ctrl+C to stop.")
    batch = []
    while True:
        try:
            r = collect_reading()
            batch.append(r)
            log.info(
                f"CPU={r['cpu_usage_percent']:.1f}%  "
                f"Temp={r['cpu_temp_celsius'] or 'N/A'}C  "
                f"RAM={r['ram_usage_percent']:.1f}%  "
                f"Power={r['estimated_power_watts']:.1f}W"
            )
            if len(batch) >= args.batch:
                post_readings(session, args.url, args.machine_id, batch)
                batch = []
            time.sleep(max(1, args.interval - 2))
        except KeyboardInterrupt:
            log.info("Stopped. Flushing remaining readings...")
            if batch:
                post_readings(session, args.url, args.machine_id, batch)
            break
        except Exception as e:
            log.error(f"Error: {e}. Retrying in {args.interval}s...")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
