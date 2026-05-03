"""
services/influx_service.py
──────────────────────────
Production-ready InfluxDB 2.x integration for the Microgrid middleware.

Responsibilities
────────────────
* Bootstrap a singleton InfluxDB client on first import.
* Expose three domain-specific write helpers that accept Pydantic models
  and convert them to properly tagged/fielded InfluxDB Points.
* Absorb every write failure with structured logging so the FastAPI
  control-flow is never interrupted.
* Stay easy to extend: add a new ``write_*`` function following the same
  pattern for measurements like ``relay_events``, ``alerts``, or
  ``system_health``.

Environment variables (loaded from .env via python-dotenv):
    INFLUX_URL    – InfluxDB base URL  (e.g. http://localhost:8086)
    INFLUX_TOKEN  – API token with write access
    INFLUX_ORG    – Organisation name  (e.g. microgrid)
    INFLUX_BUCKET – Target bucket      (default: microgrid)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ── Local imports (resolved relative to app/ on the Python path) ──────────────
from models import PowerData, PredictionRequest, PredictionResponse

# ── Load .env so this module works when imported before FastAPI starts ─────────
load_dotenv()

# ── Structured logger ─────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── InfluxDB configuration ────────────────────────────────────────────────────
INFLUX_URL: str    = os.getenv("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN: str  = os.getenv("INFLUX_TOKEN",  "")
INFLUX_ORG: str    = os.getenv("INFLUX_ORG",    "microgrid")
INFLUX_BUCKET: str = os.getenv("INFLUX_BUCKET", "microgrid")

# ── Measurement names ─────────────────────────────────────────────────────────
MEASUREMENT_POWER_DATA:          str = "power_data"
MEASUREMENT_PREDICTION_REQUEST:  str = "prediction_request"
MEASUREMENT_PREDICTION_RESPONSE: str = "prediction_response"


# ══════════════════════════════════════════════════════════════════════════════
# Singleton client management
# ══════════════════════════════════════════════════════════════════════════════

_client:    Optional[InfluxDBClient] = None
_write_api  = None   # influxdb_client WriteApi (SYNCHRONOUS)


def _get_write_api():
    """
    Return a shared synchronous WriteApi, initialising the InfluxDB client
    on the first call (singleton pattern).

    Returns ``None`` when the token is missing or the client cannot be
    created, so callers can short-circuit without raising an exception.
    """
    global _client, _write_api

    if _write_api is not None:
        return _write_api

    if not INFLUX_TOKEN:
        logger.warning(
            "InfluxDB integration disabled: INFLUX_TOKEN is not set. "
            "Writes will be silently skipped."
        )
        return None

    try:
        logger.info(
            "Initialising InfluxDB client → url=%s  org=%s  bucket=%s",
            INFLUX_URL, INFLUX_ORG, INFLUX_BUCKET,
        )
        _client   = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        _write_api = _client.write_api(write_options=SYNCHRONOUS)
        logger.info("InfluxDB client ready.")
    except Exception as exc:
        logger.error("Failed to initialise InfluxDB client: %s", exc, exc_info=True)
        _client   = None
        _write_api = None

    return _write_api


def close() -> None:
    """
    Gracefully close the InfluxDB client.

    Call this from a FastAPI ``shutdown`` lifespan handler to flush
    any pending writes before the process exits.
    """
    global _client, _write_api
    if _client:
        _client.close()
        logger.info("InfluxDB client closed.")
    _client   = None
    _write_api = None


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _now_ns() -> int:
    """Return the current wall-clock time in nanoseconds (UTC)."""
    return time.time_ns()


def _safe_write(point: Point, context: str) -> None:
    """
    Write a single ``Point`` to InfluxDB, swallowing every exception.

    Args:
        point:   The InfluxDB Point to write.
        context: Human-readable label used in log messages (e.g. "power_data").
    """
    api = _get_write_api()
    if api is None:
        # Token missing or client failed to initialise — already logged once.
        return

    try:
        api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        logger.debug("InfluxDB write OK  [%s]", context)
    except Exception as exc:
        logger.error(
            "InfluxDB write FAILED [%s]: %s", context, exc, exc_info=True
        )


def _build_power_data_point(data: PowerData) -> Point:
    """
    Convert a ``PowerData`` Pydantic model into an InfluxDB ``Point``.

    Measurement: ``power_data``

    Tags
    ────
    * ``grid_available``       – bool (stored as string "True"/"False")
    * ``using_inverter``       – bool

    Fields
    ──────
    All numeric sensor readings from Sources, Loads, and Info sub-models.
    """
    return (
        Point(MEASUREMENT_POWER_DATA)
        # ── Tags (low-cardinality booleans → good for filtering) ─────────────
        .tag("grid_available",  str(data.info.grid_available))
        .tag("using_inverter",  str(data.info.using_inverter))
        # ── Source fields ─────────────────────────────────────────────────────
        .field("solar_power",   float(data.sources.solar_power))
        .field("wind_power",    float(data.sources.wind_power))
        .field("battery_power", float(data.sources.battery_power))
        .field("battery_soc",   float(data.sources.battery_soc))
        # ── Load fields ───────────────────────────────────────────────────────
        .field("critical",      float(data.loads.critical))
        .field("non_critical",  float(data.loads.non_critical))
        .field("grid",          float(data.loads.grid))
        .field("inverter",      float(data.loads.inverter))
        .field("total",         float(data.loads.total))
        # ── Info / derived fields ─────────────────────────────────────────────
        .field("critical_on_inverter",     bool(data.info.critical_on_inverter))
        .field("non_critical_on_inverter", bool(data.info.non_critical_on_inverter))
        # ── Timestamp ─────────────────────────────────────────────────────────
        .time(_now_ns(), WritePrecision.NS)
    )


def _build_prediction_request_point(data: PredictionRequest) -> Point:
    """
    Convert a ``PredictionRequest`` Pydantic model into an InfluxDB ``Point``.

    Measurement: ``prediction_request``

    Tags
    ────
    * ``grid_available`` – int (0 / 1)

    Fields
    ──────
    All nine numeric features that form the RL observation vector.
    """
    return (
        Point(MEASUREMENT_PREDICTION_REQUEST)
        # ── Tag ───────────────────────────────────────────────────────────────
        .tag("grid_available",       str(data.grid_available))
        # ── Fields ────────────────────────────────────────────────────────────
        .field("solar_power_w",        float(data.solar_power_w))
        .field("wind_power_w",         float(data.wind_power_w))
        .field("battery_soc",          float(data.battery_soc))
        .field("critical_load_w",      float(data.critical_load_w))
        .field("noncritical_load_w",   float(data.noncritical_load_w))
        .field("solar_forecast_w",     float(data.solar_forecast_w))
        .field("wind_forecast_w",      float(data.wind_forecast_w))
        .field("load_forecast_w",      float(data.load_forecast_w))
        .field("power_cut_probability",float(data.power_cut_probability))
        # ── Timestamp ─────────────────────────────────────────────────────────
        .time(_now_ns(), WritePrecision.NS)
    )


def _build_prediction_response_point(data: PredictionResponse) -> Point:
    """
    Convert a ``PredictionResponse`` Pydantic model into an InfluxDB ``Point``.

    Measurement: ``prediction_response``

    Tags
    ────
    * ``action_id``          – int (the discrete action chosen by the RL agent)
    * ``critical_source``    – str (e.g. "grid", "inverter", "battery")
    * ``noncritical_source`` – str

    Fields
    ──────
    Numeric output and boolean flags from the AI decision.
    """
    return (
        Point(MEASUREMENT_PREDICTION_RESPONSE)
        # ── Tags (useful for grouping queries by action / source) ─────────────
        .tag("action_id",           str(data.action_id))
        .tag("critical_source",     data.critical_source)
        .tag("noncritical_source",  data.noncritical_source)
        # ── Fields ────────────────────────────────────────────────────────────
        .field("inverter_total_w",   float(data.inverter_total_w))
        .field("overloaded",         bool(data.overloaded))
        .field("constraint_applied", bool(data.constraint_applied))
        # ── Timestamp ─────────────────────────────────────────────────────────
        .time(_now_ns(), WritePrecision.NS)
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public write API
# ══════════════════════════════════════════════════════════════════════════════

def write_power_data(data: PowerData) -> None:
    """
    Write a real-time telemetry snapshot to InfluxDB.

    Called immediately after the ESP32 telemetry payload is validated by
    the ``/api/telemetry`` endpoint.  Failures are logged and swallowed so
    the downstream pipeline (ML forecast → Core AI) is never blocked.

    Args:
        data: Validated ``PowerData`` model from the telemetry endpoint.
    """
    point = _build_power_data_point(data)
    _safe_write(point, context=MEASUREMENT_POWER_DATA)


def write_prediction_request(data: PredictionRequest) -> None:
    """
    Persist the ML/RL model input vector to InfluxDB.

    Call this *before* dispatching the ``PredictionRequest`` to the Core AI
    service so the observation state is captured even if the AI call fails.

    Args:
        data: Validated ``PredictionRequest`` model (the RL observation).
    """
    point = _build_prediction_request_point(data)
    _safe_write(point, context=MEASUREMENT_PREDICTION_REQUEST)


def write_prediction_response(data: PredictionResponse) -> None:
    """
    Persist the AI decision output to InfluxDB.

    Call this immediately after a successful ``PredictionResponse`` is
    received from the Core AI service.

    Args:
        data: Validated ``PredictionResponse`` model (the RL action).
    """
    point = _build_prediction_response_point(data)
    _safe_write(point, context=MEASUREMENT_PREDICTION_RESPONSE)
