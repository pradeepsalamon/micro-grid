from pydantic import BaseModel, Field, field_validator


class Sources(BaseModel):
    solar_power: float
    wind_power: float
    battery_power: float
    battery_soc: float


class Loads(BaseModel):
    critical: float
    non_critical: float
    grid: float
    inverter: float
    total: float


class Info(BaseModel):
    grid_available: bool
    critical_on_inverter: bool
    non_critical_on_inverter: bool
    using_inverter: bool


class PowerData(BaseModel):
    sources: Sources
    loads: Loads
    info: Info


class PredictionRequest(BaseModel):
    solar_power_w: float = Field(..., ge=0)
    wind_power_w: float = Field(..., ge=0)
    battery_soc: float = Field(..., ge=0.0, le=1.0)
    critical_load_w: float = Field(..., ge=0)
    noncritical_load_w: float = Field(..., ge=0)
    grid_available: int = Field(...)
    solar_forecast_w: float = Field(..., ge=0)
    wind_forecast_w: float = Field(..., ge=0)
    load_forecast_w: float = Field(..., ge=0)
    power_cut_probability: float = Field(..., ge=0.0, le=1.0)

    @field_validator("grid_available")
    @classmethod
    def validate_grid_available(cls, value):
        if value not in (0, 1):
            raise ValueError("grid_available must be 0 or 1")
        return value


class PredictionResponse(BaseModel):
    action_id: int
    critical_source: str
    noncritical_source: str
    inverter_total_w: float
    overloaded: bool
    constraint_applied: bool