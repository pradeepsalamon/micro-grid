from pydantic import BaseModel


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
    critical_on_inverter: bool
    non_critical_on_inverter: bool
    using_inverter: bool


class PowerData(BaseModel):
    sources: Sources
    loads: Loads
    info: Info
