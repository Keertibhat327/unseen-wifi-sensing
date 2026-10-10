"""Thresholds are research parameters in units of calibrated background noise."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DetectorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    calibration_min_windows: int = Field(default=5, ge=2)
    noise_floor: float = Field(default=0.05, gt=0)
    calibration_max_drift_sigma: float = Field(default=4, gt=0)
    cleaning_window_seconds: float = Field(default=0.075, gt=0)
    outlier_sigma: float = Field(default=8, gt=0)
    coherent_carrier_fraction: float = Field(default=0.5, gt=0, le=1)
    motion_on_sigma: float = Field(default=8, gt=0)
    motion_off_sigma: float = Field(default=4, gt=0)
    presence_on_sigma: float = Field(default=8, gt=0)
    presence_off_sigma: float = Field(default=4, gt=0)
    presence_enter_seconds: float = Field(default=1, gt=0)
    vacancy_seconds: float = Field(default=8, gt=0)
    fall_impact_sigma: float = Field(default=80, gt=0)
    fall_stillness_sigma: float = Field(default=3, gt=0)
    fall_stillness_seconds: float = Field(default=2, gt=0)
    fall_timeout_seconds: float = Field(default=6, gt=0)
    fall_cooldown_seconds: float = Field(default=3, ge=0)
    gap_tolerance_seconds: float = Field(default=0.15, ge=0)

    @model_validator(mode="after")
    def thresholds(self) -> "DetectorSettings":
        if self.motion_off_sigma >= self.motion_on_sigma:
            raise ValueError("motion off threshold must be below on threshold")
        if self.presence_off_sigma >= self.presence_on_sigma:
            raise ValueError("presence off threshold must be below on threshold")
        if self.fall_stillness_sigma >= self.motion_on_sigma:
            raise ValueError("stillness threshold must be below motion on threshold")
        if self.fall_timeout_seconds < self.fall_stillness_seconds:
            raise ValueError("fall timeout must allow the stillness interval")
        return self
