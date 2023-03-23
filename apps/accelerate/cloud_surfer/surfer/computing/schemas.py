from typing import Optional

from pydantic.main import BaseModel

from surfer.computing import VMProvider


class VMPricing(BaseModel):
    currency: str
    region: str
    price_hr: Optional[float]
    price_hr_spot: Optional[float]
    price_hr_1yr: Optional[float]
    price_hr_3yr: Optional[float]


class HardwareInfo(BaseModel):
    class Config:
        frozen = True
        extra = "forbid"

    cpu: str
    operating_system: str
    memory_gb: int
    accelerator: Optional[str]


class VMInfo(BaseModel):
    hardware_info: HardwareInfo
    sku: str
    provider: VMProvider
    pricing: Optional[VMPricing]
