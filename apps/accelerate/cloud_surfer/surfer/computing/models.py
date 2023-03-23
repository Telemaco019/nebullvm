import enum
from dataclasses import dataclass
from typing import Optional


class VMProvider(str, enum.Enum):
    AZURE = "azure"
    GCP = "gcp"
    AWS = "aws"


class Accelerator(str, enum.Enum):
    NVIDIA_TESLA_V100 = "V100", "NVIDIA Tesla V100"
    NVIDIA_TESLA_P100 = "P100", "NVIDIA Tesla P100"
    NVIDIA_TESLA_T4 = "T4", "NVIDIA Tesla T4"
    NVIDIA_TESLA_P4 = "P4", "NVIDIA Tesla P4"
    NVIDIA_TESLA_K80 = "K80", "NVIDIA Tesla K80"
    NVIDIA_TESLA_A100 = "A100", "NVIDIA Tesla A100"
    NVIDIA_TESLA_A10G = "A10G", "NVIDIA Tesla A10G"
    TPU_V2_8 = "TPU-v2-8", "TPU v2-8"
    TPU_V2_32 = "TPU-v2-32", "TPU v2-32"

    def __new__(cls, *values):
        obj = super().__new__(cls)
        obj._value_ = values[0]
        obj.display_name = values[1]
        return obj

    def __eq__(self, other):
        return self.value == other.value

    def is_tpu(self):
        return self.value in [
            Accelerator.TPU_V2_8.value,
            Accelerator.TPU_V2_32.value,
        ]


@dataclass(frozen=True)
class VMPricingInfo:
    currency: str
    region: str
    sku: str
    price_hr: Optional[float]
    price_hr_spot: Optional[float]
    price_hr_1yr: Optional[float]
    price_hr_3yr: Optional[float]
