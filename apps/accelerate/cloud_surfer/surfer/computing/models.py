import enum


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

    def __new__(cls, *values):
        obj = super().__new__(cls)
        obj._value_ = values[0]
        obj.display_name = values[1]
        return obj
