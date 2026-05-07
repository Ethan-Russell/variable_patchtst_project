from .models import (
    VariablePatchAlgorithm,
    UniformPatchAlgorithm,
    get_perfect_patches,
    get_uniform_patches,
)
from .utils import (
    load_electricity_data,
    read_electricity_data,
    ElectricityLoadDataset,
    StandardScaler,
)

__all__ = [
    "hello",
    "VariablePatchAlgorithm",
    "UniformPatchAlgorithm",
    "get_perfect_patches",
    "get_uniform_patches",
    "load_electricity_data",
    "read_electricity_data",
    "ElectricityLoadDataset",
    "StandardScaler",
]

def hello() -> str:
    return "Hello from variable-patchtst-project!"
