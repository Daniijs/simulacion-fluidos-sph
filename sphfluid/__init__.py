"""sphfluid: Position Based Fluids on the GPU with Taichi."""

from .config import FluidParams, OutputParams, SceneConfig, SolverParams, load_config

__all__ = ["FluidParams", "OutputParams", "SceneConfig", "SolverParams", "load_config"]
__version__ = "2.0.0"
