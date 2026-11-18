import pytest
import taichi as ti


@pytest.fixture(scope="session", autouse=True)
def taichi_cpu():
    ti.init(arch=ti.cpu, log_level=ti.WARN, random_seed=0)
    yield
