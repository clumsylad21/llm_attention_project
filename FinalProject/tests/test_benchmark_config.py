import pytest

from src.benchmark.config import BenchmarkConfig


def test_benchmark_config_defaults_are_valid():
    config = BenchmarkConfig()

    assert config.device_requested == "auto"
    assert config.dtype_name == "fp32"
    assert config.batch == 1
    assert config.enable_stage6 is True


def test_benchmark_config_converts_to_stage5_kwargs_without_stage6_flag():
    config = BenchmarkConfig(enable_stage6=False)

    kwargs = config.to_stage5_kwargs()

    assert "enable_stage6" not in kwargs
    assert kwargs["device_requested"] == "auto"


def test_benchmark_config_converts_to_stage6_kwargs_with_stage6_flag():
    config = BenchmarkConfig(enable_stage6=False)

    kwargs = config.to_stage6_kwargs()

    assert kwargs["enable_stage6"] is False


def test_benchmark_config_rejects_bad_dtype():
    with pytest.raises(ValueError):
        BenchmarkConfig(dtype_name="int8")


def test_benchmark_config_rejects_invalid_sizes():
    with pytest.raises(ValueError):
        BenchmarkConfig(prompt_len=0)

    with pytest.raises(ValueError):
        BenchmarkConfig(gen_steps=0)

    with pytest.raises(ValueError):
        BenchmarkConfig(iters=0)

    with pytest.raises(ValueError):
        BenchmarkConfig(warmup=-1)