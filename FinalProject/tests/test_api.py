from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def tiny_cpu_payload() -> dict:
    return {
        "device": "cpu",
        "dtype": "fp32",
        "batch": 1,
        "heads": 2,
        "head_dim": 16,
        "prompt_len": 16,
        "gen_steps": 2,
        "warmup": 0,
        "iters": 1,
        "seed": 0,
        "enable_compile": False,
        "enable_cuda_graphs": False,
        "enable_stage6": False,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_stage5_api_tiny_cpu_smoke():
    response = client.post("/benchmark/stage5", json=tiny_cpu_payload())

    assert response.status_code == 200
    data = response.json()

    assert data["stage"] == "stage5"
    assert data["device"] == "cpu"
    assert data["all_correct"] is True
    assert data["best_backend"] != "unknown"
    assert "raw" not in data


def test_stage6_api_tiny_cpu_smoke():
    response = client.post("/benchmark/stage6", json=tiny_cpu_payload())

    assert response.status_code == 200
    data = response.json()

    assert data["stage"] == "stage6"
    assert data["device"] == "cpu"
    assert data["all_correct"] is True
    assert data["best_backend"] != "unknown"

    backend_names = {backend["name"] for backend in data["backends"]}
    assert "stage6_custom_cuda" in backend_names