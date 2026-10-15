from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


def load_compose() -> dict:
    with (ROOT / "docker-compose.yml").open() as compose_file:
        return yaml.safe_load(compose_file)


def test_compose_declares_required_services_health_checks_and_persistence() -> None:
    compose = load_compose()
    services = compose["services"]

    assert {"api", "worker", "postgres", "redis", "phoenix"} <= services.keys()
    assert {"postgres_data", "redis_data", "phoenix_data", "uploads"} <= compose["volumes"].keys()

    for service_name in ("postgres", "redis", "phoenix"):
        assert "healthcheck" in services[service_name]

    assert services["api"]["ports"] == ["${APP_PORT:-8000}:8000"]
    assert services["api"]["volumes"] == ["uploads:/tmp/adaptive_rag_ingestion"]
    assert services["worker"]["volumes"] == ["uploads:/tmp/adaptive_rag_ingestion"]
    assert services["worker"]["command"] == ["python", "-m", "adaptive.ingestion.worker"]
    assert services["postgres"]["volumes"] == ["postgres_data:/var/lib/postgresql/data"]
    assert services["redis"]["volumes"] == ["redis_data:/data"]
    assert services["phoenix"]["volumes"] == ["phoenix_data:/mnt/data"]


def test_api_depends_on_healthy_infrastructure_and_has_required_environment() -> None:
    compose = load_compose()
    api = compose["services"]["api"]
    worker = compose["services"]["worker"]

    assert api["depends_on"] == {
        "postgres": {"condition": "service_healthy"},
        "redis": {"condition": "service_healthy"},
        "phoenix": {"condition": "service_healthy"},
    }
    assert {"DATABASE_URL", "REDIS_HOST", "PHOENIX_ENDPOINT", "APP_HOST", "APP_PORT"} <= api["environment"].keys()
    assert api["environment"]["PHOENIX_API_BASE_URL"] == "${PHOENIX_API_BASE_URL:-http://phoenix:6006}"
    assert api["environment"]["PHOENIX_ENDPOINT"] == "${PHOENIX_ENDPOINT:-http://phoenix:6006/v1/traces}"
    assert api["environment"]["PHOENIX_CAPTURE_CONTENT"] == "${PHOENIX_CAPTURE_CONTENT:-false}"

    assert worker["depends_on"] == {
        "postgres": {"condition": "service_healthy"},
        "redis": {"condition": "service_healthy"},
        "phoenix": {"condition": "service_healthy"},
    }
    assert {"DATABASE_URL", "INGESTION_TEMP_DIR"} <= worker["environment"].keys()
    assert worker["build"] == api["build"]


def test_dockerfile_and_entrypoint_define_locked_non_root_startup() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text()

    assert "FROM" in dockerfile
    assert "AS builder" in dockerfile
    assert "uv.lock" in dockerfile
    assert "USER app" in dockerfile
    assert "alembic upgrade head" in entrypoint
    assert "exec uvicorn" in entrypoint
    assert 'exec "$@"' in entrypoint


def test_test_service_is_opt_in_and_uses_dev_dependencies() -> None:
    compose = load_compose()
    service = compose["services"]["test"]
    assert service["profiles"] == ["test"]
    assert service["build"]["target"] == "test"
    assert service["entrypoint"] == ["uv", "run", "--frozen", "--group", "dev", "pytest"]
