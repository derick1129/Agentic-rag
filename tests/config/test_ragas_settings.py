"""Optional Ragas judge configuration contracts."""

import pytest

from adaptive.config import Settings


def test_ragas_settings_have_non_secret_defaults() -> None:
    settings = Settings(DATABASE_URL="postgresql://db")

    assert settings.ragas_judge_api_key == ""
    assert settings.ragas_judge_api_base == "https://api.groq.com/openai/v1"
    assert settings.ragas_judge_model == "openai/gpt-oss-20b"


def test_ragas_settings_accept_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGAS_JUDGE_API_KEY", "test-key")
    monkeypatch.setenv("RAGAS_JUDGE_API_BASE", "https://groq.example/v1")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-20b-test")

    settings = Settings(DATABASE_URL="postgresql://db")

    assert settings.ragas_judge_api_key == "test-key"
    assert settings.ragas_judge_api_base == "https://groq.example/v1"
    assert settings.ragas_judge_model == "openai/gpt-oss-20b-test"
