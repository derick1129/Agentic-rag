from adaptive.config import Settings


def test_phoenix_content_capture_is_opt_in_and_independent_of_enabled(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/test")
    monkeypatch.delenv("PHOENIX_CAPTURE_CONTENT", raising=False)
    monkeypatch.setenv("PHOENIX_ENABLED", "true")
    defaults = Settings()
    assert defaults.phoenix_enabled is True
    assert defaults.phoenix_capture_content is False

    monkeypatch.setenv("PHOENIX_CAPTURE_CONTENT", "true")
    monkeypatch.setenv("PHOENIX_ENABLED", "false")
    enabled = Settings()
    assert enabled.phoenix_capture_content is True
    assert enabled.phoenix_enabled is False


def test_phoenix_api_and_evaluation_settings_keep_keys_secret(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://db/test")
    monkeypatch.setenv("PHOENIX_API_BASE_URL", "http://phoenix:6006")
    monkeypatch.setenv("PHOENIX_EVAL_GROQ_API_BASE", "https://groq.example/v1")
    monkeypatch.setenv("PHOENIX_EVAL_GROQ_MODEL", "llama-test")
    monkeypatch.setenv("PHOENIX_EVAL_GROQ_API_KEY", "groq-secret")
    settings = Settings()

    assert settings.phoenix_api_base_url == "http://phoenix:6006"
    assert settings.phoenix_eval_groq_api_base == "https://groq.example/v1"
    assert settings.phoenix_eval_groq_model == "llama-test"
    assert settings.phoenix_eval_groq_api_key.get_secret_value() == "groq-secret"
    assert "groq-secret" not in repr(settings)
