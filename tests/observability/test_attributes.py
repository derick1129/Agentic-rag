from adaptive.observability.attributes import TelemetryConfig, normalize_attributes


def test_content_capture_allows_only_explicit_content_keys_and_never_credentials() -> None:
    attributes = {
        "query": "private question",
        "answer": "private answer",
        "retrieved_context": "private context",
        "prompt_template": "internal prompt",
        "document_text": "unselected document",
        "api_key": "secret",
        "authorization": "Bearer secret",
        "safe.count": 2,
    }

    redacted = normalize_attributes(attributes, TelemetryConfig())
    assert redacted == {"safe.count": 2}

    captured = normalize_attributes(attributes, TelemetryConfig(capture_content=True))
    assert captured == {
        "query": "private question",
        "answer": "private answer",
        "retrieved_context": "private context",
        "safe.count": 2,
    }
