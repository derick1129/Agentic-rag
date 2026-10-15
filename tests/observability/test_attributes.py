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
    assert "answer" not in redacted
    assert redacted == {"safe.count": 2}

    captured = normalize_attributes(attributes, TelemetryConfig(capture_content=True))
    assert captured == {
        "query": "private question",
        "answer": "private answer",
        "retrieved_context": "private context",
        "safe.count": 2,
    }

    credentials_remain_filtered = normalize_attributes(
        attributes,
        TelemetryConfig(capture_content=True, redact_credentials=False),
    )
    assert not {"api_key", "authorization"} & credentials_remain_filtered.keys()
    credentials_remain_filtered_by_default = normalize_attributes(
        attributes, TelemetryConfig(redact_credentials=False)
    )
    assert not {"api_key", "authorization"} & credentials_remain_filtered_by_default.keys()
