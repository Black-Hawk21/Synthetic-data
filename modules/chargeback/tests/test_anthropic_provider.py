import json

from aegis.providers.anthropic_provider import _extract_json


def test_strips_markdown_code_fence():
    assert json.loads(_extract_json('```json\n{"a": 1}\n```')) == {"a": 1}


def test_strips_trailing_prose_after_json():
    assert json.loads(_extract_json('{"a": 1}\n\nNote: this reflects standard policy.')) == {"a": 1}


def test_strips_leading_prose_before_json():
    assert json.loads(_extract_json('Here is the result:\n{"a": 1, "b": [1, 2]}')) == {"a": 1, "b": [1, 2]}


def test_plain_json_with_no_wrapping():
    assert json.loads(_extract_json('{"a": 1}')) == {"a": 1}


def test_reasoning_before_json_with_a_brace_inside_the_prose():
    # The vision prompt now asks for chain-of-thought reasoning before the final JSON;
    # if that reasoning happens to contain a stray '{' (e.g. describing an object),
    # extraction must still land on the real, complete JSON object, not the fragment.
    text = (
        "STEP 1: the object is a {generic thing} on a table. STEP 2: looks fine.\n\n"
        'Final answer:\n{"a": 1, "nested": {"x": 2}}'
    )
    assert json.loads(_extract_json(text)) == {"a": 1, "nested": {"x": 2}}


def test_does_not_return_a_nested_finding_instead_of_the_full_result():
    # Each item inside "findings" is itself valid JSON on its own -- extraction must
    # return the full top-level object, not just the last/first nested fragment.
    text = (
        "Reasoning about findings...\n"
        '{"artifact_score": 0.1, "findings": ['
        '{"type": "x", "confidence": 0.5, "description": "y"}, '
        '{"type": "z", "confidence": 0.9, "description": "w"}]}'
    )
    result = json.loads(_extract_json(text))
    assert result["artifact_score"] == 0.1
    assert len(result["findings"]) == 2
