"""
Shared JSON parsing utilities.
"""

from __future__ import annotations

import json
from typing import Any


def parse_json_response(
    raw_response: str,
    response_name: str,
) -> dict[str, Any]:
    cleaned_response = raw_response.strip()

    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]
    elif cleaned_response.startswith("```"):
        cleaned_response = cleaned_response[3:]

    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]

    cleaned_response = cleaned_response.strip()

    try:
        parsed_response = json.loads(cleaned_response)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"{response_name} did not return valid JSON.\n"
            f"Raw response:\n{raw_response}"
        ) from error

    if not isinstance(parsed_response, dict):
        raise ValueError(
            f"{response_name} must return one JSON object."
        )

    return parsed_response