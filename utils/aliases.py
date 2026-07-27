"""
Company alias loading and normalization utilities.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from config import COMPANY_ALIAS_PATH


def normalize_company_name(
    company_name: str,
) -> str:
    """
    Normalize a company name into a stable comparison key.
    """

    if not isinstance(company_name, str):
        return ""

    normalized = company_name.lower().strip()

    normalized = normalized.replace(
        "&",
        " and ",
    )

    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    removable_terms = {
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "company",
        "co",
        "limited",
        "ltd",
        "plc",
        "holdings",
        "holding",
    }

    words = [
        word
        for word in normalized.split()
        if word not in removable_terms
    ]

    return " ".join(words)


def load_company_aliases() -> dict[str, str]:
    """
    Load company aliases from data/company_aliases.json.
    """

    alias_path = Path(
        COMPANY_ALIAS_PATH
    )

    if not alias_path.exists():
        return {}

    try:
        with alias_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_aliases = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return {}

    if not isinstance(raw_aliases, dict):
        return {}

    aliases: dict[str, str] = {}

    for alias, canonical_name in raw_aliases.items():
        if not isinstance(alias, str):
            continue

        if not isinstance(canonical_name, str):
            continue

        normalized_alias = normalize_company_name(
            alias
        )

        if normalized_alias:
            aliases[normalized_alias] = (
                canonical_name.strip()
            )

    return aliases