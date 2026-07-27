"""
vector_store.py — ChromaDB storage, company resolution, and retrieval.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any

import chromadb

from chunking import DocumentChunk
from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DATABASE_PATH,
    ENABLE_DEBUG_LOGGING,
)
from utils.aliases import (
    load_company_aliases,
    normalize_company_name,
)


# -------------------------------------------------------------------
# CHROMA CLIENT AND COLLECTION
# -------------------------------------------------------------------

def get_chroma_client() -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client.
    """

    return chromadb.PersistentClient(
        path=CHROMA_DATABASE_PATH,
    )


def get_collection():
    """
    Get the SEC filing collection or create it when necessary.
    """

    client = get_chroma_client()

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
    )


def reset_collection():
    """
    Delete the collection and create a new empty collection.
    """

    client = get_chroma_client()

    try:
        client.delete_collection(
            name=CHROMA_COLLECTION_NAME,
        )
    except Exception:
        # The collection may not exist.
        pass

    clear_company_cache()

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
    )


# -------------------------------------------------------------------
# STORAGE
# -------------------------------------------------------------------

def store_chunks(
    chunks: list[DocumentChunk],
) -> int:
    """
    Store filing chunks and metadata in ChromaDB.
    """

    if not chunks:
        return get_collection().count()

    collection = get_collection()

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"]
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    # New documents may introduce new company names.
    clear_company_cache()

    return collection.count()


# -------------------------------------------------------------------
# INDEXED COMPANY DISCOVERY
# -------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_indexed_company_names() -> tuple[str, ...]:
    """
    Read all unique company names stored in Chroma metadata.

    The result is cached because the indexed company list normally
    remains unchanged while the application is running.
    """

    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        return ()

    company_names: set[str] = set()

    page_size = 5000
    offset = 0

    while offset < total_chunks:
        result = collection.get(
            limit=min(
                page_size,
                total_chunks - offset,
            ),
            offset=offset,
            include=["metadatas"],
        )

        metadatas = result.get(
            "metadatas",
            [],
        ) or []

        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue

            company = metadata.get("company")

            if (
                isinstance(company, str)
                and company.strip()
            ):
                company_names.add(
                    company.strip()
                )

        returned_count = len(
            result.get("ids", []) or []
        )

        if returned_count == 0:
            break

        offset += returned_count

    return tuple(
        sorted(company_names)
    )


def clear_company_cache() -> None:
    """
    Clear cached company metadata after collection changes.
    """

    get_indexed_company_names.cache_clear()


# -------------------------------------------------------------------
# COMPANY RESOLUTION
# -------------------------------------------------------------------

def calculate_company_match_score(
    requested_company: str,
    indexed_company: str,
) -> float:
    """
    Calculate a deterministic similarity score between two company names.
    """

    requested_normalized = normalize_company_name(
        requested_company
    )

    indexed_normalized = normalize_company_name(
        indexed_company
    )

    if (
        not requested_normalized
        or not indexed_normalized
    ):
        return 0.0

    if requested_normalized == indexed_normalized:
        return 1.0

    requested_words = set(
        requested_normalized.split()
    )

    indexed_words = set(
        indexed_normalized.split()
    )

    if requested_words and indexed_words:
        overlap = len(
            requested_words & indexed_words
        )

        union = len(
            requested_words | indexed_words
        )

        word_overlap_score = (
            overlap / union
            if union
            else 0.0
        )
    else:
        word_overlap_score = 0.0

    sequence_score = SequenceMatcher(
        None,
        requested_normalized,
        indexed_normalized,
    ).ratio()

    containment_score = 0.0

    if (
        requested_normalized in indexed_normalized
        or indexed_normalized in requested_normalized
    ):
        containment_score = 1.0

    return max(
        sequence_score,
        word_overlap_score,
        containment_score,
    )


def find_best_indexed_company(
    company_name: str,
    minimum_score: float = 0.72,
) -> tuple[str | None, float]:
    """
    Find the best exact indexed company name for a user-provided name.
    """

    indexed_companies = get_indexed_company_names()

    if not indexed_companies:
        return None, 0.0

    best_company: str | None = None
    best_score = 0.0

    for indexed_company in indexed_companies:
        score = calculate_company_match_score(
            requested_company=company_name,
            indexed_company=indexed_company,
        )

        if score > best_score:
            best_company = indexed_company
            best_score = score

    if best_score < minimum_score:
        return None, best_score

    return best_company, best_score


def resolve_company_name(
    company_name: str | None,
) -> str | None:
    """
    Resolve a user-facing company name to the exact value stored
    in Chroma metadata.

    Resolution order:

    1. Exact stored-name match
    2. Alias dictionary
    3. Normalized exact match
    4. Deterministic fuzzy match
    """

    if not company_name:
        return None

    requested = company_name.strip()

    if not requested:
        return None

    indexed_companies = get_indexed_company_names()

    # 1. Exact stored metadata match.
    if requested in indexed_companies:
        return requested

    requested_normalized = normalize_company_name(
        requested
    )

    aliases = load_company_aliases()

    # 2. Resolve through the manually curated alias dictionary.
    alias_target = aliases.get(
        requested_normalized
    )

    if alias_target:
        if alias_target in indexed_companies:
            resolved = alias_target

        else:
            resolved, _ = find_best_indexed_company(
                alias_target
            )

        if resolved:
            if ENABLE_DEBUG_LOGGING:
                print(
                    "\nCOMPANY RESOLUTION:"
                    f" requested={requested!r},"
                    f" resolved={resolved!r},"
                    " method=alias"
                )

            return resolved

    # 3. Compare normalized names exactly.
    for indexed_company in indexed_companies:
        if (
            normalize_company_name(indexed_company)
            == requested_normalized
        ):
            if ENABLE_DEBUG_LOGGING:
                print(
                    "\nCOMPANY RESOLUTION:"
                    f" requested={requested!r},"
                    f" resolved={indexed_company!r},"
                    " method=normalized_exact"
                )

            return indexed_company

    # 4. Controlled fuzzy matching.
    resolved, score = find_best_indexed_company(
        requested
    )

    if ENABLE_DEBUG_LOGGING:
        print(
            "\nCOMPANY RESOLUTION:"
            f" requested={requested!r},"
            f" resolved={resolved!r},"
            f" score={score:.3f},"
            " method=fuzzy"
        )

    return resolved


# -------------------------------------------------------------------
# FILTER BUILDING
# -------------------------------------------------------------------

def build_metadata_filter(
    company: str | None = None,
    report_year: int | None = None,
    section: str | None = None,
) -> dict[str, Any] | None:
    """
    Build a Chroma-compatible metadata filter.
    """

    conditions: list[dict[str, Any]] = []

    if company:
        conditions.append(
            {
                "company": {
                    "$eq": company,
                }
            }
        )

    if report_year is not None:
        conditions.append(
            {
                "report_year": {
                    "$eq": report_year,
                }
            }
        )

    if section:
        conditions.append(
            {
                "section": {
                    "$eq": section,
                }
            }
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "$and": conditions,
    }


# -------------------------------------------------------------------
# RETRIEVAL
# -------------------------------------------------------------------

def search_chunks(
    query: str,
    number_of_results: int = 5,
    company: str | None = None,
    report_year: int | None = None,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search ChromaDB using semantic similarity and optional metadata.

    A user-facing company name is automatically resolved to the exact
    company value stored in ChromaDB before applying the filter.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "The search query cannot be empty."
        )

    if number_of_results <= 0:
        raise ValueError(
            "number_of_results must be greater than zero."
        )

    collection = get_collection()
    available_chunks = collection.count()

    if available_chunks == 0:
        return []

    resolved_company = resolve_company_name(
        company
    )

    # Do not run an exact company-filtered search using an unresolved
    # user-facing name. That would silently produce zero results.
    if company and not resolved_company:
        if ENABLE_DEBUG_LOGGING:
            print(
                "\nSEARCH SKIPPED:"
                f" company={company!r} could not be resolved."
            )

        return []

    where_filter = build_metadata_filter(
        company=resolved_company,
        report_year=report_year,
        section=section,
    )

    result_count = min(
        number_of_results,
        available_chunks,
    )

    query_arguments: dict[str, Any] = {
        "query_texts": [
            query.strip()
        ],
        "n_results": result_count,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter is not None:
        query_arguments["where"] = where_filter

    if ENABLE_DEBUG_LOGGING:
        print("\nVECTOR SEARCH")
        print(f"Query: {query.strip()}")
        print(f"Requested company: {company}")
        print(f"Resolved company: {resolved_company}")
        print(f"Report year: {report_year}")
        print(f"Section: {section}")
        print(f"Where filter: {where_filter}")

    results = collection.query(
        **query_arguments,
    )

    documents = (
        results.get(
            "documents",
            [[]],
        )[0]
        or []
    )

    metadatas = (
        results.get(
            "metadatas",
            [[]],
        )[0]
        or []
    )

    distances = (
        results.get(
            "distances",
            [[]],
        )[0]
        or []
    )

    ids = (
        results.get(
            "ids",
            [[]],
        )[0]
        or []
    )

    retrieved_chunks: list[dict[str, Any]] = []

    for (
        chunk_id,
        document,
        metadata,
        distance,
    ) in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        retrieved_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": document or "",
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    if ENABLE_DEBUG_LOGGING:
        print(
            f"SEARCH RESULTS: {len(retrieved_chunks)}"
        )

    return retrieved_chunks


# -------------------------------------------------------------------
# DEBUGGING
# -------------------------------------------------------------------

def display_search_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:
    """
    Print retrieval results in a readable terminal format.
    """

    print("\nSearch query:")
    print(query)

    print("\nRetrieved chunks:")
    print("=" * 80)

    if not results:
        print("No chunks were found.")
        return

    for position, result in enumerate(
        results,
        start=1,
    ):
        metadata = (
            result.get("metadata")
            or {}
        )

        print(f"\nResult {position}")
        print(
            f"Chunk ID: "
            f"{result.get('chunk_id', 'unknown')}"
        )
        print(
            f"Distance: "
            f"{result.get('distance', 'unknown')}"
        )
        print(
            f"Company: "
            f"{metadata.get('company', 'unknown')}"
        )
        print(
            f"Report year: "
            f"{metadata.get('report_year', 'unknown')}"
        )
        print(
            f"Section: "
            f"{metadata.get('section_title', 'unknown')}"
        )
        print(
            f"Source: "
            f"{metadata.get('source', 'unknown')}"
        )
        print("-" * 80)
        print(
            result.get(
                "text",
                "",
            )
        )
        print("=" * 80)


if __name__ == "__main__":
    collection = get_collection()

    print(
        "The SEC filing collection currently contains "
        f"{collection.count()} chunks."
    )

    companies = get_indexed_company_names()

    print(
        f"The collection contains "
        f"{len(companies)} unique company names."
    )

    print("\nFirst indexed companies:")

    for company_name in companies[:20]:
        print(f"- {company_name}")