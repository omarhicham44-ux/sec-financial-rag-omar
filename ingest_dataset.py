import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunking import DocumentChunk
from vector_store import reset_collection, store_chunks


DATASET_FOLDER = Path("data/sec_filings")

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

BATCH_SIZE = 100


SECTION_TITLES = {
    "item_1": "Business",
    "item_1A": "Risk Factors",
    "item_1B": "Unresolved Staff Comments",
    "item_2": "Properties",
    "item_3": "Legal Proceedings",
    "item_4": "Mine Safety Disclosures",
    "item_5": "Market for Common Equity",
    "item_6": "Selected Financial Data",
    "item_7": (
        "Management Discussion and Analysis of "
        "Financial Condition and Results of Operations"
    ),
    "item_7A": (
        "Quantitative and Qualitative Disclosures "
        "About Market Risk"
    ),
    "item_8": "Financial Statements and Supplementary Data",
    "item_9": (
        "Changes in and Disagreements with Accountants"
    ),
    "item_9A": "Controls and Procedures",
    "item_9B": "Other Information",
    "item_10": (
        "Directors, Executive Officers and "
        "Corporate Governance"
    ),
    "item_11": "Executive Compensation",
    "item_12": (
        "Security Ownership of Certain Beneficial Owners "
        "and Management"
    ),
    "item_13": (
        "Certain Relationships, Related Transactions "
        "and Director Independence"
    ),
    "item_14": "Principal Accountant Fees and Services",
    "item_15": (
        "Exhibits and Financial Statement Schedules"
    ),
}


def clean_text(text: str) -> str:
    """
    Remove unnecessary spaces and blank lines while preserving
    readable paragraph boundaries.
    """

    text = text.replace("\u00a0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def get_report_year(
    filing: dict[str, Any],
) -> int:
    """
    Determine the report year from period_of_report.

    Falls back to filing_date if period_of_report is missing.
    """

    period_of_report = str(
        filing.get("period_of_report", "")
    )

    filing_date = str(
        filing.get("filing_date", "")
    )

    date_value = period_of_report or filing_date

    try:
        return int(date_value[:4])

    except (TypeError, ValueError):
        return 0


def make_chunk_id(
    source: str,
    section: str,
    chunk_number: int,
) -> str:
    """
    Create a stable and unique ChromaDB ID.
    """

    raw_id = (
        f"{source}|{section}|{chunk_number}"
    )

    digest = hashlib.sha256(
        raw_id.encode("utf-8")
    ).hexdigest()

    return digest


def create_filing_chunks(
    filing: dict[str, Any],
    source_path: Path,
) -> list[DocumentChunk]:
    """
    Convert one SEC filing JSON document into section-aware chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    company = str(
        filing.get("company", "Unknown company")
    ).strip()

    cik = str(
        filing.get("cik", "")
    ).strip()

    filing_type = str(
        filing.get("filing_type", "")
    ).strip()

    filing_date = str(
        filing.get("filing_date", "")
    ).strip()

    period_of_report = str(
        filing.get("period_of_report", "")
    ).strip()

    report_year = get_report_year(filing)

    source = source_path.name

    filing_chunks: list[DocumentChunk] = []

    for section_key, section_title in SECTION_TITLES.items():
        raw_section_text = filing.get(
            section_key,
            "",
        )

        if not isinstance(raw_section_text, str):
            continue

        cleaned_section = clean_text(
            raw_section_text
        )

        if not cleaned_section:
            continue

        section_header = (
            f"Company: {company}\n"
            f"CIK: {cik}\n"
            f"Filing type: {filing_type}\n"
            f"Report year: {report_year}\n"
            f"SEC section: {section_key}\n"
            f"Section title: {section_title}\n\n"
        )

        section_chunks = text_splitter.split_text(
            cleaned_section
        )

        total_section_chunks = len(
            section_chunks
        )

        for index, chunk_text in enumerate(
            section_chunks,
            start=1,
        ):
            final_text = (
                section_header
                + chunk_text
            )

            chunk_id = make_chunk_id(
                source=source,
                section=section_key,
                chunk_number=index,
            )

            metadata = {
                "company": company,
                "cik": cik,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "period_of_report": period_of_report,
                "report_year": report_year,
                "section": section_key,
                "section_title": section_title,
                "source": source,
                "source_path": str(source_path),
                "chunk_number": index,
                "section_chunk_count": total_section_chunks,
            }

            filing_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": final_text,
                    "metadata": metadata,
                }
            )

    return filing_chunks


def load_json_file(
    file_path: Path,
) -> dict[str, Any]:
    """
    Read and validate one JSON filing.
    """

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            filing = json.load(file)

    except UnicodeDecodeError:
        with file_path.open(
            "r",
            encoding="utf-8-sig",
        ) as file:
            filing = json.load(file)

    if not isinstance(filing, dict):
        raise ValueError(
            f"Expected one JSON object in {file_path.name}."
        )

    return filing


def store_batch(
    chunks: list[DocumentChunk],
) -> int:
    """
    Store one batch and return the total collection count.
    """

    if not chunks:
        return 0

    return store_chunks(chunks)


def ingest_dataset(
    rebuild: bool = True,
) -> None:
    """
    Read every JSON filing, chunk it, and store it in ChromaDB.
    """

    if not DATASET_FOLDER.exists():
        raise FileNotFoundError(
            "Dataset folder was not found:\n"
            f"{DATASET_FOLDER.resolve()}"
        )

    json_files = sorted(
        DATASET_FOLDER.glob("*.json")
    )

    if not json_files:
        raise FileNotFoundError(
            "No JSON files were found inside:\n"
            f"{DATASET_FOLDER.resolve()}"
        )

    if rebuild:
        reset_collection()

        print(
            "\nCreated a new empty SEC filing collection."
        )

    print(
        f"\nFound {len(json_files)} JSON filing files."
    )

    pending_chunks: list[DocumentChunk] = []

    processed_files = 0
    failed_files = 0
    total_created_chunks = 0
    collection_count = 0

    for file_number, file_path in enumerate(
        json_files,
        start=1,
    ):
        try:
            filing = load_json_file(
                file_path
            )

            filing_chunks = create_filing_chunks(
                filing=filing,
                source_path=file_path,
            )

            pending_chunks.extend(
                filing_chunks
            )

            total_created_chunks += len(
                filing_chunks
            )

            processed_files += 1

            company = filing.get(
                "company",
                "Unknown company",
            )

            print(
                f"[{file_number}/{len(json_files)}] "
                f"{company}: "
                f"{len(filing_chunks)} chunks"
            )

            while len(pending_chunks) >= BATCH_SIZE:
                batch = pending_chunks[:BATCH_SIZE]

                pending_chunks = pending_chunks[
                    BATCH_SIZE:
                ]

                collection_count = store_batch(
                    batch
                )

        except Exception as error:
            failed_files += 1

            print(
                f"[ERROR] {file_path.name}: {error}"
            )

    if pending_chunks:
        collection_count = store_batch(
            pending_chunks
        )

    print("\n" + "=" * 80)
    print("DATASET INGESTION COMPLETE")
    print("=" * 80)
    print(f"Files discovered: {len(json_files)}")
    print(f"Files processed: {processed_files}")
    print(f"Files failed: {failed_files}")
    print(f"Chunks created: {total_created_chunks}")
    print(
        f"Chunks currently stored in ChromaDB: "
        f"{collection_count}"
    )


if __name__ == "__main__":
    ingest_dataset(
        rebuild=True,
    )