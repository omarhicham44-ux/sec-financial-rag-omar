from pathlib import Path
from typing import TypedDict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunk(TypedDict):
    """
    Represents one chunk produced from a source document.
    """

    chunk_id: str
    text: str
    metadata: dict


def load_text_file(file_path: str) -> str:
    """
    Load a UTF-8 text file from the project.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document was not found: {path.resolve()}"
        )

    if not path.is_file():
        raise ValueError(
            f"The supplied path is not a file: {path.resolve()}"
        )

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(
            f"The document is empty: {path.resolve()}"
        )

    return text


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """
    Split text into overlapping chunks while attempting to preserve
    natural text boundaries.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_text(text)

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def create_document_chunks(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[DocumentChunk]:
    """
    Load a document, split it into chunks, and add metadata.
    """

    path = Path(file_path)

    document_text = load_text_file(file_path)

    text_chunks = split_text_into_chunks(
        text=document_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    document_chunks: list[DocumentChunk] = []

    total_chunks = len(text_chunks)

    for index, chunk_text in enumerate(text_chunks, start=1):
        document_chunks.append(
            {
                "chunk_id": f"{path.stem}_chunk_{index}",
                "text": chunk_text,
                "metadata": {
                    "source": path.name,
                    "source_path": str(path),
                    "chunk_number": index,
                    "total_chunks": total_chunks,
                },
            }
        )

    return document_chunks


def display_chunks(
    chunks: list[DocumentChunk],
) -> None:
    """
    Display the generated chunks in the terminal.
    """

    print(f"\nTotal chunks created: {len(chunks)}")
    print("=" * 80)

    for chunk in chunks:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Metadata: {chunk['metadata']}")
        print("-" * 80)
        print(chunk["text"])
        print("=" * 80)


if __name__ == "__main__":
    chunks = create_document_chunks(
        file_path="data/sample_manual.txt",
        chunk_size=500,
        chunk_overlap=100,
    )

    display_chunks(chunks)