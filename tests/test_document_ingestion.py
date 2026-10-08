import pytest


ingestion = pytest.importorskip(
    "neuromind.document_ingestion",
    reason="Document ingestion API is not present in neuromind-v0.1 yet",
)


def test_document_ingestion_module_imports():
    assert ingestion is not None


def test_document_ingestion_exposes_supported_entrypoint():
    candidates = (
        "DocumentIngestor",
        "DocumentIngestion",
        "ingest_document",
        "ingest",
    )
    assert any(hasattr(ingestion, name) for name in candidates), (
        "neuromind.document_ingestion must expose an ingestion class or "
        "ingestion function"
    )
