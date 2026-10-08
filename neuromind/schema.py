"""Structured document records emitted by the NeuroMindAI ingestion pipeline."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="allow")
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None


class LineItem(BaseModel):
    model_config = ConfigDict(extra="allow")
    description: str = ""
    quantity: float | None = None
    unit_price: float | None = None
    total_price: float | None = None


class TableCell(BaseModel):
    model_config = ConfigDict(extra="allow")
    row: int | None = None
    column: int | None = None
    text: str = ""
    bounding_box: BoundingBox | None = None


class TextBlock(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str = ""
    bounding_box: BoundingBox | None = None


class EnhancedDocumentRecord(BaseModel):
    """Normalized OCR/document representation.

    Extra fields are preserved so model-specific extraction can evolve without
    breaking the ingestion contract.
    """

    model_config = ConfigDict(extra="allow")

    document_type: str | None = None
    entity_name: str | None = None
    transaction_date: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    line_items: list[LineItem] = Field(default_factory=list)
    text_blocks: list[TextBlock] = Field(default_factory=list)
    table_cells: list[TableCell] = Field(default_factory=list)
    layout: list[dict[str, Any]] = Field(default_factory=list)
