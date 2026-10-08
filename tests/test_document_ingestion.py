import asyncio
import json

import httpx

from neuromind.document_ingestion import AsyncModelWorker


def test_worker_payload_and_atomic_record(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    records = tmp_path / "records"
    archive = tmp_path / "archive"
    inbox.mkdir()
    image = inbox / "invoice.png"
    image.write_bytes(b"fake-image")

    monkeypatch.setattr("neuromind.document_ingestion.OUTPUT_DIR", records)
    monkeypatch.setattr("neuromind.document_ingestion.ARCHIVE_DIR", archive)
    monkeypatch.setattr("neuromind.document_ingestion.PROCESS_DELAY_SECONDS", 0.0)

    async def run():
        worker = AsyncModelWorker()

        async def post(url, **kwargs):
            assert kwargs["json"]["model"]
            assert kwargs["json"]["stream"] is False
            content = json.dumps({
                "document_type": "invoice",
                "entity_name": "Example Vendor",
                "line_items": [{"description": "Widget", "quantity": 2}],
                "table_cells": [{"row": 0, "column": 0, "text": "Widget"}],
            })
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={"message": {"content": content}},
            )

        worker.client.post = post
        await worker.process_file(image)
        await worker.close()

    asyncio.run(run())

    output = records / "invoice_record.json"
    assert output.exists()
    assert not image.exists()
    assert (archive / "invoice.png").exists()
    data = json.loads(output.read_text())
    assert data["document_data"]["entity_name"] == "Example Vendor"
    assert data["document_data"]["line_items"][0]["description"] == "Widget"
