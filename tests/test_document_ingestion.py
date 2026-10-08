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
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            content = json.dumps({
                "document_type": "invoice",
                "entity_name": "Example Vendor",
                "line_items": [{"description": "Widget", "quantity": 2}],
                "table_cells": [{"row": 0, "column": 0, "text": "Widget"}],
            })
            return httpx.Response(
                200,
                request=request,
                json={"message": {"content": content}},
            )

        worker = AsyncModelWorker()
        await worker.client.aclose()
        worker.client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
        await worker.process_file(image)
        await worker.close()

        assert requests
        payload = json.loads(requests[0].content)
        assert payload["model"]
        assert payload["stream"] is False
        assert payload["format"]["type"] == "object"

    asyncio.run(run())

    output = records / "invoice_record.json"
    assert output.exists()
    assert not image.exists()
    assert (archive / "invoice.png").exists()
    data = json.loads(output.read_text())
    assert data["document_data"]["entity_name"] == "Example Vendor"
    assert data["document_data"]["line_items"][0]["description"] == "Widget"
