"""Real-time image-to-structured-record ingestion for NeuroMindAI.

Pipeline:
    image_inbox -> watchdog -> stability check -> Ollama vision model
    -> Pydantic validation -> atomic JSON record -> archive
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import time
from pathlib import Path

import httpx
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from .schema import EnhancedDocumentRecord

WATCH_DIR = Path(os.getenv("NEUROMIND_WATCH_DIR", "./image_inbox"))
OUTPUT_DIR = Path(os.getenv("NEUROMIND_OUTPUT_DIR", "./jsoniq_records"))
ARCHIVE_DIR = Path(os.getenv("NEUROMIND_ARCHIVE_DIR", "./archive"))
OLLAMA_ENDPOINT = os.getenv(
    "OLLAMA_ENDPOINT", "http://localhost:11434/api/chat"
)
MODEL_NAME = os.getenv("NEOMIND_MODEL", "neomind")
MAX_CONCURRENT_TASKS = max(1, int(os.getenv("MAX_CONCURRENT_TASKS", "4")))
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROCESS_DELAY_SECONDS = float(os.getenv("NEUROMIND_SETTLE_DELAY", "1.0"))
STABILITY_CHECKS = max(1, int(os.getenv("NEUROMIND_STABILITY_CHECKS", "2")))


class AsyncModelWorker:
    def __init__(self) -> None:
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(240.0, connect=10.0)
        )
        self.in_flight: set[str] = set()

    async def wait_for_file_stability(self, path: Path) -> None:
        previous_size = -1
        stable = 0
        while stable < STABILITY_CHECKS:
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                raise
            if size > 0 and size == previous_size:
                stable += 1
            else:
                stable = 0
            previous_size = size
            await asyncio.sleep(PROCESS_DELAY_SECONDS)

    async def process_file(self, img_path: str | Path) -> None:
        path = Path(img_path)
        key = str(path.resolve())
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or key in self.in_flight:
            return

        output_path = OUTPUT_DIR / f"{path.stem}_record.json"
        archive_path = ARCHIVE_DIR / path.name
        if output_path.exists() or archive_path.exists():
            return

        self.in_flight.add(key)
        async with self.semaphore:
            start = time.monotonic()
            try:
                await self.wait_for_file_stability(path)

                with path.open("rb") as source:
                    image_b64 = base64.b64encode(source.read()).decode("ascii")

                payload = {
                    "model": MODEL_NAME,
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Extract all text, layout coordinates, and table "
                            "cells. Return only data matching the supplied schema."
                        ),
                        "images": [image_b64],
                    }],
                    "stream": False,
                    "format": EnhancedDocumentRecord.model_json_schema(),
                }

                response = await self.client.post(
                    OLLAMA_ENDPOINT, json=payload
                )
                response.raise_for_status()
                body = response.json()
                content = body.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Ollama returned no message content")

                parsed = json.loads(content)
                record = EnhancedDocumentRecord.model_validate(parsed)

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                temp_path = output_path.with_suffix(".json.tmp")
                envelope = {
                    "source_file": path.name,
                    "document_data": record.model_dump(mode="json"),
                }
                temp_path.write_text(
                    json.dumps(
                        envelope, indent=4, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
                temp_path.replace(output_path)

                ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(archive_path))

                duration = time.monotonic() - start
                print(
                    f"📦 [Archived] {path.name} -> {output_path.name} "
                    f"in {duration:.2f}s"
                )
            except FileNotFoundError:
                print(f"⚠️ [Skipped] File disappeared before processing: {path}")
            except (json.JSONDecodeError, ValueError) as exc:
                print(f"❌ [Invalid Model Output] {path.name}: {exc}")
            except httpx.HTTPError as exc:
                print(f"❌ [Ollama HTTP Fault] {path.name}: {exc}")
            except Exception as exc:
                print(f"❌ [Pipeline Fault] {path.name}: {exc}")
            finally:
                self.in_flight.discard(key)

    async def close(self) -> None:
        await self.client.aclose()


class DocumentInboxHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, worker: AsyncModelWorker):
        self.loop = loop
        self.worker = worker

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        if Path(event.src_path).suffix.lower() in SUPPORTED_EXTENSIONS:
            asyncio.run_coroutine_threadsafe(
                self.worker.process_file(event.src_path), self.loop
            )


async def main() -> None:
    for directory in (WATCH_DIR, OUTPUT_DIR, ARCHIVE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    worker = AsyncModelWorker()
    observer = Observer()
    observer.schedule(
        DocumentInboxHandler(loop, worker),
        path=str(WATCH_DIR),
        recursive=False,
    )
    observer.start()

    print("📡 NeuroMindAI document watchdog active")
    print(f"   Inbox:   {WATCH_DIR}")
    print(f"   Records: {OUTPUT_DIR}")
    print(f"   Archive: {ARCHIVE_DIR}")
    print(f"   Ollama:  {OLLAMA_ENDPOINT}")
    print(f"   Model:   {MODEL_NAME}")
    print(f"   Workers: {MAX_CONCURRENT_TASKS}")

    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        observer.stop()
        await asyncio.to_thread(observer.join)
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
