import asyncio
import base64
import json
import os
import shutil
from pathlib import Path

import httpx
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from schema import EnhancedDocumentRecord


WATCH_DIR = Path("./image_inbox")
OUTPUT_DIR = Path("./jsoniq_records")
ARCHIVE_DIR = Path("./archive")

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
METRICS_ENDPOINT = "http://localhost:8000/metrics"

# Set this to the exact model installed in Ollama.
MODEL_NAME = os.getenv("NEOMIND_MODEL", "neomind")

# Maximum number of simultaneous vision requests.
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "4"))

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


class AsyncModelWorker:
    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=240.0,
                write=60.0,
                pool=30.0,
            )
        )
        self.in_flight: set[str] = set()

    async def wait_for_file_stability(
        self,
        img_path: Path,
        checks: int = 3,
        interval: float = 0.5,
    ) -> None:
        """
        Wait until the file size remains unchanged for several checks.
        This prevents processing partially-written files.
        """
        previous_size = -1

        for _ in range(checks):
            if not img_path.exists():
                raise FileNotFoundError(img_path)

            current_size = img_path.stat().st_size

            if current_size == previous_size:
                return

            previous_size = current_size
            await asyncio.sleep(interval)

        if img_path.stat().st_size != previous_size:
            raise RuntimeError(
                f"File is still changing: {img_path}"
            )

    async def process_file(self, img_path: str):
        path = Path(img_path)
        filename = path.name

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        # Prevent duplicate watchdog events from scheduling the same file.
        key = str(path.resolve())

        if key in self.in_flight:
            return

        self.in_flight.add(key)

        try:
            output_path = OUTPUT_DIR / f"{path.stem}_record.json"
            archive_path = ARCHIVE_DIR / filename

            if output_path.exists():
                print(
                    f"⏭️ [Already Processed] {filename}"
                )
                return

            async with self.semaphore:
                print(
                    f"👁️‍🗨️ [Watchdog Processing] Processing: {filename}"
                )

                # Allow filesystem writes to settle.
                await self.wait_for_file_stability(path)

                start_time = asyncio.get_running_loop().time()

                with path.open("rb") as f:
                    b64_image = base64.b64encode(
                        f.read()
                    ).decode("utf-8")

                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Extract text, layout coordinates, "
                                "and all table cells. Return only "
                                "the structured document record."
                            ),
                            "images": [b64_image],
                        }
                    ],
                    "stream": False,
                    "format": (
                        EnhancedDocumentRecord.model_json_schema()
                    ),
                }

                response = await self.client.post(
                    OLLAMA_ENDPOINT,
                    json=payload,
                )

                response.raise_for_status()

                result = response.json()

                if "message" not in result:
                    raise ValueError(
                        "Ollama response does not contain 'message'"
                    )

                content = result["message"].get("content")

                if not content:
                    raise ValueError(
                        "Ollama returned an empty message"
                    )

                parsed_json = json.loads(content)

                # Validate against the Pydantic schema.
                validated = EnhancedDocumentRecord.model_validate(
                    parsed_json
                )

                record = {
                    "source_file": filename,
                    "document_data": validated.model_dump(
                        mode="json"
                    ),
                }

                # Write atomically so consumers never see
                # a partially-written JSON record.
                temporary_output = (
                    OUTPUT_DIR
                    / f".{path.stem}_record.tmp.json"
                )

                with temporary_output.open(
                    "w",
                    encoding="utf-8",
                ) as out_f:
                    json.dump(
                        record,
                        out_f,
                        indent=4,
                        ensure_ascii=False,
                    )
                    out_f.write("\n")

                temporary_output.replace(output_path)

                duration = (
                    asyncio.get_running_loop().time()
                    - start_time
                )

                # Optional monitoring hook.
                try:
                    await self.client.get(
                        METRICS_ENDPOINT,
                        timeout=5.0,
                    )
                except httpx.HTTPError:
                    # Metrics must never break document ingestion.
                    pass

                # Move only after the JSON record has been
                # successfully created.
                shutil.move(
                    str(path),
                    str(archive_path),
                )

                print(
                    f"📦 [Archived] Successfully parsed "
                    f"{filename} in {duration:.2f}s"
                )

        except FileNotFoundError:
            print(
                f"⚠️ [File Missing] {filename} "
                "disappeared before processing."
            )

        except json.JSONDecodeError as exc:
            print(
                f"❌ [Invalid JSON] Ollama returned invalid "
                f"structured output for {filename}: {exc}"
            )

        except httpx.HTTPError as exc:
            print(
                f"❌ [Ollama HTTP Fault] {filename}: {exc}"
            )

        except Exception as exc:
            print(
                f"❌ [Pipeline Critical Fault] "
                f"Failed processing {filename}: {exc}"
            )

        finally:
            self.in_flight.discard(key)

    async def close(self):
        await self.client.aclose()


class DocumentInboxHandler(FileSystemEventHandler):
    def __init__(self, loop, worker):
        super().__init__()
        self.loop = loop
        self.worker = worker

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        asyncio.run_coroutine_threadsafe(
            self.worker.process_file(str(path)),
            self.loop,
        )


async def main():
    for directory in (
        WATCH_DIR,
        OUTPUT_DIR,
        ARCHIVE_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    loop = asyncio.get_running_loop()
    worker = AsyncModelWorker()

    observer = Observer()

    handler = DocumentInboxHandler(
        loop,
        worker,
    )

    observer.schedule(
        handler,
        path=str(WATCH_DIR),
        recursive=False,
    )

    observer.start()

    print(
        "📡 Real-time watchdog active."
    )
    print(
        f"📥 Inbox:   {WATCH_DIR.resolve()}"
    )
    print(
        f"📄 Output:  {OUTPUT_DIR.resolve()}"
    )
    print(
        f"📦 Archive: {ARCHIVE_DIR.resolve()}"
    )
    print(
        f"🤖 Model:   {MODEL_NAME}"
    )
    print(
        f"⚡ Workers: {MAX_CONCURRENT_TASKS}"
    )

    try:
        while True:
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        pass

    finally:
        observer.stop()

        # Observer.join() is blocking, so run it outside
        # the asyncio event loop.
        await asyncio.to_thread(observer.join)

        await worker.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 NeuroMindAI document ingestion stopped.")
