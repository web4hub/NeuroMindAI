import os
import json
import glob
import asyncio
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, PlainTextResponse
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import pandas as pd
import chromadb
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from apscheduler.schedulers.background import BackgroundScheduler
from styler import export_styled_excel

app = FastAPI(title="Enterprise Secure LMLM Stack", version="3.5.0")

# --- Prometheus Telemetry Metrics ---
INFERENCE_COUNTER = Counter('lmlm_documents_processed_total', 'Total documents ingested by LMLM backend')
LATENCY_HISTOGRAM = Histogram('lmlm_inference_latency_seconds', 'Time spent processing document payloads')

# --- Asymmetric JWT Cryptography Configuration ---
PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
PUBLIC_KEY = PRIVATE_KEY.public_key()
JWT_ALGORITHM = "RS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/auth/token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=60))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, PRIVATE_KEY, algorithm=JWT_ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, PUBLIC_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token parameters.")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials.")

# --- Directories & DB Setup ---
RECORDS_DIR, EXPORTS_DIR, CHROMA_DIR = "./jsoniq_records", "./exports", "./vector_db"
for d in [RECORDS_DIR, EXPORTS_DIR, CHROMA_DIR]: os.makedirs(d, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
vector_collection = chroma_client.get_or_create_collection(name="document_line_items")

# --- Core Business Logic ---
def compile_warehouse_data():
    target_pattern = os.path.join(RECORDS_DIR, "*.json")
    record_files = [f for f in glob.glob(target_pattern) if "unified_warehouse_manifest.json" not in f]
    if not record_files: return None

    compiled_documents, flat_rows = [], []

    for file_path in record_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
                compiled_documents.append(record)
                
                doc_meta = record.get("document_data", {})
                source_file = record.get("source_file", "Unknown")
                vendor = doc_meta.get("entity_name", "Unknown")
                grand_total = float(doc_meta.get("total_amount", 0.0))
                line_items = doc_meta.get("line_items", [])
                
                base_meta = {
                    "Source File": source_file, "Vendor": vendor,
                    "Doc Type": doc_meta.get("document_type"), "Date": doc_meta.get("transaction_date"),
                    "Currency": doc_meta.get("currency", "USD"), "Subtotal": doc_meta.get("subtotal"),
                    "Tax": doc_meta.get("tax_amount"), "Grand Total": grand_total
                }
                
                if line_items:
                    for idx, item in enumerate(line_items):
                        item_desc = item.get("description", "")
                        row = base_meta.copy()
                        row.update({
                            "Item Description": item_desc, "Quantity": item.get("quantity"),
                            "Unit Price": item.get("unit_price"), "Item Total": item.get("total_price")
                        })
                        flat_rows.append(row)
                        
                        # Sync to Chroma Vector Database
                        vector_collection.upsert(
                            ids=[f"{Path(source_file).stem}_row_{idx}"],
                            documents=[item_desc],
                            metadatas=[{"vendor": vendor, "source_file": source_file, "total_price": float(item.get("total_price", 0))}]
                        )
                else:
                    row = base_meta.copy()
                    row.update({"Item Description": "No items extracted", "Quantity": 0, "Unit Price": 0, "Item Total": 0})
                    flat_rows.append(row)
        except Exception:
            continue

    with open(os.path.join(RECORDS_DIR, "unified_warehouse_manifest.json"), 'w', encoding='utf-8') as out_f:
        json.dump({"documents": compiled_documents}, out_f, indent=4, ensure_ascii=False)

    if flat_rows:
        excel_path = os.path.join(EXPORTS_DIR, "document_master_ledger.xlsx")
        export_styled_excel(flat_rows, excel_path)
        return excel_path
    return None

scheduler = BackgroundScheduler()
scheduler.add_job(compile_warehouse_data, 'cron', hour=0, minute=0)
scheduler.start()

# --- OAuth2 Authentication Routes ---
@app.post("/v1/auth/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Strict sandbox credential check
    if form_data.username == "admin" and form_data.password == "securepassword123":
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password.")

# --- Secured Application Routes ---
@app.post("/v1/collection/compile", dependencies=[Depends(get_current_user)])
async def manual_compile():
    excel_file = compile_warehouse_data()
    if not excel_file: raise HTTPException(status_code=404, detail="No source profiles found.")
    return {"success": True, "message": "Warehouse databases compiled."}

@app.get("/v1/collection/export/excel", dependencies=[Depends(get_current_user)])
async def export_excel():
    excel_path = os.path.join(EXPORTS_DIR, "document_master_ledger.xlsx")
    if not os.path.exists(excel_path): compile_warehouse_data()
    return FileResponse(path=excel_path, filename="document_master_ledger.xlsx")

@app.get("/v1/collection/search", dependencies=[Depends(get_current_user)])
async def hybrid_search(query: str, vendor_filter: Optional[str] = None, limit: int = 5):
    """Executes high-speed vector lookup query with explicit metadata filter options."""
    try:
        where_clause = {"vendor": vendor_filter} if vendor_filter else None
        results = vector_collection.query(query_texts=[query], n_results=limit, where=where_clause)
        return {"success": True, "matches": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Metrics Telemetry Target ---
@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    """Exposes native metrics formatting payload endpoints out to Prometheus servers."""
    return ge
