"""
main.py — Receipt ML API v5
Auto-download models từ Google Drive khi khởi động
"""
import os
import gdown
import joblib
import unicodedata
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager


# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════

MODELS = {
    "models/merchant_classifier_latest.pkl": "1dA8NWZWZwI23fGmdn4yZZ_iFAlLu-lLA",
    "models/lineitem_classifier_latest.pkl" : "11vnfeosffql7Q_HEA82-RgMQOdEMuZJq",
}

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))


# ════════════════════════════════════════════════════════════
# DOWNLOAD MODELS KHI KHỞI ĐỘNG
# ════════════════════════════════════════════════════════════

def download_models():
    """Download model từ Drive nếu chưa có"""
    os.makedirs("models", exist_ok=True)
    for path, file_id in MODELS.items():
        if os.path.exists(path):
            print(f"✅ Already exists: {path}")
            continue
        print(f"⬇️  Downloading {path}...")
        gdown.download(id=file_id, output=path, quiet=False)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"✅ Downloaded: {path} ({size:.1f} MB)")
        else:
            raise RuntimeError(f"❌ Failed to download {path}")


# Load models vào memory
merchant_model = None
lineitem_model  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Chạy khi server khởi động"""
    global merchant_model, lineitem_model
    print("🚀 Server starting...")
    download_models()
    merchant_model = joblib.load("models/merchant_classifier_latest.pkl")
    lineitem_model  = joblib.load("models/lineitem_classifier_latest.pkl")
    print("✅ Models loaded! Server ready.")
    yield
    print("👋 Server shutting down...")


# ════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title="Receipt ML API",
    version="5.0",
    description="Phân loại hóa đơn Việt Nam bằng ML",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for c, r in {"đ": "d", "Đ": "D"}.items():
        text = text.replace(c, r)
    text = unicodedata.normalize("NFC", text).lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text).strip()


def classify(text: str, model) -> dict:
    """Phân loại text, trả về category + confidence"""
    clean = preprocess(text)
    if not clean:
        return {"category": "Khac", "confidence": 0.0, "low_confidence": True}

    cat   = model.predict([clean])[0]
    proba = model.predict_proba([clean])[0]
    conf  = float(max(proba))
    low   = conf < CONFIDENCE_THRESHOLD

    return {
        "category"      : "Khac" if low else cat,
        "confidence"    : round(conf, 3),
        "low_confidence": low,
    }


# ════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════

class TextBlock(BaseModel):
    text  : str
    y     : float
    height: float = 0.03
    x     : float = 0.0
    width : float = 1.0

class ReceiptRequest(BaseModel):
    blocks: list[TextBlock]

class ClassifyRequest(BaseModel):
    text        : str
    use_lineitem: bool = False


# ════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status" : "ok",
        "version": "5.0",
        "models" : {
            "merchant": os.path.exists("models/merchant_classifier_latest.pkl"),
            "lineitem" : os.path.exists("models/lineitem_classifier_latest.pkl"),
        }
    }


@app.get("/health")
def health():
    """Render dùng endpoint này để check server sống không"""
    return {"status": "healthy"}


@app.post("/classify")
def classify_text(req: ClassifyRequest):
    """
    Phân loại 1 chuỗi text đơn giản.
    Dùng để test model nhanh.
    """
    model = lineitem_model if req.use_lineitem else merchant_model
    return classify(req.text, model)


@app.post("/receipt")
def process_receipt(req: ReceiptRequest):
    """
    Pipeline đầy đủ:
    text blocks từ ML Kit → extract → classify → trả kết quả

    Response:
    - receipt_type = "normal"      → is_multi_cat = False → EditTransactionFragment
    - receipt_type = "supermarket" → is_multi_cat = True  → MultiCategoryFragment
    """
    from extractor import extract_receipt

    # Bước 1: Extract structured data
    blocks  = [b.model_dump() for b in req.blocks]
    receipt = extract_receipt(blocks)

    # Bước 2: Classify merchant
    merchant_result = classify(
        receipt.get("merchant_name") or "",
        merchant_model
    )
    receipt["merchant_category"]   = merchant_result["category"]
    receipt["merchant_confidence"] = merchant_result["confidence"]
    receipt["merchant_low_conf"]   = merchant_result["low_confidence"]

    if receipt["receipt_type"] == "normal":
        # ── Hóa đơn thông thường: 1 category ─────────────────
        receipt["breakdown"]    = {
            merchant_result["category"]: receipt.get("total_amount") or 0
        }
        receipt["is_multi_cat"] = False

    else:
        # ── Hóa đơn siêu thị: nhiều category ─────────────────
        breakdown = {}
        for item in receipt.get("items", []):
            # Thử lineitem_model trước
            r = classify(item["name"], lineitem_model)
            # Nếu không chắc → thử merchant_model
            if r["low_confidence"]:
                r2 = classify(item["name"], merchant_model)
                if not r2["low_confidence"]:
                    r = r2

            item["category"]       = r["category"]
            item["confidence"]     = r["confidence"]
            item["low_confidence"] = r["low_confidence"]

            # Cộng dồn vào breakdown
            cat = r["category"]
            breakdown[cat] = breakdown.get(cat, 0) + item.get("total_price", 0)

        receipt["breakdown"]    = breakdown
        receipt["is_multi_cat"] = len(breakdown) > 1

    return receipt
