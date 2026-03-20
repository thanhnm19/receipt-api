"""
main.py — Receipt ML API v5
Auto-download models từ Google Drive khi khởi động
"""
import os
import re
import unicodedata
import joblib
import gdown
from fastapi import FastAPI
from pydantic import BaseModel
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
# KEYWORD FALLBACK — xử lý tên mới chưa có trong dataset
# ════════════════════════════════════════════════════════════

KEYWORD_MAP = {
    "An_uong": [
        "cafe", "coffee", "ca phe", "tra sua", "bubble tea", "boba",
        "quan an", "nha hang", "restaurant", "buffet", "lau", "nuong",
        "pho", "bun", "com", "banh mi", "pizza", "burger", "kfc",
        "gofood", "grabfood", "shopeefood", "baemin", "order",
        "che", "kem", "sinh to", "nuoc ep", "smoothie",
    ],
    "Di_lai": [
        "grab", "be ", "taxi", "xe om", "giao thong",
        "xang", "dau nhot", "sua xe", "rua xe", "bai xe",
        "may bay", "ve tau", "ve xe", "xe buyt", "metro",
        "airlines", "vietjet", "bamboo", "vietnam air",
        "phuong trang", "hoang long", "thuê xe", "thue xe",
    ],
    "Hoc_tap": [
        "hoc phi", "hoc tieng", "ielts", "toeic", "toefl",
        "truong", "dai hoc", "mam non", "tieu hoc", "thpt",
        "khoa hoc", "course", "udemy", "coursera", "skillshare",
        "sach giao khoa", "van phong pham", "but", "vo",
        "gia su", "luyen thi", "hoc dan", "hoc ve", "hoc boi",
    ],
    "Gia_dinh": [
        "tien dien", "tien nuoc", "tien nha", "thue nha",
        "evn", "sawaco", "internet", "wifi", "cap quang",
        "viettel", "mobifone", "vinaphone", "fpt",
        "quan ly chung cu", "phi dich vu", "bao duong",
        "may lanh", "tu lanh", "may giat", "dien may",
        "noi that", "sua chua", "tho dien", "ve sinh nha",
    ],
    "Suc_khoe": [
        "benh vien", "phong kham", "bac si", "y te",
        "nha thuoc", "thuoc", "vitamin", "vaccine",
        "kham benh", "xet nghiem", "sieu am", "chup x quang",
        "gym", "yoga", "the duc", "tap the thao",
        "bao hiem suc khoe", "bhyt", "bhxh",
        "nha khoa", "rang", "mat kinh", "thi luc",
    ],
    "Lam_dep": [
        "spa", "nail", "tiem nail", "lam mong",
        "cat toc", "uon toc", "nhuom toc", "salon",
        "massage", "tham my", "phun xam",
        "my pham", "son moi", "kem duong", "serum",
        "wax long", "facial", "cham soc da",
        "nuoc hoa", "make up", "trang diem",
    ],
    "Thu_cung": [
        "thu cung", "cho meo", "cho cun", "meo con",
        "thu y", "bac si thu y", "tiem thu y",
        "thuc an cho", "thuc an meo", "cat ve sinh",
        "pet", "petmart", "petcity", "grooming",
        "cat long cho", "tam cho", "vaccine cho meo",
    ],
    "Giai_tri": [
        "cinema", "cgv", "bhd", "lotte cinema", "phim",
        "netflix", "spotify", "youtube", "apple music",
        "game", "steam", "garena", "nap the", "nap xu",
        "karaoke", "billiard", "bowling", "escape room",
        "ve concert", "su kien", "vinpearl", "suoi tien",
        "board game", "giai tri",
    ],
    "Mua_sam": [
        "shopee", "lazada", "tiki", "sendo",
        "uniqlo", "zara", "h&m", "pull bear",
        "giay dep", "quan ao", "tui xach", "dong ho",
        "dien thoai", "laptop", "may tinh", "tai nghe",
        "the gioi di dong", "fpt shop", "cellphones",
        "sac du phong", "op lung", "phu kien",
    ],
    "Du_lich": [
        "khach san", "hotel", "resort", "villa",
        "airbnb", "booking", "agoda", "traveloka",
        "tour", "du lich", "ve may bay", "visa",
        "thue phong", "nha nghi", "hostel",
        "cap treo", "tham quan", "vui choi",
        "vietravel", "saigontourist", "klook",
    ],
}


def keyword_fallback(text: str) -> tuple:
    """
    Tìm category dựa trên keyword.
    Trả về (category, score) — score = số keyword khớp
    """
    norm = _no_accent(text.lower())
    best_cat   = None
    best_score = 0

    for cat, keywords in KEYWORD_MAP.items():
        score = sum(1 for kw in keywords if kw in norm)
        if score > best_score:
            best_score = score
            best_cat   = cat

    return best_cat, best_score


def _no_accent(text: str) -> str:
    for c, r in {"đ": "d", "Đ": "D"}.items():
        text = text.replace(c, r)
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ════════════════════════════════════════════════════════════
# DOWNLOAD MODELS
# ════════════════════════════════════════════════════════════

def download_models():
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


merchant_model = None
lineitem_model  = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global merchant_model, lineitem_model
    print("🚀 Server starting...")
    download_models()
    merchant_model = joblib.load("models/merchant_classifier_latest.pkl")
    lineitem_model  = joblib.load("models/lineitem_classifier_latest.pkl")
    print("✅ Models loaded! Server ready.")
    yield


# ════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════

app = FastAPI(
    title="Receipt ML API",
    version="5.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════════
# PREPROCESSING
# ════════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _no_accent(text)
    text = unicodedata.normalize("NFC", text).lower()
    nfkd = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9\s]", " ", text).strip()


def classify(text: str, model) -> dict:
    """
    Phân loại text theo 3 lớp:
    1. ML model  — nếu confidence >= threshold → trả về luôn
    2. Keyword   — nếu model không chắc nhưng có keyword khớp
    3. Fallback  → Khac
    """
    clean = preprocess(text)
    if not clean:
        return {"category": "Khac", "confidence": 0.0,
                "low_confidence": True, "source": "fallback"}

    # Lớp 1: ML model
    cat   = model.predict([clean])[0]
    proba = model.predict_proba([clean])[0]
    conf  = float(max(proba))

    if conf >= CONFIDENCE_THRESHOLD:
        return {"category": cat, "confidence": round(conf, 3),
                "low_confidence": False, "source": "model"}

    # Lớp 2: Keyword fallback
    kw_cat, kw_score = keyword_fallback(text)
    if kw_cat and kw_score >= 1:
        return {"category": kw_cat, "confidence": round(conf, 3),
                "low_confidence": True, "source": "keyword"}

    # Lớp 3: Không xác định
    return {"category": "Khac", "confidence": round(conf, 3),
            "low_confidence": True, "source": "fallback"}


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
    return {"status": "ok", "version": "5.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/classify")
def classify_text(req: ClassifyRequest):
    model = lineitem_model if req.use_lineitem else merchant_model
    return classify(req.text, model)

@app.post("/receipt")
def process_receipt(req: ReceiptRequest):
    from extractor import extract_receipt

    blocks  = [b.model_dump() for b in req.blocks]
    receipt = extract_receipt(blocks)

    # Classify merchant
    merchant_result = classify(
        receipt.get("merchant_name") or "", merchant_model
    )
    receipt["merchant_category"]   = merchant_result["category"]
    receipt["merchant_confidence"] = merchant_result["confidence"]
    receipt["merchant_low_conf"]   = merchant_result["low_confidence"]
    receipt["merchant_source"]     = merchant_result["source"]

    if receipt["receipt_type"] == "normal":
        receipt["breakdown"]    = {
            merchant_result["category"]: receipt.get("total_amount") or 0
        }
        receipt["is_multi_cat"] = False
    else:
        breakdown = {}
        for item in receipt.get("items", []):
            r = classify(item["name"], lineitem_model)
            if r["low_confidence"]:
                r2 = classify(item["name"], merchant_model)
                if not r2["low_confidence"]:
                    r = r2
            item["category"]       = r["category"]
            item["confidence"]     = r["confidence"]
            item["low_confidence"] = r["low_confidence"]
            item["source"]         = r["source"]
            cat = r["category"]
            breakdown[cat] = breakdown.get(cat, 0) + item.get("total_price", 0)

        receipt["breakdown"]    = breakdown
        receipt["is_multi_cat"] = len(breakdown) > 1

    return receipt


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
