"""
main.py - Receipt ML API v6

Enhancements:
- Hybrid classification (merchant model + line-item model + keyword fallback)
- Better confidence strategy for unseen/new text
- Multi-category breakdown with amount detail
- Normal receipts can infer multi-category from line candidates when reliable
"""

import os
import re
import unicodedata
from contextlib import asynccontextmanager
from typing import Any, Optional

import gdown
import joblib
from fastapi import FastAPI
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

MODELS = {
	"models/merchant_classifier_latest.pkl": "1dA8NWZWZwI23fGmdn4yZZ_iFAlLu-lLA",
	"models/lineitem_classifier_latest.pkl": "11vnfeosffql7Q_HEA82-RgMQOdEMuZJq",
}

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.40"))
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "3"))
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_AUTO_DOWNLOAD = os.getenv("MODEL_AUTO_DOWNLOAD", "1") == "1"
MODEL_DOWNLOAD_QUIET = os.getenv("MODEL_DOWNLOAD_QUIET", "1") == "1"
MERCHANT_MODEL_PATH = os.getenv(
	"MERCHANT_MODEL_PATH",
	os.path.join(MODELS_DIR, "merchant_classifier_latest.pkl"),
)
LINEITEM_MODEL_PATH = os.getenv(
	"LINEITEM_MODEL_PATH",
	os.path.join(MODELS_DIR, "lineitem_classifier_latest.pkl"),
)

MODEL_RUNTIME_STATUS = {
	"merchant_loaded": False,
	"lineitem_loaded": False,
	"merchant_path": MERCHANT_MODEL_PATH,
	"lineitem_path": LINEITEM_MODEL_PATH,
	"auto_download": MODEL_AUTO_DOWNLOAD,
}


# ============================================================
# KEYWORD FALLBACK
# ============================================================

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
		"phuong trang", "hoang long", "thue xe",
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
		"uniqlo", "zara", "h m", "pull bear",
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


def _no_accent(text: str) -> str:
	for c, r in {"d": "d", "D": "D", "đ": "d", "Đ": "D"}.items():
		text = text.replace(c, r)
	nfkd = unicodedata.normalize("NFKD", text)
	return "".join(c for c in nfkd if not unicodedata.combining(c))


def preprocess(text: str) -> str:
	if not isinstance(text, str):
		return ""
	text = _no_accent(text)
	text = unicodedata.normalize("NFC", text).lower()
	return re.sub(r"[^a-z0-9\s]", " ", text).strip()


def keyword_fallback(text: str) -> tuple[Optional[str], int]:
	norm = _no_accent(text.lower())
	best_cat: Optional[str] = None
	best_score = 0

	for cat, keywords in KEYWORD_MAP.items():
		score = sum(1 for kw in keywords if kw in norm)
		if score > best_score:
			best_score = score
			best_cat = cat

	return best_cat, best_score


merchant_model = None
lineitem_model = None


def _predict_model(text: str, model: Any, source_name: str) -> Optional[dict]:
	if model is None:
		return None

	clean = preprocess(text)
	if not clean:
		return None

	cat = model.predict([clean])[0]
	proba = model.predict_proba([clean])[0]
	conf = float(max(proba))
	classes = list(model.classes_)

	ranked = sorted(
		[{"category": c, "confidence": float(p)} for c, p in zip(classes, proba)],
		key=lambda x: x["confidence"],
		reverse=True,
	)

	return {
		"category": cat,
		"confidence": conf,
		"source": source_name,
		"top_candidates": ranked[:TOP_K_CANDIDATES],
	}


def classify_hybrid(text: str, prefer_lineitem: bool = False) -> dict:
	result, _ = _classify_hybrid_core(text, prefer_lineitem=prefer_lineitem)
	return result


def download_models() -> None:
	os.makedirs(MODELS_DIR, exist_ok=True)
	resolved = {
		MERCHANT_MODEL_PATH: MODELS.get("models/merchant_classifier_latest.pkl"),
		LINEITEM_MODEL_PATH: MODELS.get("models/lineitem_classifier_latest.pkl"),
	}
	for path, file_id in resolved.items():
		if os.path.exists(path):
			print(f"Already exists: {path}")
			continue
		if not MODEL_AUTO_DOWNLOAD:
			raise RuntimeError(
				"Model file missing while MODEL_AUTO_DOWNLOAD=0: " + path
			)
		print(f"Downloading {path}...")
		gdown.download(id=file_id, output=path, quiet=MODEL_DOWNLOAD_QUIET)
		if not os.path.exists(path):
			raise RuntimeError(f"Failed to download {path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
	global merchant_model, lineitem_model
	print("Server starting...")
	download_models()
	merchant_model = joblib.load(MERCHANT_MODEL_PATH)
	lineitem_model = joblib.load(LINEITEM_MODEL_PATH)
	MODEL_RUNTIME_STATUS["merchant_loaded"] = merchant_model is not None
	MODEL_RUNTIME_STATUS["lineitem_loaded"] = lineitem_model is not None
	MODEL_RUNTIME_STATUS["merchant_path"] = MERCHANT_MODEL_PATH
	MODEL_RUNTIME_STATUS["lineitem_path"] = LINEITEM_MODEL_PATH
	print("Models loaded. Server ready.")
	yield


app = FastAPI(title="Receipt ML API", version="6.0", lifespan=lifespan)


class TextBlock(BaseModel):
	text: str
	y: float
	height: float = 0.03
	x: float = 0.0
	width: float = 1.0


class ReceiptRequest(BaseModel):
	blocks: list[TextBlock]


class ClassifyRequest(BaseModel):
	text: str
	use_lineitem: bool = False
	debug: bool = False


def _classify_hybrid_core(text: str, prefer_lineitem: bool = False) -> tuple[dict, dict]:
	clean = preprocess(text)
	if not clean:
		result = {
			"category": "Khac",
			"confidence": 0.0,
			"low_confidence": True,
			"source": "fallback",
			"top_candidates": [],
		}
		debug = {
			"input_text": text,
			"clean_text": clean,
			"prefer_lineitem": prefer_lineitem,
			"merchant_model": None,
			"lineitem_model": None,
			"keyword": {"category": None, "score": 0, "boost": 0.0},
			"category_scores": {},
			"decision": {
				"final_category": "Khac",
				"final_confidence": 0.0,
				"low_confidence": True,
			},
		}
		return result, debug

	merchant_pred = _predict_model(text, merchant_model, "merchant_model")
	lineitem_pred = _predict_model(text, lineitem_model, "lineitem_model")
	kw_cat, kw_score = keyword_fallback(text)

	category_scores: dict[str, float] = {}
	candidate_pool: dict[str, float] = {}
	kw_boost = 0.0

	if merchant_pred:
		w = 0.55 if not prefer_lineitem else 0.40
		category_scores[merchant_pred["category"]] = (
			category_scores.get(merchant_pred["category"], 0.0)
			+ merchant_pred["confidence"] * w
		)
		for c in merchant_pred["top_candidates"]:
			candidate_pool[c["category"]] = max(candidate_pool.get(c["category"], 0.0), float(c["confidence"]) * w)

	if lineitem_pred:
		w = 0.75 if prefer_lineitem else 0.45
		category_scores[lineitem_pred["category"]] = (
			category_scores.get(lineitem_pred["category"], 0.0)
			+ lineitem_pred["confidence"] * w
		)
		for c in lineitem_pred["top_candidates"]:
			candidate_pool[c["category"]] = max(candidate_pool.get(c["category"], 0.0), float(c["confidence"]) * w)

	if kw_cat and kw_score > 0:
		kw_boost = min(0.30 + (0.05 * kw_score), 0.60)
		category_scores[kw_cat] = category_scores.get(kw_cat, 0.0) + kw_boost
		candidate_pool[kw_cat] = max(candidate_pool.get(kw_cat, 0.0), kw_boost)

	if not category_scores:
		result = {
			"category": "Khac",
			"confidence": 0.0,
			"low_confidence": True,
			"source": "fallback",
			"top_candidates": [],
		}
		debug = {
			"input_text": text,
			"clean_text": clean,
			"prefer_lineitem": prefer_lineitem,
			"merchant_model": merchant_pred,
			"lineitem_model": lineitem_pred,
			"keyword": {"category": kw_cat, "score": kw_score, "boost": kw_boost},
			"category_scores": {},
			"decision": {
				"final_category": "Khac",
				"final_confidence": 0.0,
				"low_confidence": True,
			},
		}
		return result, debug

	best_cat, best_score = max(category_scores.items(), key=lambda x: x[1])
	final_conf = round(min(best_score, 0.99), 3)
	low_conf = final_conf < CONFIDENCE_THRESHOLD

	merged_candidates = sorted(
		[{"category": k, "confidence": round(v, 3)} for k, v in candidate_pool.items()],
		key=lambda x: x["confidence"],
		reverse=True,
	)[:TOP_K_CANDIDATES]

	source_tags = []
	if merchant_pred:
		source_tags.append("merchant")
	if lineitem_pred:
		source_tags.append("lineitem")
	if kw_cat and kw_score > 0:
		source_tags.append("keyword")

	result = {
		"category": best_cat if not low_conf else (best_cat if kw_cat == best_cat else "Khac"),
		"confidence": final_conf,
		"low_confidence": low_conf,
		"source": "+".join(source_tags) if source_tags else "fallback",
		"top_candidates": merged_candidates,
	}

	debug = {
		"input_text": text,
		"clean_text": clean,
		"prefer_lineitem": prefer_lineitem,
		"merchant_model": merchant_pred,
		"lineitem_model": lineitem_pred,
		"keyword": {"category": kw_cat, "score": kw_score, "boost": round(kw_boost, 3)},
		"category_scores": {k: round(v, 3) for k, v in sorted(category_scores.items(), key=lambda x: x[1], reverse=True)},
		"decision": {
			"final_category": result["category"],
			"final_confidence": final_conf,
			"low_confidence": low_conf,
			"threshold": CONFIDENCE_THRESHOLD,
		},
	}
	return result, debug


def _build_breakdown_detail(breakdown: dict[str, int], total_amount: Optional[int]) -> list[dict]:
	if not breakdown:
		return []
	total = total_amount or sum(breakdown.values())
	if total <= 0:
		total = sum(breakdown.values())

	rows = []
	for cat, amount in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
		ratio = round((amount / total), 4) if total > 0 else 0.0
		rows.append({"category": cat, "amount": amount, "ratio": ratio})
	return rows


def _is_other_category_name(category: str) -> bool:
	key = preprocess(str(category or "")).replace("_", " ").strip()
	return key in {"khac", "other", "others"}


def _meaningful_category_count(breakdown: dict[str, int], total_amount: int) -> int:
	if not breakdown:
		return 0
	threshold = max(12_000, int(max(total_amount, 0) * 0.12)) if total_amount > 0 else 12_000
	return sum(
		1
		for cat, amount in breakdown.items()
		if amount >= threshold and not _is_other_category_name(cat)
	)


def _sanitize_breakdown(
	breakdown: dict[str, int],
	total_amount: int,
	merchant_category: Optional[str] = None,
) -> dict[str, int]:
	cleaned = {
		str(cat): int(amount)
		for cat, amount in (breakdown or {}).items()
		if int(amount or 0) > 0
	}

	total_ref = int(total_amount or 0)
	if total_ref <= 0:
		total_ref = sum(cleaned.values())

	if not cleaned:
		if total_ref > 0:
			fallback = merchant_category or "Khac"
			return {fallback: total_ref}
		return {}

	non_other = {cat: amt for cat, amt in cleaned.items() if not _is_other_category_name(cat)}
	dominant = max(non_other, key=non_other.get) if non_other else max(cleaned, key=cleaned.get)
	dominant_amount = cleaned.get(dominant, 0)
	tiny_threshold = max(12_000, int(total_ref * 0.12)) if total_ref > 0 else 12_000

	for cat in list(cleaned.keys()):
		if cat == dominant:
			continue
		amount = cleaned.get(cat, 0)
		is_other = _is_other_category_name(cat)
		tiny_amount = amount <= tiny_threshold
		small_vs_dominant = dominant_amount > 0 and amount <= int(dominant_amount * 0.25)
		if is_other and (tiny_amount or small_vs_dominant):
			cleaned[dominant] = cleaned.get(dominant, 0) + amount
			del cleaned[cat]

	if total_ref > 0:
		current_sum = sum(cleaned.values())
		diff = total_ref - current_sum
		if diff > 0:
			meaningful_count = _meaningful_category_count(cleaned, total_ref)
			if meaningful_count <= 1 or diff <= max(8_000, int(total_ref * 0.08)):
				cleaned[dominant] = cleaned.get(dominant, 0) + diff
			else:
				cleaned["Khac"] = cleaned.get("Khac", 0) + diff
		elif diff < 0:
			decrease = abs(diff)
			if decrease <= max(8_000, int(total_ref * 0.08)):
				cleaned[dominant] = max(0, cleaned.get(dominant, 0) - decrease)

	cleaned = {cat: amt for cat, amt in cleaned.items() if amt > 0}
	if not cleaned and total_ref > 0:
		fallback = merchant_category or "Khac"
		cleaned = {fallback: total_ref}

	return cleaned


def _infer_normal_items_for_multicat(blocks: list[dict]) -> list[dict]:
	# Reuse extractor's parser to avoid duplicate parsing heuristics.
	from extractor import TextBlock as ExtractorTextBlock, extract_items, split_zones

	text_blocks = [
		ExtractorTextBlock(
			text=b.get("text", "").strip(),
			y=float(b.get("y", 0.5)),
			height=float(b.get("height", 0.03)),
			x=float(b.get("x", 0.0)),
			width=float(b.get("width", 1.0)),
		)
		for b in blocks
		if b.get("text", "").strip()
	]
	text_blocks.sort(key=lambda t: t.y)
	_, body, _ = split_zones(text_blocks)
	items = extract_items(body)
	return [
		{
			"name": i.name,
			"quantity": i.quantity,
			"unit_price": i.unit_price,
			"total_price": i.total_price,
		}
		for i in items
	]


@app.get("/")
def root():
	return {
		"status": "ok",
		"version": "6.0",
		"classification_engine": "hybrid_v2",
		"models": MODEL_RUNTIME_STATUS,
	}


@app.get("/health")
def health():
	all_loaded = MODEL_RUNTIME_STATUS["merchant_loaded"] and MODEL_RUNTIME_STATUS["lineitem_loaded"]
	return {
		"status": "healthy" if all_loaded else "degraded",
		"models": MODEL_RUNTIME_STATUS,
	}


@app.get("/model/status")
def model_status():
	return {
		"models": MODEL_RUNTIME_STATUS,
		"ready": MODEL_RUNTIME_STATUS["merchant_loaded"] and MODEL_RUNTIME_STATUS["lineitem_loaded"],
	}


@app.post("/classify")
def classify_text(req: ClassifyRequest):
	result, debug_info = _classify_hybrid_core(req.text, prefer_lineitem=req.use_lineitem)
	if req.debug:
		result["debug"] = debug_info
	return result


@app.post("/debug/classify")
def classify_text_debug(req: ClassifyRequest):
	result, debug_info = _classify_hybrid_core(req.text, prefer_lineitem=req.use_lineitem)
	result["debug"] = debug_info
	return result


@app.post("/receipt")
def process_receipt(req: ReceiptRequest):
	from extractor import extract_receipt

	blocks = [b.model_dump() for b in req.blocks]
	receipt = extract_receipt(blocks)

	merchant_result = classify_hybrid(receipt.get("merchant_name") or "", prefer_lineitem=False)
	receipt["merchant_category"] = merchant_result["category"]
	receipt["merchant_confidence"] = merchant_result["confidence"]
	receipt["merchant_low_conf"] = merchant_result["low_confidence"]
	receipt["merchant_source"] = merchant_result["source"]
	receipt["merchant_top_candidates"] = merchant_result["top_candidates"]

	total_amount = receipt.get("total_amount") or 0

	if receipt.get("receipt_type") == "supermarket":
		breakdown: dict[str, int] = {}
		for item in receipt.get("items", []):
			r = classify_hybrid(item.get("name", ""), prefer_lineitem=True)
			item["category"] = r["category"]
			item["confidence"] = r["confidence"]
			item["low_confidence"] = r["low_confidence"]
			item["source"] = r["source"]
			item["top_candidates"] = r["top_candidates"]

			cat = r["category"]
			breakdown[cat] = breakdown.get(cat, 0) + int(item.get("total_price", 0) or 0)

		breakdown = _sanitize_breakdown(breakdown, total_amount, merchant_result["category"])
		resolved_total = total_amount if total_amount > 0 else sum(breakdown.values())

		receipt["breakdown"] = breakdown
		receipt["category_breakdown"] = _build_breakdown_detail(breakdown, resolved_total)
		receipt["is_multi_cat"] = _meaningful_category_count(breakdown, resolved_total) > 1
		receipt["classification_engine"] = "hybrid_v2"
		return receipt

	inferred_items = _infer_normal_items_for_multicat(blocks)
	inferred_breakdown: dict[str, int] = {}

	for item in inferred_items:
		r = classify_hybrid(item.get("name", ""), prefer_lineitem=True)
		item["category"] = r["category"]
		item["confidence"] = r["confidence"]
		item["source"] = r["source"]
		amount = int(item.get("total_price", 0) or 0)
		inferred_breakdown[r["category"]] = inferred_breakdown.get(r["category"], 0) + amount

	inferred_sum = sum(inferred_breakdown.values())
	coverage_ok = (total_amount <= 0) or (total_amount > 0 and (0.70 <= (inferred_sum / total_amount) <= 1.25))
	multi_ok = _meaningful_category_count(inferred_breakdown, total_amount or inferred_sum) > 1

	if inferred_items and inferred_sum > 0 and coverage_ok and multi_ok:
		inferred_breakdown = _sanitize_breakdown(inferred_breakdown, total_amount, merchant_result["category"])
		resolved_total = total_amount if total_amount > 0 else sum(inferred_breakdown.values())

		receipt["breakdown"] = inferred_breakdown
		receipt["category_breakdown"] = _build_breakdown_detail(inferred_breakdown, resolved_total)
		receipt["is_multi_cat"] = _meaningful_category_count(inferred_breakdown, resolved_total) > 1
		receipt["inferred_items"] = inferred_items
	else:
		receipt["breakdown"] = {merchant_result["category"]: total_amount}
		receipt["category_breakdown"] = _build_breakdown_detail(receipt["breakdown"], total_amount)
		receipt["is_multi_cat"] = False

	receipt["classification_engine"] = "hybrid_v2"
	return receipt


class ReceiptDebugRequest(BaseModel):
	blocks: list[TextBlock]
	debug: bool = True


@app.post("/debug/receipt")
def process_receipt_debug(req: ReceiptDebugRequest):
	from extractor import extract_receipt

	blocks = [b.model_dump() for b in req.blocks]
	receipt = extract_receipt(blocks)
	receipt["debug"] = {
		"merchant": None,
		"items": [],
		"normal_inferred_items": [],
	}

	merchant_result, merchant_debug = _classify_hybrid_core(receipt.get("merchant_name") or "", prefer_lineitem=False)
	receipt["merchant_category"] = merchant_result["category"]
	receipt["merchant_confidence"] = merchant_result["confidence"]
	receipt["merchant_low_conf"] = merchant_result["low_confidence"]
	receipt["merchant_source"] = merchant_result["source"]
	receipt["merchant_top_candidates"] = merchant_result["top_candidates"]
	receipt["debug"]["merchant"] = merchant_debug

	total_amount = receipt.get("total_amount") or 0

	if receipt.get("receipt_type") == "supermarket":
		breakdown: dict[str, int] = {}
		for item in receipt.get("items", []):
			r, r_debug = _classify_hybrid_core(item.get("name", ""), prefer_lineitem=True)
			item["category"] = r["category"]
			item["confidence"] = r["confidence"]
			item["low_confidence"] = r["low_confidence"]
			item["source"] = r["source"]
			item["top_candidates"] = r["top_candidates"]
			receipt["debug"]["items"].append({
				"name": item.get("name", ""),
				"total_price": item.get("total_price", 0),
				"classification": r_debug,
			})

			cat = r["category"]
			breakdown[cat] = breakdown.get(cat, 0) + int(item.get("total_price", 0) or 0)

		breakdown = _sanitize_breakdown(breakdown, total_amount, merchant_result["category"])
		resolved_total = total_amount if total_amount > 0 else sum(breakdown.values())

		receipt["breakdown"] = breakdown
		receipt["category_breakdown"] = _build_breakdown_detail(breakdown, resolved_total)
		receipt["is_multi_cat"] = _meaningful_category_count(breakdown, resolved_total) > 1
		receipt["classification_engine"] = "hybrid_v2"
		return receipt

	inferred_items = _infer_normal_items_for_multicat(blocks)
	inferred_breakdown: dict[str, int] = {}

	for item in inferred_items:
		r, r_debug = _classify_hybrid_core(item.get("name", ""), prefer_lineitem=True)
		item["category"] = r["category"]
		item["confidence"] = r["confidence"]
		item["source"] = r["source"]
		receipt["debug"]["normal_inferred_items"].append({
			"name": item.get("name", ""),
			"total_price": item.get("total_price", 0),
			"classification": r_debug,
		})
		amount = int(item.get("total_price", 0) or 0)
		inferred_breakdown[r["category"]] = inferred_breakdown.get(r["category"], 0) + amount

	inferred_sum = sum(inferred_breakdown.values())
	coverage_ok = (total_amount <= 0) or (total_amount > 0 and (0.70 <= (inferred_sum / total_amount) <= 1.25))
	multi_ok = _meaningful_category_count(inferred_breakdown, total_amount or inferred_sum) > 1

	if inferred_items and inferred_sum > 0 and coverage_ok and multi_ok:
		inferred_breakdown = _sanitize_breakdown(inferred_breakdown, total_amount, merchant_result["category"])
		resolved_total = total_amount if total_amount > 0 else sum(inferred_breakdown.values())

		receipt["breakdown"] = inferred_breakdown
		receipt["category_breakdown"] = _build_breakdown_detail(inferred_breakdown, resolved_total)
		receipt["is_multi_cat"] = _meaningful_category_count(inferred_breakdown, resolved_total) > 1
		receipt["inferred_items"] = inferred_items
	else:
		receipt["breakdown"] = {merchant_result["category"]: total_amount}
		receipt["category_breakdown"] = _build_breakdown_detail(receipt["breakdown"], total_amount)
		receipt["is_multi_cat"] = False

	receipt["debug"]["normal_resolution"] = {
		"inferred_sum": inferred_sum,
		"total_amount": total_amount,
		"coverage_ok": coverage_ok,
		"multi_ok": multi_ok,
		"used_inferred_breakdown": bool(inferred_items and inferred_sum > 0 and coverage_ok and multi_ok),
	}
	receipt["classification_engine"] = "hybrid_v2"
	return receipt
