"""
extractor.py  v2
────────────────
Nhận text blocks từ ML Kit (text + tọa độ y),
tự động phát hiện loại hóa đơn rồi trả về:

Hóa đơn thông thường (normal):
{
  "receipt_type"  : "normal",
  "merchant_name" : str,
  "date"          : "YYYY-MM-DD",
  "total_amount"  : int,
  "items"         : [],
  "breakdown"     : {},
  "raw_text"      : str
}

Hóa đơn siêu thị (supermarket):
{
  "receipt_type"  : "supermarket",
  "merchant_name" : str,
  "date"          : "YYYY-MM-DD",
  "total_amount"  : int,
  "items"         : [{ name, quantity, unit_price, total_price }],
  "breakdown"     : {},   ← được điền bởi model sau
  "raw_text"      : str
}
"""

import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional


# ════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════

@dataclass
class TextBlock:
    text: str
    y: float
    height: float
    x: float = 0.0
    width: float = 1.0

@dataclass
class LineItem:
    name: str
    quantity: float = 1.0
    unit_price: int = 0
    total_price: int = 0

@dataclass
class ReceiptData:
    receipt_type: str = "normal"      # "normal" | "supermarket"
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    total_amount: Optional[int] = None
    items: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)
    raw_text: str = ""

    def to_dict(self):
        return asdict(self)


# ════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ════════════════════════════════════════════════════════════

def remove_accents(text: str) -> str:
    for c, r in {'đ': 'd', 'Đ': 'D'}.items():
        text = text.replace(c, r)
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))

def normalize(text: str) -> str:
    return remove_accents(text.lower().strip())

def clean_number(text: str) -> Optional[int]:
    text = re.sub(r'[đĐ]|vnd|vnđ', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s', '', text)
    text = re.sub(r'[.,](?=\d{3}(\D|$))', '', text)
    text = re.sub(r'[.,]', '', text)
    try:
        return int(text)
    except ValueError:
        return None

def extract_numbers_from_line(text: str) -> list:
    text = re.sub(r'[đĐ]|vnd|vnđ', '', text, flags=re.IGNORECASE)
    results = []
    for m in re.findall(r'\d{1,3}(?:[.,]\d{3})+|\d{4,}', text):
        n = clean_number(m)
        if n and n >= 100:
            results.append(n)
    return results


# ════════════════════════════════════════════════════════════
# ZONE SPLITTING
# ════════════════════════════════════════════════════════════

def split_zones(blocks: list) -> tuple:
    """Chia blocks thành header / body / footer theo tọa độ y"""
    header = [b for b in blocks if b.y < 0.25]
    body   = [b for b in blocks if 0.25 <= b.y < 0.75]
    footer = [b for b in blocks if b.y >= 0.75]
    return header, body, footer


# ════════════════════════════════════════════════════════════
# RECEIPT TYPE DETECTION
# ════════════════════════════════════════════════════════════

# Từ khóa gợi ý hóa đơn siêu thị / cửa hàng nhiều sản phẩm
_SUPERMARKET_KW = [
    # Tên loại cửa hàng
    'sieu thi', 'supermarket', 'hypermarket',
    'tap hoa', 'grocery', 'minimart', 'convenience',
    'vinmart', 'coopmart', 'bigc', 'lotte mart',
    'aeon', 'emart', 'mega market',
    'bach hoa xanh', 'winmart',
    # Cột tiêu đề bảng sản phẩm
    'ten hang', 'ten sp', 'san pham', 'mat hang',
    'so luong', 's.luong', 'sl', 'qty', 'quantity',
    'don gia', 'd.gia', 'unit price',
    'thanh tien', 'subtotal',
    # Ký hiệu phân cách cột
    '---', '===', '___',
]

# Từ khóa gợi ý hóa đơn thông thường (1 dịch vụ)
_NORMAL_KW = [
    'cafe', 'coffee', 'tra sua', 'bubble tea',
    'restaurant', 'nha hang',
    'taxi', 'grab', 'be car',
    'cinema', 'cgv', 'bhd',
    'spa', 'nail', 'salon',
    'phong kham', 'nha thuoc',
    'hoc phi', 'khoa hoc',
]

def detect_receipt_type(blocks: list, body_items_count: int) -> str:
    """
    Phát hiện loại hóa đơn dựa trên:
    1. Số lượng item parse được trong body
    2. Keyword trong toàn bộ text
    3. Sự xuất hiện của cột bảng (tiêu đề SL/Qty/Đơn giá)

    Trả về: "supermarket" hoặc "normal"
    """
    full_text = ' '.join(normalize(b.text) for b in blocks)

    # Tìm keyword siêu thị
    supermarket_score = sum(1 for kw in _SUPERMARKET_KW if kw in full_text)
    normal_score      = sum(1 for kw in _NORMAL_KW      if kw in full_text)

    # Nếu parse được >= 3 sản phẩm → chắc chắn siêu thị
    if body_items_count >= 3:
        return "supermarket"

    # Nếu có keyword siêu thị rõ ràng
    if supermarket_score >= 2:
        return "supermarket"

    # Nếu parse được 1-2 sản phẩm và có keyword siêu thị
    if body_items_count >= 1 and supermarket_score >= 1:
        return "supermarket"

    return "normal"


# ════════════════════════════════════════════════════════════
# MERCHANT EXTRACTION
# ════════════════════════════════════════════════════════════

_NOT_MERCHANT_KW = [
    'duong', 'pho', 'quan', 'huyen', 'tp.', 'p.', 'q.',
    'tel', 'phone', 'hotline', 'email', 'www', 'http',
    'mst', 'tax', 'gpkd', 'so gkd',
    'hoa don', 'invoice', 'receipt', 'bill',
    'ngay', 'gio', 'date', 'time',
    'so hd', 'ma hd', 'order',
    '---', '===',
]

def extract_merchant(header_blocks: list) -> Optional[str]:
    candidates = []
    for b in header_blocks:
        text = b.text.strip()
        norm = normalize(text)
        if len(text) < 3: continue
        if re.match(r'^[\d\s.,\-/:()+@#]+$', text): continue
        if any(kw in norm for kw in _NOT_MERCHANT_KW): continue
        candidates.append(b)

    if not candidates:
        return header_blocks[0].text.strip() if header_blocks else None

    # Block có height lớn nhất = chữ to nhất = tên cửa hàng
    return max(candidates, key=lambda b: b.height).text.strip()


# ════════════════════════════════════════════════════════════
# DATE EXTRACTION
# ════════════════════════════════════════════════════════════

_DATE_PATTERNS = [
    (r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b', 'dmy'),
    (r'\b(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})\b', 'ymd'),
    (r'\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})\b',  'dmy2'),
    (r'ngay\s+(\d{1,2})\s+thang\s+(\d{1,2})\s+nam\s+(\d{4})', 'dmy'),
]

def extract_date(all_blocks: list) -> Optional[str]:
    full = normalize(' '.join(b.text for b in all_blocks))
    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, full)
        if not m: continue
        try:
            g = [int(x) for x in m.groups()]
            if fmt == 'dmy':   d, mo, y = g[0], g[1], g[2]
            elif fmt == 'ymd': y, mo, d = g[0], g[1], g[2]
            elif fmt == 'dmy2': d, mo, y = g[0], g[1], g[2] + 2000
            else: continue
            if not (1 <= d <= 31 and 1 <= mo <= 12 and 2000 <= y <= 2100): continue
            return f"{y:04d}-{mo:02d}-{d:02d}"
        except Exception:
            continue
    return None


# ════════════════════════════════════════════════════════════
# TOTAL EXTRACTION
# ════════════════════════════════════════════════════════════

_TOTAL_KW = [
    'tong cong', 'tong tien', 'thanh tien',
    'phai thanh toan', 'so tien thanh toan',
    'grand total', 'total amount', 'total', 'amount due',
]
_NOT_TOTAL_KW = [
    'giam gia', 'discount', 'khuyen mai', 'voucher',
    'tien mat', 'cash', 'tien thua', 'change',
    'thue', 'vat', 'phi phuc vu', 'tich luy', 'diem',
]

def extract_total(footer_blocks: list, all_blocks: list) -> Optional[int]:
    candidates = []
    for b in footer_blocks:
        norm = normalize(b.text)
        if any(kw in norm for kw in _NOT_TOTAL_KW): continue
        if any(kw in norm for kw in _TOTAL_KW):
            nums = extract_numbers_from_line(b.text)
            if nums: candidates.append(max(nums))

    if candidates:
        return max(candidates)

    # Fallback: số lớn nhất trong toàn bộ hóa đơn
    all_nums = []
    for b in all_blocks:
        norm = normalize(b.text)
        if any(kw in norm for kw in _NOT_TOTAL_KW): continue
        all_nums.extend(extract_numbers_from_line(b.text))

    valid = [n for n in all_nums if 1_000 <= n <= 100_000_000]
    return max(valid) if valid else None


# ════════════════════════════════════════════════════════════
# LINE ITEM EXTRACTION (chỉ dùng cho supermarket)
# ════════════════════════════════════════════════════════════

# Từ khóa dòng cần BỎ QUA — quan trọng để tránh false positive
_SKIP_ITEM_KW = [
    # Tổng tiền / footer
    'tong', 'total', 'thanh tien', 'giam gia', 'discount',
    'thue', 'vat', 'tien mat', 'cash', 'tien thua',
    # Header hóa đơn
    'cam on', 'thank', 'hotline', 'dia chi', 'mst',
    'phuc vu', 'tich diem', 'hoa don', 'invoice',
    # Tiêu đề cột bảng (dòng header không phải sản phẩm)
    'ten hang', 'ten sp', 'so luong', 'don gia',
    'qty', 'unit price', 'amount',
    # Thông tin cửa hàng
    'tel', 'phone', 'email', 'www', 'ngay', 'gio',
    'ma hd', 'so hd', 'barcode',
    # Phân cách
    '---', '===', '___',
]

# Dòng THỰC SỰ là địa chỉ/metadata (pattern)
_METADATA_PATTERNS = [
    re.compile(r'\b(duong|pho|quan|huyen|tinh|tp)\b', re.IGNORECASE),
    re.compile(r'\b(tel|phone|dt|fax)\s*[:\.]?\s*[\d\-\+\(\)]{6,}'),
    re.compile(r'(mst|tax id|gpkd)\s*[:\.]?\s*\d'),
    re.compile(r'https?://|www\.'),
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
]

def _is_metadata_line(text: str) -> bool:
    """Kiểm tra dòng có phải địa chỉ / SĐT / metadata không"""
    norm = normalize(text)
    if any(kw in norm for kw in _SKIP_ITEM_KW):
        return True
    for pattern in _METADATA_PATTERNS:
        if pattern.search(text):
            return True
    return False

def _parse_item(line: str) -> Optional[LineItem]:
    """
    Parse 1 dòng text thành LineItem.
    Hỗ trợ 5 format phổ biến của hóa đơn Việt Nam.
    """
    line = line.strip()

    # P1: Tab-separated — hóa đơn VAT chuẩn
    # VD: "Sua tuoi Vinamilk\t1\t28500\t28500"
    parts = line.split('\t')
    if len(parts) >= 3:
        try:
            name  = _clean_name(parts[0])
            qty   = float(parts[1]) if len(parts) > 3 else 1.0
            total = clean_number(parts[-1]) or 0
            unit  = clean_number(parts[-2]) if len(parts) > 3 else 0
            if total >= 100 and len(name) >= 2:
                return LineItem(name=name, quantity=qty,
                                unit_price=unit, total_price=total)
        except Exception:
            pass

    # Split theo 2+ spaces làm delimiter
    segs = re.split(r'\s{2,}', line)

    if len(segs) >= 2:
        right = segs[-1]
        left  = ' '.join(segs[:-1]).strip()
        total = clean_number(right)

        if total and total >= 100:
            # P2: "qty  Tên  giá"
            m = re.match(r'^(\d{1,2})\s+(.+)$', left)
            if m:
                name = _clean_name(m.group(2))
                qty  = float(m.group(1))
                if len(name) >= 2:
                    return LineItem(name=name, quantity=qty,
                                    unit_price=int(total/qty),
                                    total_price=total)
            # P3: "Tên x qty"
            m = re.match(r'^(.+?)\s+x\s*(\d+)$', left, re.IGNORECASE)
            if m:
                name = _clean_name(m.group(1))
                qty  = float(m.group(2))
                if len(name) >= 2:
                    return LineItem(name=name, quantity=qty,
                                    unit_price=int(total/qty),
                                    total_price=total)
            # P5: "Tên  giá" — không có qty
            name = _clean_name(left)
            if len(name) >= 2 and not re.match(r'^[\d.,]+$', name):
                return LineItem(name=name, quantity=1.0,
                                unit_price=total, total_price=total)

    # P4: "Tên qty unit total" — cột cố định không có delimiter
    m = re.match(r'^(.+?)\s+(\d{1,2})\s+(\d[\d.,]+)\s+(\d[\d.,]+)$', line)
    if m:
        name  = _clean_name(m.group(1))
        qty   = float(m.group(2))
        unit  = clean_number(m.group(3)) or 0
        total = clean_number(m.group(4)) or 0
        if total >= 100 and len(name) >= 2:
            return LineItem(name=name, quantity=qty,
                            unit_price=unit, total_price=total)
    return None

def _clean_name(name: str) -> str:
    name = re.sub(r'^[\-*.#\s]+', '', name)
    name = re.sub(r'[\-*.#\s]+$', '', name)
    name = re.sub(r'^[A-Z]{0,3}\d{3,}\s+', '', name)  # bỏ mã SP ở đầu
    return name.strip()

def extract_items(body_blocks: list) -> list:
    """
    Tách danh sách sản phẩm từ vùng body.
    Có nhiều lớp filter để tránh nhầm địa chỉ/metadata thành sản phẩm.
    """
    results = []
    lines = [b.text.strip() for b in body_blocks if b.text.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Lớp filter 1: metadata / skip keywords
        if _is_metadata_line(line):
            i += 1; continue

        # Lớp filter 2: dòng quá ngắn hoặc chỉ toàn số/ký tự đặc biệt
        if len(line) < 3 or re.match(r'^[\d\s.,\-/:()+]+$', line):
            i += 1; continue

        # Thử parse dòng đơn
        item = _parse_item(line)
        if item:
            results.append(item)
            i += 1
            continue

        # Thử ghép 2 dòng (tên ở trên, giá ở dưới)
        if i + 1 < len(lines) and not _is_metadata_line(lines[i+1]):
            combined = f"{line}  {lines[i+1]}"
            item = _parse_item(combined)
            if item:
                results.append(item)
                i += 2
                continue

        # Không parse được → bỏ qua (không thêm tên rỗng vào list)
        i += 1

    return results


# ════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def extract_receipt(blocks: list) -> dict:
    """
    Input : list of { text, y, height, x?, width? }  ← từ ML Kit
    Output: ReceiptData.to_dict()

    receipt_type = "normal"      → hóa đơn 1 dịch vụ (cafe, grab, ...)
    receipt_type = "supermarket" → hóa đơn nhiều sản phẩm
    """
    text_blocks = [
        TextBlock(
            text   = b.get('text', '').strip(),
            y      = float(b.get('y', 0.5)),
            height = float(b.get('height', 0.03)),
            x      = float(b.get('x', 0.0)),
            width  = float(b.get('width', 1.0)),
        )
        for b in blocks
        if b.get('text', '').strip()
    ]

    if not text_blocks:
        return ReceiptData().to_dict()

    text_blocks.sort(key=lambda b: b.y)
    header, body, footer = split_zones(text_blocks)

    # Trích xuất items trước để detect loại hóa đơn
    raw_items = extract_items(body)

    receipt_type = detect_receipt_type(text_blocks, len(raw_items))

    # Nếu là hóa đơn thông thường → không giữ items
    # (tránh nhầm các dòng phụ thành sản phẩm)
    final_items = raw_items if receipt_type == "supermarket" else []

    result = ReceiptData(
        receipt_type  = receipt_type,
        merchant_name = extract_merchant(header),
        date          = extract_date(text_blocks),
        total_amount  = extract_total(footer, text_blocks),
        items         = [asdict(item) for item in final_items],
        breakdown     = {},   # được điền bởi model sau khi classify
        raw_text      = '\n'.join(b.text for b in text_blocks),
    )
    return result.to_dict()


# ════════════════════════════════════════════════════════════
# SELF TEST
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json

    def run_test(label, blocks):
        r = extract_receipt(blocks)
        print(f"\n{'='*50}")
        print(f"📋 {label}")
        print(f"{'='*50}")
        print(f"  Type     : {r['receipt_type']}")
        print(f"  Merchant : {r['merchant_name']}")
        print(f"  Date     : {r['date']}")
        total = r['total_amount']
        print(f"  Total    : {total:,}đ" if total else "  Total    : None")
        print(f"  Items    : {len(r['items'])}")
        for item in r['items']:
            print(f"    {item['name']:<35} {item['total_price']:>8,}đ")

    # Test 1: Hóa đơn siêu thị
    run_test("Hóa đơn siêu thị VinMart", [
        {'text': 'VINMART SUPERMARKET',                   'y': 0.02, 'height': 0.06},
        {'text': '123 Nguyen Hue Q1 HCM',                'y': 0.08, 'height': 0.03},
        {'text': 'Tel: 028-1234-5678',                   'y': 0.13, 'height': 0.03},
        {'text': '14/06/2024 10:35',                     'y': 0.20, 'height': 0.03},
        {'text': 'STT  Ten hang         SL  Don gia  TT', 'y': 0.26, 'height': 0.03},
        {'text': 'Mi goi Hao Hao\t3\t5,000\t15,000',     'y': 0.31, 'height': 0.03},
        {'text': 'Sua tuoi Vinamilk 1L        28,500',   'y': 0.36, 'height': 0.03},
        {'text': 'Dau goi Clear 650ml         95,000',   'y': 0.41, 'height': 0.03},
        {'text': 'Bot giat Omo 3kg           185,000',   'y': 0.46, 'height': 0.03},
        {'text': 'Vitamin C Nature Made      120,000',   'y': 0.51, 'height': 0.03},
        {'text': 'Tong cong:   443,500',                 'y': 0.78, 'height': 0.03},
        {'text': 'Thanh toan:  443,500',                 'y': 0.85, 'height': 0.03},
    ])

    # Test 2: Hóa đơn thông thường (cafe)
    run_test("Hóa đơn Highlands Coffee", [
        {'text': 'HIGHLANDS COFFEE',      'y': 0.02, 'height': 0.07},
        {'text': 'Vincom Center Q1',      'y': 0.10, 'height': 0.03},
        {'text': 'Tel: 028-9999-8888',    'y': 0.15, 'height': 0.03},
        {'text': '14/06/2024 09:15',      'y': 0.21, 'height': 0.03},
        {'text': 'Bac xiu nong',          'y': 0.40, 'height': 0.03},
        {'text': 'Banh mi thit nguoi',    'y': 0.48, 'height': 0.03},
        {'text': 'Tong cong:  75,000',    'y': 0.80, 'height': 0.03},
        {'text': 'Thanh toan: 75,000',    'y': 0.87, 'height': 0.03},
    ])

    # Test 3: Hóa đơn nhà thuốc (normal — không phải siêu thị)
    run_test("Hóa đơn nhà thuốc Pharmacity", [
        {'text': 'NHA THUOC PHARMACITY',          'y': 0.02, 'height': 0.06},
        {'text': '45 Le Loi Q1 TPHCM',            'y': 0.08, 'height': 0.03},
        {'text': '15/06/2024 14:30',              'y': 0.18, 'height': 0.03},
        {'text': 'Vitamin C 1000mg       85,000', 'y': 0.30, 'height': 0.03},
        {'text': 'Omega 3 DHC           120,000', 'y': 0.37, 'height': 0.03},
        {'text': 'Khau trang y te        25,000', 'y': 0.44, 'height': 0.03},
        {'text': 'Tong cong: 230,000',            'y': 0.78, 'height': 0.03},
    ])
