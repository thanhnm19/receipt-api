# receipt-api

API trich xuat hoa don + phan loai category bang hybrid model.

## Train lai model backend (tach biet hoan toan voi app)

Ban co the train lai model ma khong can sua app Android.
App chi goi API, con model nam va duoc train trong thu muc receipt-api.

### 1) Chuan bi data train

Tao 2 file CSV (UTF-8) trong thu muc `data/`:

- `data/merchant_train.csv`
- `data/lineitem_train.csv`

Format bat buoc (2 cot):

```csv
text,category
"tra sua tran chau",An_uong
"cat toc goi dau",Lam_dep
```

Luu y:
- `category` nen dung dung schema hien tai: `An_uong`, `Di_lai`, `Hoc_tap`, `Gia_dinh`, `Suc_khoe`, `Lam_dep`, `Thu_cung`, `Giai_tri`, `Mua_sam`, `Du_lich`, `Khac`.
- Nen co it nhat 30-50 mau moi category cho vong dau.
- Repo da co file mau de tham khao:
	- `data/merchant_train.sample.csv`
	- `data/lineitem_train.sample.csv`

### 2) Chay train

PowerShell:

```powershell
cd receipt-api
pip install -r requirements.txt
python train_models.py --merchant-csv data/merchant_train.csv --lineitem-csv data/lineitem_train.csv
```

Script se:
- Tien xu ly text
- Chia train/validation
- In metric (Macro F1, Weighted F1, classification report)
- Backup model cu vao `models/backups/`
- Ghi model moi vao:
	- `models/merchant_classifier_latest.pkl`
	- `models/lineitem_classifier_latest.pkl`

### 3) Nap model moi

Khoi dong lai API sau khi train:

```powershell
uvicorn main:app --reload --port 8000
```

Kiem tra:
- `GET /model/status`
- `POST /debug/classify`
- `POST /debug/receipt`

### 4) Vong lap cai thien (khuyen nghi)

1. Lay mau sai tu `ocr_category_feedbacks` (predicted vs corrected).
2. Dua cac mau sai vao CSV train.
3. Train lai.
4. So sanh metric va test lai `/debug/receipt`.
5. Chi deploy model moi khi metric va test thuc te tot hon.

## Cach don gian de luu model va chay lai o may khac

Muc tieu: ban co the copy project qua may moi, chay 2 buoc la len.

Buoc 1: tai model va luu vao thu muc models

PowerShell:

powershell -ExecutionPolicy Bypass -File .\prepare_models.ps1

Buoc 2: chay API bang model da luu san

PowerShell:

powershell -ExecutionPolicy Bypass -File .\run_api.ps1

Luu y:
- prepare_models.ps1 se tao .venv, cai dependency, tai model.
- run_api.ps1 se dat MODEL_AUTO_DOWNLOAD=0 de bat buoc dung model local.
- Ban co the copy nguyen thu muc du an sang may khac va chay lai y nhu tren.

## Bien cau hinh model

- MODELS_DIR: thu muc chua model, mac dinh la models
- MERCHANT_MODEL_PATH: duong dan model merchant
- LINEITEM_MODEL_PATH: duong dan model line item
- MODEL_AUTO_DOWNLOAD: 1 de tu tai, 0 de chi dung file local

Kiem tra trang thai model:

GET /model/status

## Chay local

1. Cai dependency

pip install -r requirements.txt

2. Chay server

uvicorn main:app --reload --port 8000

Model se duoc auto-download vao thu muc `models/` khi startup.

## Endpoint debug de de fix

### 1) Debug classify text ngan

- Endpoint: `POST /debug/classify`
- Muc dich: xem model nao de xuat gi, keyword boost bao nhieu, score hop nhat ra sao.

Sample:

```bash
curl -X POST http://127.0.0.1:8000/debug/classify \
	-H "Content-Type: application/json" \
	-d '{
		"text": "sua tuoi vinamilk 1l",
		"use_lineitem": true
	}'
```

Truong debug quan trong:
- `debug.merchant_model.top_candidates`
- `debug.lineitem_model.top_candidates`
- `debug.keyword` (category/score/boost)
- `debug.category_scores`
- `debug.decision`

### 2) Debug full receipt

- Endpoint: `POST /debug/receipt`
- Muc dich: xem tung item duoc classify the nao va tai sao ra breakdown cuoi.

Sample:

```bash
curl -X POST http://127.0.0.1:8000/debug/receipt \
	-H "Content-Type: application/json" \
	-d '{
		"blocks": [
			{"text":"VINMART SUPERMARKET","y":0.02,"height":0.06},
			{"text":"14/06/2024 10:35","y":0.20,"height":0.03},
			{"text":"Mi goi Hao Hao  15,000","y":0.31,"height":0.03},
			{"text":"Sua tuoi Vinamilk 1L  28,500","y":0.36,"height":0.03},
			{"text":"Tong cong: 43,500","y":0.85,"height":0.03}
		]
	}'
```

Truong debug quan trong:
- `debug.merchant`
- `debug.items[]` (supermarket)
- `debug.normal_inferred_items[]` (normal)
- `debug.normal_resolution`

## Cach test xem model hoat dong tot chua

## A. Test dung chuc nang (functional)

1. Hoa don normal chi 1 dich vu
- Ky vong: `is_multi_cat = false`
- `breakdown` co 1 category chinh

2. Hoa don supermarket nhieu loai hang
- Ky vong: `is_multi_cat = true`
- `category_breakdown` co nhieu category

3. Hoa don co text la/khong ro
- Ky vong: co item/category roi vao `Khac`
- `low_confidence = true` o cac item kho

## B. Test chat luong model (quality)

Tao 1 tap kiem thu 50-100 mau da gan nhan that:
- 30 mau normal
- 30 mau supermarket
- phan con lai la edge case (OCR loi, viet tat, ten moi)

Do cac chi so:
- Accuracy category cho merchant
- Accuracy category cho line item
- Ty le du doan `Khac`
- Ty le `low_confidence`
- Sai lech tong tien theo breakdown:

Cong thuc:

```text
amount_coverage = sum(category_amount_du_doan) / total_amount_hoa_don
```

Muc tieu goi y:
- Merchant accuracy >= 90%
- Item accuracy >= 80% (luc dau)
- `amount_coverage` trong khoang 0.9 -> 1.1 voi hoa don supermarket ro rang

## C. Cach fix khi thay sai

1. Sai category nhung top-2 dung:
- Giam/doi weight hop nhat trong `classify_hybrid`

2. Sai vi ten moi, model chua hoc:
- Bo sung tu khoa vao `KEYWORD_MAP`
- Them du lieu train cho ten moi

3. Nhieu item roi vao `Khac`:
- Kiem tra `CONFIDENCE_THRESHOLD` (mac dinh 0.40)
- Thu ha threshold nhe (0.35) de so sanh

4. Breakdown lech tong tien:
- Kiem tra OCR parser trong `extractor.py`
- Soat `debug.items[]` de tim item parse sai total_price

## Bien moi truong de tune nhanh

- `CONFIDENCE_THRESHOLD` (default `0.40`)
- `TOP_K_CANDIDATES` (default `3`)

VD:

set CONFIDENCE_THRESHOLD=0.35
set TOP_K_CANDIDATES=5
uvicorn main:app --reload --port 8000

## Deploy de app Kotlin goi

Ban co 2 cach de app Kotlin su dung:

1. Local server cung may tinh
- Chay run_api.ps1
- Trong app Kotlin, base URL la http://<ip_may_tinh>:8000

2. Cloud server (Render)
- Giu start command: uvicorn main:app --host 0.0.0.0 --port $PORT
- Khuyen nghi de MODEL_AUTO_DOWNLOAD=1 neu khong dong goi model theo image
- App Kotlin dung base URL Render cua ban

## Cach don gian nhat (khuyen nghi): Google Cloud Run

Ly do:
- Don gian: 1 script deploy, khong can tu quan ly server
- On dinh: co HTTPS URL public de app Android goi truc tiep
- Giam timeout lan dau: model duoc dong goi san trong Docker image

### 1) Cai va dang nhap gcloud (mot lan)

PowerShell:

gcloud auth login

Neu ban chua tao project:

gcloud projects create <PROJECT_ID>

### 2) Deploy bang 1 lenh

PowerShell:

powershell -ExecutionPolicy Bypass -File .\deploy_cloud_run.ps1 -ProjectId <PROJECT_ID>

Neu bi loi khong deploy duoc, kiem tra ngay Billing:

gcloud billing projects describe <PROJECT_ID>

Neu `billingEnabled: false` thi bat Billing cho project tai:

https://console.cloud.google.com/billing/linkedaccount?project=<PROJECT_ID>

Sau do chay lai lenh deploy.

## Khong can Billing (de demo nho): Cloudflare Tunnel

Neu ban khong bat duoc Billing, cach don gian nhat de demo la:
- API chay tren may ban
- Cloudflare Tunnel tao URL public tam thoi
- App Android goi URL nay nhu API cloud

### 1) Cai cloudflared

Tai va cai tai:

https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/

Kiem tra:

cloudflared --version

### 2) Chay 1 lenh trong receipt-api

PowerShell:

powershell -ExecutionPolicy Bypass -File .\deploy_demo_cloudflare.ps1

Script se tu dong:
- tao .venv va cai requirements
- chay FastAPI local (port 8000)
- mo tunnel public va in URL `https://...trycloudflare.com`

### 3) Cau hinh app Android

Dat BASE_URL = URL tunnel vua in ra.

Vi du:

https://abc-def-123.trycloudflare.com

Sau do app goi:
- GET /model/status
- POST /receipt

Luu y cho demo:
- URL trycloudflare la tam thoi, moi lan chay co the doi
- Khi tat script thi URL het hieu luc

Script se:
- build image
- dong goi model vao image (khong can runtime download)
- deploy len Cloud Run
- in ra URL API

Mac dinh da cau hinh:
- timeout: 300 giay
- min instances: 1 (giu warm de han che cold start)
- memory: 1Gi

Ban co the doi timeout neu can:

powershell -ExecutionPolicy Bypass -File .\deploy_cloud_run.ps1 -ProjectId <PROJECT_ID> -TimeoutSeconds 600

### 3) App Android chi can doi base URL

Vi du URL sau deploy:

https://receipt-ml-api-xxxxx-uc.a.run.app

Trong app Kotlin:
- GET /model/status de check ready=true
- POST /receipt de lay category/breakdown

Ket qua can dung trong app:
- merchant_category
- is_multi_cat
- breakdown
- category_breakdown

## Goi tu Kotlin rat don gian

De an toan, app Kotlin nen check model ready truoc:

1. GET /model/status
2. Neu ready=true thi moi POST /receipt

Goi y response can dung trong Kotlin:
- is_multi_cat
- breakdown
- category_breakdown
- merchant_category
"# receipt-api" 
