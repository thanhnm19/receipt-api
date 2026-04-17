import argparse
import csv
import os
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


def remove_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = remove_accents(text)
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_category(category: str) -> str:
    value = (category or "").strip()
    if not value:
        return "Khac"

    value = value.replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value


@dataclass
class Sample:
    text: str
    category: str


def load_samples(csv_path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text", "category"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"File {csv_path} must contain columns: text, category"
            )
        for row in reader:
            text = preprocess_text(row.get("text", ""))
            category = normalize_category(row.get("category", ""))
            if len(text) < 2:
                continue
            samples.append(Sample(text=text, category=category))
    if not samples:
        raise ValueError(f"No valid rows found in {csv_path}")
    return samples


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "features",
                FeatureUnion(
                    transformer_list=[
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(2, 5),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=120000,
                            ),
                        ),
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                token_pattern=r"(?u)\\b\\w+\\b",
                                min_df=1,
                                sublinear_tf=True,
                                max_features=60000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=800,
                    multi_class="auto",
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=None,
                ),
            ),
        ]
    )


def oversample_minor_classes(
    x: list[str],
    y: list[str],
    seed: int,
    min_samples_per_class: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = {}
    for text, category in zip(x, y):
        by_class.setdefault(category, []).append(text)

    new_x = list(x)
    new_y = list(y)
    for category, texts in by_class.items():
        if not texts:
            continue
        target = max(min_samples_per_class, len(texts))
        needed = target - len(texts)
        for _ in range(needed):
            new_x.append(rng.choice(texts))
            new_y.append(category)

    return new_x, new_y


def evaluate(model: Pipeline, x_test: list[str], y_test: list[str], label: str) -> None:
    y_pred = model.predict(x_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[{label}] Evaluation")
    print(f"Macro F1   : {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))


def split_xy(samples: Iterable[Sample]) -> tuple[list[str], list[str]]:
    x = [s.text for s in samples]
    y = [s.category for s in samples]
    return x, y


def train_one(
    samples: list[Sample],
    out_path: str,
    val_ratio: float,
    seed: int,
    label: str,
    backup_dir: str | None,
    min_samples_per_class: int,
    enable_tuning: bool,
) -> None:
    x, y = split_xy(samples)
    class_counts = Counter(y)
    rare_classes = {c for c, n in class_counts.items() if n < 2}

    if rare_classes:
        print(
            f"[{label}] Found classes with <2 samples: {sorted(rare_classes)}. "
            "These classes are kept, but validation split will fallback to non-stratified."
        )

    stratify = y if not rare_classes else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify,
    )

    x_train, y_train = oversample_minor_classes(
        x_train,
        y_train,
        seed=seed,
        min_samples_per_class=min_samples_per_class,
    )
    print(f"[{label}] Train size after oversampling: {len(x_train)}")

    model = build_pipeline()
    if enable_tuning:
        min_class_count = min(Counter(y_train).values())
        if min_class_count >= 2:
            folds = min(5, min_class_count)
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            param_grid = {
                "clf__C": [1.5, 3.0, 6.0, 10.0],
                "clf__class_weight": ["balanced", None],
            }
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                verbose=0,
            )
            search.fit(x_train, y_train)
            model = search.best_estimator_
            print(f"[{label}] Best params: {search.best_params_}")
            print(f"[{label}] Best CV macro F1: {search.best_score_:.4f}")
        else:
            print(f"[{label}] Skip tuning because training labels are too small for CV")
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)

    evaluate(model, x_test, y_test, label=label)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if backup_dir and os.path.exists(out_path):
        os.makedirs(backup_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            backup_dir,
            f"{os.path.basename(out_path)}.{ts}.bak",
        )
        os.replace(out_path, backup_path)
        print(f"[{label}] Backed up old model -> {backup_path}")

    joblib.dump(model, out_path)
    print(f"[{label}] Saved model -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train receipt category models (merchant + lineitem) from CSV datasets."
    )
    parser.add_argument(
        "--merchant-csv",
        default="data/merchant_train.csv",
        help="Path to merchant training CSV with columns: text, category",
    )
    parser.add_argument(
        "--lineitem-csv",
        default="data/lineitem_train.csv",
        help="Path to line-item training CSV with columns: text, category",
    )
    parser.add_argument(
        "--merchant-out",
        default="models/merchant_classifier_latest.pkl",
        help="Output path for merchant model",
    )
    parser.add_argument(
        "--lineitem-out",
        default="models/lineitem_classifier_latest.pkl",
        help="Output path for line-item model",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--backup-dir",
        default="models/backups",
        help="Directory to keep backup of previous model files",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=45,
        help="Oversample each class in training split to at least this many samples",
    )
    parser.add_argument(
        "--disable-tuning",
        action="store_true",
        help="Disable CV-based hyperparameter tuning for faster training",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    merchant_samples = load_samples(args.merchant_csv)
    lineitem_samples = load_samples(args.lineitem_csv)

    print(f"Merchant samples: {len(merchant_samples)}")
    print(f"Line-item samples: {len(lineitem_samples)}")

    train_one(
        samples=merchant_samples,
        out_path=args.merchant_out,
        val_ratio=args.val_ratio,
        seed=args.seed,
        label="merchant",
        backup_dir=args.backup_dir,
        min_samples_per_class=args.min_samples_per_class,
        enable_tuning=not args.disable_tuning,
    )

    train_one(
        samples=lineitem_samples,
        out_path=args.lineitem_out,
        val_ratio=args.val_ratio,
        seed=args.seed,
        label="lineitem",
        backup_dir=args.backup_dir,
        min_samples_per_class=args.min_samples_per_class,
        enable_tuning=not args.disable_tuning,
    )

    print("\nDone. Restart API to load newly trained models.")


if __name__ == "__main__":
    main()
