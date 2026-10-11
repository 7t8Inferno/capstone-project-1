"""Train and package the capstone fraud model from creditcard.csv."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forest_model import RandomForestClassifier


FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
SEED = 42


def split_stratified(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    parts = [[], [], []]
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        train_end = int(len(indices) * 0.70)
        validation_end = train_end + int(len(indices) * 0.15)
        parts[0].extend(indices[:train_end])
        parts[1].extend(indices[train_end:validation_end])
        parts[2].extend(indices[validation_end:])
    output = [np.asarray(part, dtype=np.int64) for part in parts]
    for part in output:
        rng.shuffle(part)
    return output[0], output[1], output[2]


def best_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    sorted_labels = y_true[order]
    # Evaluate only after a complete score-tie group, matching a >= threshold rule.
    boundaries = np.r_[np.flatnonzero(np.diff(sorted_probabilities) != 0), len(order) - 1]
    cumulative_tp = np.cumsum(sorted_labels)[boundaries]
    predicted_count = boundaries + 1
    positives = max(int(y_true.sum()), 1)
    f1 = 2.0 * cumulative_tp / np.maximum(predicted_count + positives, 1)
    return float(sorted_probabilities[boundaries[int(np.argmax(f1))]])


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int8)
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    order = np.argsort(-probabilities, kind="stable")
    sorted_y = y_true[order]
    cumulative_tp = np.cumsum(sorted_y)
    ranks = np.arange(1, len(y_true) + 1)
    average_precision = float(np.sum((cumulative_tp / ranks) * sorted_y) / max(int(y_true.sum()), 1))
    ascending = np.argsort(probabilities, kind="stable")
    ranks_ascending = np.empty(len(y_true), dtype=np.float64)
    ranks_ascending[ascending] = np.arange(1, len(y_true) + 1)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    roc_auc = float((ranks_ascending[y_true == 1].sum() - positives * (positives + 1) / 2) / max(positives * negatives, 1))
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": average_precision,
        "roc_auc": roc_auc,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "test_transactions": int(len(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to creditcard.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--trees", type=int, default=45)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.data)
    if "Class" not in frame or any(column not in frame for column in FEATURES):
        raise ValueError("Expected the credit-card dataset columns: Time, V1–V28, Amount, Class")
    frame = frame[FEATURES + ["Class"]].dropna().copy()
    y = frame["Class"].astype(np.int8).to_numpy()
    X = frame[FEATURES].to_numpy(dtype=np.float32)
    train_ids, valid_ids, test_ids = split_stratified(y)

    model = RandomForestClassifier(n_estimators=args.trees, random_state=SEED)
    started = time.perf_counter()
    model.fit(X[train_ids], y[train_ids])
    training_seconds = time.perf_counter() - started

    threshold = best_f1_threshold(y[valid_ids], model.predict_proba(X[valid_ids])[:, 1])
    test_probabilities = model.predict_proba(X[test_ids])[:, 1]
    test_metrics = evaluate(y[test_ids], test_probabilities, threshold)

    bundle = {"model": model, "features": FEATURES, "threshold": threshold}
    with (args.output / "fraud_model.pkl").open("wb") as output:
        pickle.dump(bundle, output, protocol=pickle.HIGHEST_PROTOCOL)

    fraud_count = int(y.sum())
    metrics = {
        "project": "Real-Time Financial Transaction Fraud Detection",
        "dataset": "Anonymized credit-card fraud transactions supplied with this project",
        "dataset_file": "creditcard.csv",
        "rows": int(len(frame)),
        "fraud_count": fraud_count,
        "legitimate_count": int(len(frame) - fraud_count),
        "fraud_rate": float(fraud_count / len(frame)),
        "feature_count": len(FEATURES),
        "features": FEATURES,
        "model": "Random Forest Classifier",
        "model_engine": "NumPy implementation with bootstrap sampling and class-balanced Gini splits",
        "trees": args.trees,
        "max_depth": model.max_depth,
        "min_samples_leaf": model.min_samples_leaf,
        "class_weight": "balanced per bootstrap sample",
        "random_seed": SEED,
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15, "stratified": True},
        "training_rows": int(len(train_ids)),
        "validation_rows": int(len(valid_ids)),
        "training_seconds": round(training_seconds, 2),
        "decision_threshold": threshold,
        "test": test_metrics,
        "amount_mean": float(frame["Amount"].mean()),
        "amount_median": float(frame["Amount"].median()),
        "fraud_amount_median": float(frame.loc[frame["Class"].astype(int) == 1, "Amount"].median()),
        "activity_by_hour": [],
    }
    hours = (frame["Time"] // 3600).astype(int)
    for hour in range(int(hours.max()) + 1):
        at_hour = hours == hour
        metrics["activity_by_hour"].append({
            "hour": hour,
            "legitimate": int(np.sum(at_hour.to_numpy() & (y == 0))),
            "fraud": int(np.sum(at_hour.to_numpy() & (y == 1))),
        })
    (args.output / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Generate synthetic profile vectors from class-level distributions; never publish source rows.
    rng = np.random.default_rng(SEED)
    def synthetic_profiles(reference_ids, count, profile_type):
        reference = X[reference_ids]
        pca = reference[:, 1:29]
        mean = pca.mean(axis=0)
        covariance = np.cov(pca, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        transform = eigenvectors * np.sqrt(np.clip(eigenvalues, 1e-8, None))
        generated_pca = mean + rng.standard_normal((count, 28)) @ transform.T
        elapsed = rng.uniform(0, float(frame["Time"].max()), count)
        amounts = reference[:, -1]
        log_amounts = np.log1p(np.maximum(amounts, 0))
        amount_mean, amount_std = float(log_amounts.mean()), max(float(log_amounts.std()), 0.15)
        generated_amounts = np.maximum(0.0, np.expm1(rng.normal(amount_mean, amount_std, count)))
        generated = np.column_stack((elapsed, generated_pca, generated_amounts)).astype(np.float32)
        result = pd.DataFrame(generated, columns=FEATURES)
        result["profile_type"] = profile_type
        return result

    train_neg = train_ids[y[train_ids] == 0]
    train_pos = train_ids[y[train_ids] == 1]
    demo = pd.concat((
        synthetic_profiles(train_neg, 1150, "Typical profile"),
        synthetic_profiles(train_pos, 50, "Elevated profile"),
    ), ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    demo.insert(0, "transaction_id", [f"TXN-{i + 1:05d}" for i in range(len(demo))])
    demo.to_csv(args.output / "demo_stream.csv", index=False)

    print(json.dumps({"training_rows": len(train_ids), "validation_rows": len(valid_ids), "metrics": test_metrics, "training_seconds": round(training_seconds, 2)}, indent=2))


if __name__ == "__main__":
    main()
