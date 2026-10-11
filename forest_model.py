"""Small NumPy-only Random Forest classifier for the capstone demo."""

from __future__ import annotations

import numpy as np


class RandomForestClassifier:
    """Bootstrap forest with class-balanced Gini splits and randomized candidates."""

    def __init__(self, n_estimators=45, max_depth=10, min_samples_leaf=5, max_features="sqrt", random_state=42,
                 negative_sample_size=12000, candidate_quantiles=9):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.negative_sample_size = int(negative_sample_size)
        self.candidate_quantiles = int(candidate_quantiles)
        self.trees = []

    @staticmethod
    def _leaf_probability(y, weights):
        positive = float(np.sum(y == 1) * weights[1])
        negative = float(np.sum(y == 0) * weights[0])
        return positive / max(positive + negative, 1e-12)

    @staticmethod
    def _gini(negative_weight, positive_weight):
        total = negative_weight + positive_weight
        if total <= 0:
            return 0.0
        p = positive_weight / total
        return 2.0 * p * (1.0 - p)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        if len(positives) == 0 or len(negatives) == 0:
            raise ValueError("Training data must contain both classes")
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        feature_count = max(1, int(np.sqrt(n_features))) if self.max_features == "sqrt" else int(self.max_features)
        feature_count = min(feature_count, n_features)
        quantiles = np.linspace(0.1, 0.9, self.candidate_quantiles)
        self.trees = []

        for _ in range(self.n_estimators):
            positive_draws = rng.choice(positives, size=max(len(positives), 350), replace=True)
            negative_draws = rng.choice(negatives, size=min(self.negative_sample_size, len(negatives)), replace=True)
            sample = np.concatenate((positive_draws, negative_draws))
            rng.shuffle(sample)
            xs, ys = X[sample], y[sample]
            counts = np.bincount(ys, minlength=2).astype(np.float64)
            weights = {0: len(ys) / (2.0 * counts[0]), 1: len(ys) / (2.0 * counts[1])}
            nodes = []

            def build(rows, depth):
                node_id = len(nodes)
                nodes.append(None)
                node_y = ys[rows]
                probability = self._leaf_probability(node_y, weights)
                if depth >= self.max_depth or len(rows) < 2 * self.min_samples_leaf or np.all(node_y == node_y[0]):
                    nodes[node_id] = (-1, 0.0, -1, -1, probability)
                    return node_id

                parent_neg = float(np.sum(node_y == 0) * weights[0])
                parent_pos = float(np.sum(node_y == 1) * weights[1])
                parent_total = parent_neg + parent_pos
                parent_gini = self._gini(parent_neg, parent_pos)
                best_gain, best_feature, best_threshold = 0.0, -1, 0.0
                chosen_features = rng.choice(n_features, size=feature_count, replace=False)
                for feature in chosen_features:
                    column = xs[rows, feature]
                    candidates = np.unique(np.quantile(column, quantiles))
                    for threshold in candidates:
                        left_mask = column <= threshold
                        left_size = int(left_mask.sum())
                        right_size = len(rows) - left_size
                        if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                            continue
                        left_y = node_y[left_mask]
                        right_y = node_y[~left_mask]
                        left_neg = float(np.sum(left_y == 0) * weights[0])
                        left_pos = float(np.sum(left_y == 1) * weights[1])
                        right_neg = parent_neg - left_neg
                        right_pos = parent_pos - left_pos
                        child_gini = (
                            (left_neg + left_pos) * self._gini(left_neg, left_pos)
                            + (right_neg + right_pos) * self._gini(right_neg, right_pos)
                        ) / parent_total
                        gain = parent_gini - child_gini
                        if gain > best_gain:
                            best_gain, best_feature, best_threshold = gain, int(feature), float(threshold)

                if best_feature < 0:
                    nodes[node_id] = (-1, 0.0, -1, -1, probability)
                    return node_id
                left_rows = rows[xs[rows, best_feature] <= best_threshold]
                right_rows = rows[xs[rows, best_feature] > best_threshold]
                if len(left_rows) == 0 or len(right_rows) == 0:
                    nodes[node_id] = (-1, 0.0, -1, -1, probability)
                    return node_id
                nodes[node_id] = (best_feature, best_threshold, -2, -2, probability)
                left_id = build(left_rows, depth + 1)
                right_id = build(right_rows, depth + 1)
                nodes[node_id] = (best_feature, best_threshold, left_id, right_id, probability)
                return node_id

            build(np.arange(len(ys), dtype=np.int32), 0)
            self.trees.append(nodes)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        probabilities = np.zeros(len(X), dtype=np.float64)
        for tree in self.trees:
            tree_probs = np.empty(len(X), dtype=np.float64)

            def visit(node_id, row_ids):
                feature, threshold, left_id, right_id, probability = tree[node_id]
                if feature < 0:
                    tree_probs[row_ids] = probability
                    return
                mask = X[row_ids, feature] <= threshold
                visit(left_id, row_ids[mask])
                visit(right_id, row_ids[~mask])

            visit(0, np.arange(len(X), dtype=np.int32))
            probabilities += tree_probs
        probabilities /= max(len(self.trees), 1)
        return np.column_stack((1.0 - probabilities, probabilities))
