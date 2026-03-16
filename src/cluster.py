from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def find_best_k(
    X: pd.DataFrame,
    k_range: Iterable[int] = range(2, 9),
) -> Tuple[int, pd.DataFrame]:
    """Evaluate KMeans across candidate cluster counts and return the best k."""
    scores = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append({"k": k, "silhouette_score": score})

    scores_df = pd.DataFrame(scores)
    best_row = scores_df.sort_values("silhouette_score", ascending=False).iloc[0]
    return int(best_row["k"]), scores_df


def fit_kmeans(X: pd.DataFrame, n_clusters: int) -> Tuple[KMeans, pd.Series]:
    """Fit the final KMeans model and return cluster labels."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = pd.Series(model.fit_predict(X), index=X.index, name="cluster")
    return model, labels


def run_pca(X: pd.DataFrame) -> pd.DataFrame:
    """Project the scaled features into two dimensions for visualization."""
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(X)
    return pd.DataFrame(coordinates, columns=["pca_1", "pca_2"], index=X.index)
