# nym_pipeline.py
import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    adjusted_rand_score
)
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, DBSCAN

# classifiers for proxy separability
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    import hdbscan
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

# Feature list (cleaned)
FEATURES = [
    "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
    "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Mean", "Bwd IAT Mean",
    "Down/Up Ratio",
    "Pkt Size Avg", "Pkt Len Std",
    "Fwd Seg Size Avg", "Bwd Seg Size Avg",
]

# ------------------------------
# Data utilities
# ------------------------------
def load_nym_dataset(csv_path: str, features: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep only features, drop NA
    df = df[features].dropna().reset_index(drop=True)
    return df

def standardize(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X.values)

# Safe silhouette: if only one cluster or all noise, return NaN
def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    labels_unique = np.unique(labels)
    if len(labels_unique) <= 1:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")

def compute_internal_indices(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    # remove noise label (-1) for indices that require >=2 clusters
    mask = labels != -1
    if mask.sum() <= 1 or len(np.unique(labels[mask])) <= 1:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    sil = safe_silhouette(X[mask], labels[mask])
    db  = davies_bouldin_score(X[mask], labels[mask])
    ch  = calinski_harabasz_score(X[mask], labels[mask])
    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}

# ------------------------------
# Clustering methods
# ------------------------------
@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    params: Dict
    metrics: Dict[str, float]
    n_clusters: int
    noise_rate: Optional[float] = None

def run_hdbscan(X: np.ndarray,
                min_cluster_size_grid=(30, 50, 80),
                min_samples_grid=(None, 5, 10),
                random_state: int = 42) -> ClusterResult:
    if not HAS_HDBSCAN:
        # fallback to DBSCAN with a small epsilon sweep
        best = None
        for eps in (0.3, 0.5, 0.8, 1.0):
            db = DBSCAN(eps=eps, min_samples=10).fit(X)
            labels = db.labels_
            m = compute_internal_indices(X, labels)
            n_cl = len(set(labels)) - (1 if -1 in labels else 0)
            noise = float(np.mean(labels == -1)) if -1 in labels else 0.0
            score = (np.nan_to_num(m["silhouette"], nan=-1.0))
            if (best is None) or (score > best["score"]):
                best = {"labels": labels, "params": {"eps": eps, "min_samples": 10}, "metrics": m,
                        "n_clusters": n_cl, "noise_rate": noise, "score": score}
        return ClusterResult("DBSCAN-fallback", best["labels"], best["params"], best["metrics"],
                             best["n_clusters"], best["noise_rate"])

    best = None
    for mcs in min_cluster_size_grid:
        for ms in min_samples_grid:
            hdb = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, prediction_data=False)
            labels = hdb.fit_predict(X)
            m = compute_internal_indices(X, labels)
            n_cl = len(set(labels)) - (1 if -1 in labels else 0)
            noise = float(np.mean(labels == -1)) if -1 in labels else 0.0
            score = (np.nan_to_num(m["silhouette"], nan=-1.0))  # maximize silhouette
            if (best is None) or (score > best["score"]):
                best = {"labels": labels, "params": {"min_cluster_size": mcs, "min_samples": ms},
                        "metrics": m, "n_clusters": n_cl, "noise_rate": noise, "score": score}
    return ClusterResult("HDBSCAN", best["labels"], best["params"], best["metrics"],
                         best["n_clusters"], best["noise_rate"])

def run_gmm_bic(X: np.ndarray, k_range=range(2, 9), random_state: int = 42) -> ClusterResult:
    best = None
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=random_state)
        gmm.fit(X)
        labels = gmm.predict(X)
        m = compute_internal_indices(X, labels)
        bic = gmm.bic(X)
        score = -bic  # minimize BIC -> maximize -BIC
        if (best is None) or (score > best["score"]):
            best = {"labels": labels, "params": {"k": k, "covariance_type": "full"},
                    "metrics": m, "n_clusters": k, "score": score}
    return ClusterResult("GMM-BIC", best["labels"], best["params"], best["metrics"], best["n_clusters"])

def run_spectral_eigengap(X: np.ndarray, k_range=range(2, 9), random_state: int = 42) -> ClusterResult:
    # simple eigengap heuristic via silhouette selection
    best = None
    for k in k_range:
        sc = SpectralClustering(
            n_clusters=k, eigen_solver="arpack", affinity="nearest_neighbors",
            random_state=random_state, n_neighbors=10, assign_labels="kmeans"
        )
        labels = sc.fit_predict(X)
        m = compute_internal_indices(X, labels)
        score = np.nan_to_num(m["silhouette"], nan=-1.0)
        if (best is None) or (score > best["score"]):
            best = {"labels": labels, "params": {"k": k, "affinity": "nearest_neighbors"},
                    "metrics": m, "n_clusters": k, "score": score}
    return ClusterResult("Spectral", best["labels"], best["params"], best["metrics"], best["n_clusters"])

# ------------------------------
# Stability via bootstrapped ARI
# ------------------------------
def bootstrap_ari(X: np.ndarray, cluster_func, n_boot=20, sample_frac=0.8, random_state: int = 42) -> float:
    """
    cluster_func: function(X_sub) -> labels
    returns mean ARI across bootstraps vs clustering on full X
    """
    rng = np.random.RandomState(random_state)
    full_labels = cluster_func(X)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(np.arange(X.shape[0]), size=int(sample_frac * X.shape[0]), replace=False)
        labels_b = cluster_func(X[idx])
        # Compare only over the subset indices present
        aris.append(adjusted_rand_score(full_labels[idx], labels_b))
    return float(np.mean(aris))

# ------------------------------
# Proxy separability (pseudo-label classification)
# ------------------------------
def proxy_classification(X: np.ndarray, labels: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    # remove noise if present (label = -1) unless it forms a valid class
    mask = labels != -1
    Xp, yp = X[mask], labels[mask]
    # Guard: need >= 2 classes
    if len(np.unique(yp)) < 2:
        return pd.DataFrame(columns=["Model","Accuracy","Precision","Recall","F1"])
    X_train, X_test, y_train, y_test = train_test_split(
        Xp, yp, test_size=0.30, random_state=random_state, stratify=yp
    )
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM (RBF)":     SVC(kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN (k=5)":     KNeighborsClassifier(n_neighbors=5),
    }
    rows = []
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
        rows.append([name, acc, prec, rec, f1])
    df = pd.DataFrame(rows, columns=["Model","Accuracy","Precision","Recall","F1"])
    return (df * 100).round(2)

def confusion_for_model(X: np.ndarray, labels: np.ndarray, model_name="SVM (RBF)", random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    mask = labels != -1
    Xp, yp = X[mask], labels[mask]
    X_train, X_test, y_train, y_test = train_test_split(
        Xp, yp, test_size=0.30, random_state=random_state, stratify=yp
    )
    model_map = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "SVM (RBF)":     SVC(kernel="rbf", gamma="scale", probability=True, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "KNN (k=5)":     KNeighborsClassifier(n_neighbors=5),
    }
    clf = model_map[model_name]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    return cm, np.unique(np.concatenate([y_test, y_pred]))
