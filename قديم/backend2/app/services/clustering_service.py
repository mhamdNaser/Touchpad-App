# app/services/clustering_service.py
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
import joblib

class ClusteringService:
    def __init__(self, n_clusters: int = 28, random_state: int = 42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.mapping = {}  # cluster_id -> majority label

    def fit(self, latent_vectors: np.ndarray, labels: list):
        """latent_vectors: (N, D), labels: list of original labels (same length)
        بعد التدريب نولّد mapping من كل cluster للحرف الأكثر تكراراً
        """
        self.kmeans.fit(latent_vectors)
        cluster_ids = self.kmeans.predict(latent_vectors)
        cluster_to_labels = defaultdict(list)
        for cid, lab in zip(cluster_ids, labels):
            cluster_to_labels[cid].append(lab)
        mapping = {}
        for cid, labs in cluster_to_labels.items():
            most = Counter(labs).most_common(1)[0][0]
            mapping[cid] = most
        self.mapping = mapping
        return mapping

    def predict_label(self, latent_vector: np.ndarray):
        cid = int(self.kmeans.predict(latent_vector.reshape(1, -1))[0])
        return self.mapping.get(cid, None), cid

    def save(self, path_prefix: str):
        joblib.dump(self.kmeans, path_prefix + "_kmeans.joblib")
        joblib.dump(self.mapping, path_prefix + "_mapping.joblib")

    def load(self, path_prefix: str):
        self.kmeans = joblib.load(path_prefix + "_kmeans.joblib")
        self.mapping = joblib.load(path_prefix + "_mapping.joblib")
