import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class EchoChamberAnalyzer:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading SBERT model...")
        self.model = SentenceTransformer(model_name)
        print("SBERT ready\n")

        self.df = None
        self.embeddings = None

    # LOAD DATA
    def load_data(self, csv_file):
        self.df = pd.read_csv(csv_file)

        if "title" not in self.df.columns:
            raise ValueError("CSV must contain a 'title' column")

        print(f"Loaded {len(self.df)} videos\n")
        return self.df

    def create_embeddings(self):
        print("Creating SBERT embeddings...")

        self.embeddings = self.model.encode(
            self.df["title"].tolist(),
            show_progress_bar=True
        )

        print(f"Embedding shape: {self.embeddings.shape}\n")
        return self.embeddings

    def cluster_videos(self, n_clusters=5):
        print("Clustering videos...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        self.df["cluster"] = kmeans.fit_predict(self.embeddings)

        print(
            self.df["cluster"]
            .value_counts()
            .sort_index(),
            "\n"
        )

        return self.df["cluster"]

    def visualize_clusters(self, save_path):
        print("Generating 2D cluster visualization...")

        pca = PCA(n_components=2, random_state=42)
        points_2d = pca.fit_transform(self.embeddings)

        plt.figure(figsize=(10, 8))

        for cluster_id in sorted(self.df["cluster"].unique()):
            mask = self.df["cluster"] == cluster_id
            plt.scatter(
                points_2d[mask, 0],
                points_2d[mask, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.7
            )

        plt.title("Semantic Geometry of YouTube Recommendations")
        plt.xlabel("PCA Dimension 1")
        plt.ylabel("PCA Dimension 2")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved plot to {save_path}\n")


    def generate_sbert_dataset(self, save_path="output/sbert_dataset.csv"):
        print("Generating SBERT-derived dataset...")

        if self.embeddings is None or "cluster" not in self.df.columns:
            raise RuntimeError("Run embeddings and clustering first")

        # PCA geometry
        pca = PCA(n_components=2, random_state=42)
        points_2d = pca.fit_transform(self.embeddings)

        # Similarity to previous recommendation
        sim_to_prev = [0.0]
        for i in range(1, len(self.embeddings)):
            sim = cosine_similarity(
                self.embeddings[i].reshape(1, -1),
                self.embeddings[i - 1].reshape(1, -1)
            )[0][0]
            sim_to_prev.append(sim)

        # Similarity to global centroid
        global_centroid = self.embeddings.mean(axis=0)
        sim_to_centroid = cosine_similarity(
            self.embeddings,
            global_centroid.reshape(1, -1)
        ).flatten()

        # Cluster sizes
        cluster_sizes = self.df["cluster"].map(
            self.df["cluster"].value_counts()
        )

        # Build dataset
        dataset = self.df.copy()
        dataset["pca_x"] = points_2d[:, 0]
        dataset["pca_y"] = points_2d[:, 1]
        dataset["sim_to_prev"] = sim_to_prev
        dataset["sim_to_centroid"] = sim_to_centroid
        dataset["cluster_size"] = cluster_sizes

        dataset.to_csv(save_path, index=False)

        print(f"SBERT dataset saved to {save_path}\n")
        return dataset

    # USER ECHO SCORE
    def calculate_user_echo_score(self, user_indices):
        user_df = self.df.iloc[user_indices]
        user_embeddings = self.embeddings[user_indices]

        # A. Cluster dominance (entropy)
        probs = user_df["cluster"].value_counts(normalize=True)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(self.df["cluster"].nunique())
        dominance = 1 - entropy / max_entropy

        # B. intra-user similarity
        sim = cosine_similarity(user_embeddings)
        upper = sim[np.triu_indices_from(sim, k=1)]
        similarity = upper.mean() if len(upper) > 0 else 0

        # C. global isolation
        user_centroid = user_embeddings.mean(axis=0)
        global_centroid = self.embeddings.mean(axis=0)
        isolation = 1 - cosine_similarity(
            user_centroid.reshape(1, -1),
            global_centroid.reshape(1, -1)
        )[0][0]

        echo_score = 100 * (
            0.4 * dominance +
            0.4 * similarity +
            0.2 * isolation
        )

        return {
            "echo_score": round(float(echo_score), 2),
            "cluster_dominance": round(float(dominance), 3),
            "intra_similarity": round(float(similarity), 3),
            "global_isolation": round(float(isolation), 3)
        }
