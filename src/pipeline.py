import os
from pathlib import Path
from analyzer import EchoChamberAnalyzer
from collection.scraper import run_scraper
from collection.graph_builder import build_graph
from Validator.validator import validate_clusters


def run_pipeline(seed_video):
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Phase 1 — Scrape
    csv_path = run_scraper(seed_video)

    # Phase 2 — SBERT
    analyzer = EchoChamberAnalyzer()
    analyzer.load_data(csv_path)
    analyzer.create_embeddings()
    analyzer.cluster_videos(n_clusters=5)
    analyzer.visualize_clusters(OUTPUT_DIR / "clusters_2d.png")
    analyzer.generate_sbert_dataset(OUTPUT_DIR / "sbert_dataset.csv")

    # Phase 3 — Graph
    graph_info = build_graph(csv_path, OUTPUT_DIR)

    # Phase 4 — Validation
    validated = validate_clusters(
        OUTPUT_DIR / "sbert_dataset.csv",
        OUTPUT_DIR / "validated_clusters.csv"
    )

    # Phase 5 — Echo score
    user_indices = list(range(len(analyzer.df)))
    echo_score = analyzer.calculate_user_echo_score(user_indices)

    return {
        "echo_score": echo_score,
        "graph": graph_info,
        "clusters": validated.to_dict(orient="records"),
        "logs": analyzer.df[["step", "title", "url"]].to_dict(orient="records")
    }
