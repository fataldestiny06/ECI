import subprocess
import sys
from pathlib import Path

# Get project root
BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR / "SBERT-Application-main"

print("▶ Phase 1: Running scraper...")
subprocess.run(
    [sys.executable, PROJECT_DIR / "collection" / "scraper.py"],
    check=True
)

print("▶ Phase 2: Running SBERT clustering...")
subprocess.run(
    [sys.executable, PROJECT_DIR / "main.py"],
    check=True
)

print("▶ Phase 3: Running semantic validator...")
subprocess.run(
    [sys.executable, PROJECT_DIR / "Validator" / "validator.py"],
    check=True
)

print("✅ Pipeline completed successfully")
print("📁 Output file: SBERT-Application-main/output/validated_clusters.csv")