from flask import Flask, request, jsonify
from pipeline import run_pipeline

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    seed_video = data.get("seed_video")

    if not seed_video:
        return jsonify({"error": "No seed video provided"}), 400

    results = run_pipeline(seed_video)
    return jsonify(results)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
