from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def validate_clusters(input_file, output_file,
                      consensus_threshold=0.6,
                      subjectivity_threshold=0.3):

    df = pd.read_csv(input_file)
    analyzer = SentimentIntensityAnalyzer()
    results = []

    for cluster_id, group in df.groupby("cluster"):
        sentiments = []

        for text in group["title"].dropna():
            sentiments.append(analyzer.polarity_scores(str(text))["compound"])

        if not sentiments:
            consensus = subjectivity = 0
        else:
            subjectivity = sum(abs(s) for s in sentiments) / len(sentiments)
            dominant = 1 if sum(sentiments) >= 0 else -1
            consensus = sum((s * dominant) > 0.3 for s in sentiments) / len(sentiments)

        label = (
            "Radicalization Node"
            if consensus >= consensus_threshold and subjectivity >= subjectivity_threshold
            else "Benign Cluster"
        )

        results.append({
            "cluster_id": cluster_id,
            "consensus": round(consensus, 2),
            "subjectivity": round(subjectivity, 2),
            "label": label
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    return out_df
