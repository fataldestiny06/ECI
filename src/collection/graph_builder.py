import os
import pandas as pd
import networkx as nx


def build_graph(csv_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    graph_file = os.path.join(output_dir, "recommendation_graph.gexf")

    df = pd.read_csv(csv_file).sort_values("step").reset_index(drop=True)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_node(row["url"], title=row["title"], step=int(row["step"]))

    for i in range(len(df) - 1):
        G.add_edge(df.iloc[i]["url"], df.iloc[i + 1]["url"])

    nx.write_gexf(G, graph_file)

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "graph_file": graph_file
    }
