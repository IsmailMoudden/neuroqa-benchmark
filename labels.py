METRIC_INFO = {
    "recall_at_5": {
        "label": "Recall@5",
        "help": "Out of all relevant document passages, how many were found in the top 5 results? "
                "A score of 1.0 means all relevant passages were retrieved.",
    },
    "mrr": {
        "label": "MRR",
        "help": "Mean Reciprocal Rank — how high does the first relevant passage rank? "
                "1.0 = always first, 0.5 = always second. Higher is better.",
    },
    "f1": {
        "label": "F1 Score",
        "help": "Word overlap between the generated answer and the expected answer. "
                "Only meaningful when an expected answer is provided.",
    },
    "faithfulness": {
        "label": "Faithfulness",
        "help": "Does the answer stay within what the retrieved passages say? "
                "A low score means the model is adding information not in the source.",
    },
    "relevance": {
        "label": "Relevance",
        "help": "Is the answer actually addressing the question asked? "
                "Scored by an AI judge from 0 (off-topic) to 1 (perfectly on-topic).",
    },
    "composite_score": {
        "label": "Composite Score",
        "help": "Weighted combination of all metrics above. Use this to quickly compare strategies.",
    },
}

DIFFICULTY_BADGE = {"easy": "Low", "medium": "Medium", "hard": "High"}
DIFFICULTY_COLOR = {"easy": "green", "medium": "orange", "hard": "red"}
TYPE_LABELS = {
    "factual":    "Factual",
    "definition": "Definition",
    "procedural": "Procedural",
    "multi_hop":  "Multi-hop",
    "causal":     "Causal",
}
