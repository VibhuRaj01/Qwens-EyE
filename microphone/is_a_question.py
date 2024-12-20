# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="shahrukhx01/bert-mini-finetune-question-detection",
    device="cuda",
)


def is_a_question(query, pipe=pipe):
    result = pipe(query)
    label = result[0]["label"]

    return label == "LABEL_1"
