# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="shahrukhx01/bert-mini-finetune-question-detection",
    device="cuda",
)


def is_a_question(query, pipe):
    result = pipe(query)

    return result
