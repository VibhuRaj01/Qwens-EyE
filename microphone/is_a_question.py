import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO)

pipe = pipeline(
    "text-classification",
    model="shahrukhx01/bert-mini-finetune-question-detection",
    device="cuda",
)


def is_a_question(query: str, pipe=pipe) -> bool:
    try:
        result = pipe(query)
        label = result[0]["label"]
        return label == "LABEL_1"
    except Exception as e:
        logging.error(f"An error occurred while classifying the text: {e}")
        return False
