from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from model_chatbot import tokenizer, model
from context import context

# Load the model and tokenizer


# Evaluation metrics
def evaluate(context, question, expected_answer):
    inputs = tokenizer.encode_plus(
        question, context, return_tensors="pt", truncation=True, max_length=4096
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the answer span
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1
    predicted_answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    )

    # Calculate Exact Match (EM) and F1 Score
    em = int(predicted_answer == expected_answer)
    overlap = set(predicted_answer.split()) & set(expected_answer.split())
    precision = (
        len(overlap) / len(predicted_answer.split()) if predicted_answer.split() else 0
    )
    recall = (
        len(overlap) / len(expected_answer.split()) if expected_answer.split() else 0
    )
    f1 = (
        (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    )

    return {
        "Question": question,
        "Predicted Answer": predicted_answer,
        "Expected Answer": expected_answer,
        "Exact Match": em,
        "F1 Score": f1,
    }


# Test data
questions = [
    ("What NLP stands for?", "Natural Language Processing"),
    ("what is AI?", "Artificial Intelligence"),
]

# Run evaluation
results = []
for question, expected_answer in questions:
    result = evaluate(context, question, expected_answer)
    results.append(result)

# Display results
for r in results:
    print(r)
