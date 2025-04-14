from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# create model directory
os.mkdir("bigbird_model")

memory = []
mem_size = 100
model_path = (
    "./bigbird_model"  # folder that include the model, config.json, and tokenizer
)

# check if model already save or not 🙌
model = ""  # initialize the model variable
tokenizer = ""  # init the tokenizer variable


if os.path.exists(model_path):
    model_file = os.path.join(model_path, "model.safetensors")
    config_file = os.path.join(model_path, "config.json")
    tokenizer_file = os.path.join(model_path, "tokenizer_config.json")

    if os.path.exists(model_file) and os.path.exists(config_file):
        print(f"The model is avaliable")
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    else:
        print(f"Model dir found, but no model")
        tokenizer = AutoTokenizer.from_pretrained("google/bigbird-base-trivia-itc")
        model = AutoModelForQuestionAnswering.from_pretrained(
            "google/bigbird-base-trivia-itc"
        )
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
else:
    print("Error, no directory of the model found")
    exit


# save model


def resolve_pronouns(question):
    pronoun_list = [
        "it",
        "they",
        "that",
        "this",
        "those",
        "he",
        "she",
        "her",
        "him",
        "that model",
    ]  # list of avaliable pronouns
    for x in pronoun_list:
        if x in question.lower():
            for prev_q, prev_ans in reversed(memory):
                question = question.replace(x, prev_ans, 1)
                break
    return question


def answer_question(context, question):
    """
    Answer the question based on the provided context using the pre-trained model.
    """
    global memory  # Use global memory to store interactions

    # Resolve pronouns in the question
    question = resolve_pronouns(question)

    # Tokenize the input (question and context)
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    # Get the model's outputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the most likely start and end positions of the answer
    start_index = torch.argmax(outputs.start_logits)
    end_index = torch.argmax(outputs.end_logits) + 1

    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index])
    )

    # Handle cases where no answer is found
    if answer.strip() == "":
        answer = "I'm not sure about that. Can you ask in a different way?"

    # Update the memory
    memory.append((question, answer))
    if len(memory) > mem_size:
        memory.pop(0)  # Ensure memory size doesn't exceed the limit

    return answer


def chatbot_loop(context):
    from welcome_banner import welcome_banner

    user_name = ""
    print(welcome_banner)
    print("🤖Hello, please input your name so I can know who you are?🤖")
    user_name = input("Username👤: ").strip()
    print("🤖Chatbot initialized! Ask your questions or type 'exit' to leave.🤖")

    while True:
        user_input = input(f"{user_name}👤: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = answer_question(context, user_input)
        print("Bot🤖:", response)


# Set the data context
from context import context

# Run the chatbot
chatbot_loop(context)
