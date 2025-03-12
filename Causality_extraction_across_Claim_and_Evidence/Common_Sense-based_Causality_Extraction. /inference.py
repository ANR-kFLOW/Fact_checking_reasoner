import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "/data/Youss/Fact_cheking/reasoner/inference_model_based_on_sub_obj/best_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


label_mapping = {0: 'cause', 1: 'intend', 2: 'prevent', 3: 'enable', 4: 'no_relation'}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}


def predict_relation(event1_text, event2_text):

    inputs = tokenizer(
        text=event1_text,
        text_pair=event2_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_label = label_mapping[predicted_class]
    return predicted_label


# examples = [
#     {"event1": "The factory released toxic waste into the river.",
#      "event2": "The river became polluted.",
#      "expected": "cause"},
#
#     {"event1": "The government launched a public awareness campaign.",
#      "event2": "Citizens started recycling more.",
#      "expected": "intend"},
#
#     {"event1": "The firefighters built a firebreak.",
#      "event2": "The wildfire did not reach the town.",
#      "expected": "prevent"},
#
#     {"event1": "The user signed up for a premium account.",
#      "event2": "They accessed exclusive content on the website.",
#      "expected": "enable"},
#
#     {"event1": "The cat climbed a tree.",
#      "event2": "It rained heavily in the evening.",
#      "expected": "no_relation"}
# ]
#
# print("Testing Examples:\n")
# for example in examples:
#     event1 = example["event1"]
#     event2 = example["event2"]
#     expected = example["expected"]
#
#     predicted = predict_relation(event1, event2)
#     print(f"Event 1: {event1}")
#     print(f"Event 2: {event2}")
#     print(f"Expected Relation: {expected}")
#     print(f"Predicted Relation: {predicted}")
#     print("-" * 50)

