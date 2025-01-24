from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Model ID
model_id = "mistralai/Mixtral-8x7B-v0.1"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Input text
text = "Hello my name is"

# Tokenize the input and move to the same device as the model
inputs = tokenizer(text, return_tensors="pt").to(device)

# Generate text
outputs = model.generate(**inputs, max_new_tokens=20)

# Decode and print the generated text
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
