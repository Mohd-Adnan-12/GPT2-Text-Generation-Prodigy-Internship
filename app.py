from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Take input from user
prompt = input("Enter your prompt: ")

# Encode input text
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True
)

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Text:\n")
print(generated_text)
