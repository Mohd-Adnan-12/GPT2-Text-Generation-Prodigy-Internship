from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, max_length=100):
    model_name = "gpt2"
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    result = generate_text(user_prompt)
    print("\nGenerated Text:\n")
    print(result)
