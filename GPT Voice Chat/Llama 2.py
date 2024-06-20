# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes 
import torch

# Define variable to hold llama2 weights naming 
name = "meta-llama/Llama-2-70b-chat-hf"
# Set auth token variable from hugging face 
auth_token = "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"

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(name, 
    cache_dir='./model/', use_auth_token=auth_token)


# Create model
model = AutoModelForCausalLM.from_pretrained(name, 
    cache_dir='./model/', use_auth_token=auth_token, torch_dtype=torch.float16, 
    rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True)

# Setup a prompt 
prompt = "### User:What is the fastest car in  \
          the world and how much does it cost? \
          ### Assistant:"
# Pass the prompt to the tokenizer
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# Setup the text streamer 
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


# Actually run the thing
output = model.generate(**inputs, streamer=streamer, 
                        use_cache=True, max_new_tokens=float('inf'))


# Covert the output tokens back to text 
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
