from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread
import sys
import gradio as gr

# Model name
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
CYAN = '\033[36m'
RESET = '\033[0m'  # Resets to default color

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, skip_special_tokens=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 for faster and less memory-intensive computation if supported
).to("cuda")

def generate(prompt, history):

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True})
    # Generate output
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=500,
        eos_token_id = tokenizer.eos_token_id,
        pad_token_id = tokenizer.pad_token_id,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.5,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    generated_text = ""
    for new_text in streamer:
        if new_text == "< | end_of_sentence | >":
            break
        elif new_text == tokenizer.eos_token_id:
            break
        generated_text += new_text
        yield generated_text


gr.ChatInterface(
    fn=generate,
).queue().launch()
