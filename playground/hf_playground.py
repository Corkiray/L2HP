# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon3-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-7B-Instruct", device_map="auto", offload_state_dict=True)
llm = pipeline('text-generation', model=model, tokenizer=tokenizer)