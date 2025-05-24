# app/initial.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Aseelalzaben03/sadaqai-bestmodel"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
