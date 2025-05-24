# app/initial.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# اسم الريبو على HuggingFace
model_name = "Aseelalzaben03/sadaqai-bestmodel"

# تحميل الـtokenizer والموديل من HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)

