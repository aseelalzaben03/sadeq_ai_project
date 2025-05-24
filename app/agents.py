# app/agents.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from googlesearch import search
from bs4 import BeautifulSoup
from langdetect import detect
import re
import string
import time
import requests

# --------- تنظيف النص ---------
class TextCleaner:
    def __init__(self):
        self.punctuation_table = str.maketrans('', '', string.punctuation + '«»…“”–')

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(self.punctuation_table)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# --------- كشف اللغة ---------
def detect_language(text):
    try:
        return 'ar' if detect(text) == 'ar' else 'en'
    except:
        return 'en'

# --------- ArabBERT Agent ---------
class ArabBERTAgent:
    def __init__(self):
       model_path = "Aseelalzaben03/sadaqai-bestmodel"# مسار النموذج المدرب
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       self.tokenizer = AutoTokenizer.from_pretrained(model_path)
       self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
       self.label_map = {0: "real", 1: "fake"}

    def analyze_text(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze()
            pred = torch.argmax(probs).item()
            confidence = probs[pred].item()
        return {
            "label": self.label_map[pred],
            "confidence": round(confidence, 4),
            "source": "ArabBERT"
        }

# --------- Web Search Agent ---------
class GoogleSearch:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def get_page_text(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                text = ' '.join(p.get_text(strip=True) for p in paragraphs)
                return text[:1500]
        except:
            pass
        return ""

    def search(self, query, lang='ar'):
        results = {}
        try:
            urls = search(query, lang=lang, num_results=5)
            for url in urls:
                time.sleep(2)
                content = self.get_page_text(url)
                if content:
                    results[url] = content
        except Exception as e:
            print(f"Search error: {e}")
        return results

class WebSearchAgent:
    def __init__(self):
        self.searcher = GoogleSearch()
        self.cleaner = TextCleaner()
        self.bert_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def extract_keywords(self, text, lang, top_k=5):
        cleaner = TextCleaner()
        cleaned_text = cleaner.clean_text(text)
        stopwords = ['في', 'من', 'على', 'و', 'عن', 'إلى', 'التي', 'الذي', 'أن', 'إن', 'كان', 'كما', 'لذلك', 'لكن',
                     'أو', 'ما', 'لا', 'لم', 'لن', 'قد', 'هذا', 'هذه', 'هو', 'هي', 'هم', 'ثم', 'كل', 'هناك', 'بعد'] \
            if lang == 'ar' else \
            ['the', 'and', 'is', 'in', 'to', 'of', 'that', 'a', 'on', 'for', 'with', 'as', 'are', 'it', 'was', 'by']

        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words=stopwords).fit([cleaned_text])
        candidates = vectorizer.get_feature_names_out()

        doc_embedding = self.bert_model.encode([cleaned_text], convert_to_tensor=True)
        candidate_embeddings = self.bert_model.encode(candidates, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(doc_embedding, candidate_embeddings)[0]
        top_k_indices = similarities.topk(k=min(top_k, len(candidates))).indices

        return " ".join([candidates[idx] for idx in top_k_indices])

    def search_text(self, text: str) -> dict:
        lang = detect_language(text)
        keywords = self.extract_keywords(text, lang)
        sources = self.searcher.search(keywords, lang=lang)
        return self.analyze_sources(text, sources)

    def search_url(self, url: str) -> dict:
        content = self.searcher.get_page_text(url)
        if not content or len(content) < 100:
            return {"label": "unknown", "confidence": 0.0, "source": "WebSearch"}
        return self.search_text(content)

    def analyze_sources(self, input_text, sources):
        input_clean = self.cleaner.clean_text(input_text)
        input_emb = self.bert_model.encode(input_clean, convert_to_tensor=True)

        source_similarities = {}
        for url, content in sources.items():
            paragraphs = content.split('.')
            similarities = []
            for para in paragraphs:
                para_clean = self.cleaner.clean_text(para)
                if len(para_clean) > 30:
                    para_emb = self.bert_model.encode(para_clean, convert_to_tensor=True)
                    sim = util.pytorch_cos_sim(input_emb, para_emb).item()
                    similarities.append(sim)
            if similarities:
                source_similarities[url] = max(similarities)

        if not source_similarities:
            return {"label": "unknown", "confidence": 0.0, "source": "WebSearch"}

        best_url = max(source_similarities, key=source_similarities.get)
        best_score = source_similarities[best_url]

        label = "real" if best_score > 0.75 else "fake" if best_score < 0.5 else "uncertain"
        return {
            "label": label,
            "confidence": round(best_score, 4),
            "source": "WebSearch",
            "url": best_url
        }

# --------- دمج النتائج ---------
def combine_results(result1: dict, result2: dict) -> dict:
    best = result1 if result1["confidence"] >= result2["confidence"] else result2
    return {
        "final_label": best["label"],
        "final_confidence": best["confidence"],
        "details": [result1, result2]
    }

