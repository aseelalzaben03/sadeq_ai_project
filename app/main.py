# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.agents import ArabBERTAgent, WebSearchAgent, combine_results

app = FastAPI()

# تحميل العوامل مرة واحدة فقط (لتقليل استهلاك الموارد)
arabbert_agent = ArabBERTAgent()
websearch_agent = WebSearchAgent()

class InputData(BaseModel):
    input_text: Optional[str] = None
    input_url: Optional[str] = None

@app.post("/analyze")
async def analyze(input_data: InputData):
    if input_data.input_text:
        arabbert_result = arabbert_agent.analyze_text(input_data.input_text)
        websearch_result = websearch_agent.search_text(input_data.input_text)
        combined = combine_results(arabbert_result, websearch_result)
        return {"type": "text", "result": combined}
    elif input_data.input_url:
        websearch_result = websearch_agent.search_url(input_data.input_url)
        return {"type": "url", "result": websearch_result}
    else:
        raise HTTPException(status_code=400, detail="Please provide either input_text or input_url.")

