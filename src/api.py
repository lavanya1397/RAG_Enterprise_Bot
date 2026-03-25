from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_pipeline import generate_answer

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    chat_history: list = []

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        answer = generate_answer(request.query, request.chat_history)
        if not answer:
            answer = "generate_answer returned empty"
        return {
            "query": request.query,
            "answer": answer,
            "error": None
        }
    except Exception as e:
        return {
            "query": request.query,
            "answer": None,
            "error": str(e)
        }
