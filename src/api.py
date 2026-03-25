from fastapi import FastAPI
from pydantic import BaseModel

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
       from src.rag_pipeline import generate_answer
       answer = generate_answer(request.query, request.chat_history)
       return {
           "query": request.query,
           "answer": answer
        }
     except Exception as e:
        return {"error": str(e)}
