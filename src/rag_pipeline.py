import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # just the src folder
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")
if not os.path.exists(FAISS_PATH):
    raise FileNotFoundError(f"FAISS index not found at {FAISS_PATH}")

# Load embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Load reranking model
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load vector DB
vectorstore = FAISS.load_local(
    FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq client
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is missing")
client = Groq(api_key=api_key)

#Reranking function
def rerank_docs(query, docs, top_k=3):
    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked_docs[:top_k]]

#Query re-writting 
def rewrite_query(query, chat_history):
    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]]
    )

    prompt = f"""
    Convert the following conversation into a standalone question.

    Chat History:
    {history_text}

    Follow-up Question:
    {query}

    Rewritten standalone question:
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


#Generate response after re-writting and reranking 
def generate_answer(query, chat_history):

    # Step 1: Retrieve context
    rewritten_query = rewrite_query(query, chat_history)
    docs = retriever.invoke(rewritten_query)
    
    docs = rerank_docs(rewritten_query, docs)

    context = "\n\n".join([doc.page_content for doc in docs])

    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )

    # Step 2: Create prompt
    prompt = f"""
    You are a helpful assistant. Think and use your trained knowledge to answer the question asked in the context."

    Chat History:
    {history_text}
    
    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Step 3: Call Groq LLaMA model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Extract answer
    return response.choices[0].message.content


# ########## Evaluation ############
# eval_data = [
#     {
#         "question": "What is fallout in the system?",
#         "ground_truth": "Fallout refers to orders that fail during processing due to errors or mismatches."
#     },
#     {
#         "question": "What model was used?",
#         "ground_truth": "An XGBoost model was used for prediction."
#     },
#     {
#         "question": "How was model accuracy measured?",
#         "ground_truth": "Using recall and precision"
#     },
#     {
#         "question": "How is missing data handled?",
#         "ground_truth": "Missing numerical columns are filled with 0 and categorical columns were replaced with no_data, filling with the most frequent category and Context-based imputation depending on feature distribution"
#     },
#     {
#         "question": "How does the model improve customer experience?",
#         "ground_truth": "By predicting fallouts beforehand the model allows to improvise order journey for the customer and hence improves customer experience."
#     }
# ]

# results = []
# for item in eval_data:
#     question = item["question"]
#     gt = item["ground_truth"]

#     answer = generate_answer(question, [])

#     results.append({
#         "question": question,
#         "ground_truth": gt,
#         "model_answer": answer
#     })

# def evaluate_with_llm(question, gt, answer):
#     prompt = f"""
#     Question: {question}

#     Ground Truth: {gt}

#     Model Answer: {answer}

#     Is the model answer correct and relevant?
#     Answer only: Correct or Incorrect
#     """

#     response = client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[{"role": "user", "content": prompt}]
#     )

#     return response.choices[0].message.content.strip()

# correct = 0
# for item in results:
#     verdict = evaluate_with_llm(
#         item["question"],
#         item["ground_truth"],
#         item["model_answer"]
#     )

#     print(f"\nQ: {item['question']}")
#     print(f"Answer: {item['model_answer']}")
#     print(f"Verdict: {verdict}")

#     if "Correct" in verdict:
#         correct += 1

# accuracy = correct / len(results)
# print(f"\nFinal Accuracy: {accuracy}")
