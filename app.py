from flask import Flask, request, jsonify # type: ignore
from flask import render_template # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np # type: ignore
import faiss # type: ignore
import json
import requests # type: ignore
from flask_cors import CORS # type: ignore


app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

# Load knowledge base
with open("store_data_updated.json", "r") as file:
    knowledge_base = json.load(file)

# Preprocess
documents = []
for store_id, store_data in knowledge_base.items():
    if store_id == "regions":
        continue
    documents.append({
        "store_id": store_id,
        "content": f"Summary for {store_id}: Total stock: {store_data['total_stock']}, "
                   f"Avg product rate: {store_data['avg_product_rate']}, Total rate: {store_data['total_rate']}"
    })
    for category, cat_data in store_data["categories"].items():
        documents.append({
            "store_id": store_id,
            "category": category,
            "content": f"Category {category} in {store_id}: Total stock: {cat_data['total_stock']}, "
                       f"Avg rate: {cat_data['avg_product_rate']}, Total rate: {cat_data['total_rate']}. "
                       f"Comment: {cat_data.get('comment', '')} Suggestion: {cat_data.get('suggestion', '')}"
        })

# Embedding & FAISS index
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_texts = [d["content"] for d in documents]
doc_embeddings = model.encode(doc_texts, convert_to_numpy=True)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(doc_embeddings)

# Retrieve top-k documents
def retrieve_documents(query, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [documents[i] for i in indices[0]]

# Send query to local LLM (Ollama)
def ask_ollama(query, context, model_name="llama3"):
    prompt = f"""You are a helpful assistant. Based on the following context, answer the question clearly:\n\n{context}\n\nQuestion: {query}"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model_name, "prompt": prompt, "stream": False}
    )
    data = response.json()
    if "response" not in data:
        print("⚠️ Unexpected Ollama response:", data)
        return "⚠️ The assistant failed to respond. Please try again."
    return data["response"]


# Flask endpoint
@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    retrieved_docs = retrieve_documents(user_query)
    context = " ".join([doc["content"] for doc in retrieved_docs])
    response = ask_ollama(user_query, context)

    return jsonify({
        "query": user_query,
        "response": response,
        "documents_used": [doc["content"] for doc in retrieved_docs]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5050)
