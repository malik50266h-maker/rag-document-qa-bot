import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    "RAG retrieves relevant docs before answering.",
    "OpenAI Embeddings convert text into vectors for semantic search."
]

def ask_question(question):
    context = "\n".join(documents)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(ask_question("What is RAG?"))
