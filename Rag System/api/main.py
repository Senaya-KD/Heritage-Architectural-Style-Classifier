"""
main.py
=======
A small web server that exposes our existing RAG assistant and CV
classifier as HTTP endpoints, so a browser-based frontend can use them.

This file does NOT contain any new logic - it only calls the functions
we already built and tested in src/, and returns their results as JSON
instead of printing them to a terminal.
"""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.classifier import load_classifier
from src.rag_chain import load_vector_store, HeritageAssistant
from src.cv_rag_bridge import explain_uploaded_image


app = FastAPI(title="NHPT Heritage Assistant API")

# CORS lets a webpage make requests to this server. Without this, the
# browser blocks the request for security reasons, even on your own
# machine, even when testing locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load everything ONCE, when the server starts - not on every request.
# Loading the model and knowledge base takes several seconds; doing it
# per-request would make every single click painfully slow.
print("Loading CV model...")
model = load_classifier()

print("Loading knowledge base...")
vector_store = load_vector_store()

# ONE assistant per running server, shared across all requests, so
# the conversation history persists between messages sent from the
# browser.
assistant = HeritageAssistant(vector_store)

print("API ready.")


def chunks_to_json(chunks):
    """
    Convert retrieved chunks into a simple list of dictionaries,
    since Chroma's Document objects cannot be sent directly as JSON.
    """

    sources = []
    for chunk in chunks:
        sources.append({
            "style": chunk.metadata.get("style", "N/A"),
            "doc_type": chunk.metadata.get("doc_type", "reference"),
            "filename": chunk.metadata.get("filename", "unknown")
        })
    return sources


@app.post("/ask")
def ask(question: str):
    """
    Text chat endpoint. The frontend sends a question, we return the
    assistant's answer and its sources.
    """

    answer, chunks = assistant.ask(question)

    return {
        "answer": answer,
        "sources": chunks_to_json(chunks)
    }


@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Image upload endpoint. The frontend sends a photo, we:
      1. Save it temporarily to disk (the classifier needs a file path)
      2. Run it through the CV model and the RAG assistant
      3. Return the prediction and explanation as JSON
    """

    temp_path = Path("temp_upload.jpg")

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = explain_uploaded_image(str(temp_path), assistant, model)

    temp_path.unlink()   # delete the temporary file once we are done with it

    return {
        "style": result["prediction"]["style"],
        "confidence": result["prediction"]["confidence"],
        "top_3": result["prediction"]["top_3"],
        "question": result["question"],
        "answer": result["answer"],
        "sources": chunks_to_json(result["chunks"])
    }