
import os
from src.config import (
    KB_STYLES_DIR,
    KB_GENERAL_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    CHROMA_DB_DIR
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def read_frontmatter(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    metadata = {}
    body_lines = []

    inside_frontmatter = False
    frontmatter_ended = False

    for line in lines:

        stripped_line = line.strip()

        if stripped_line == "---" and inside_frontmatter == False and frontmatter_ended == False:
            inside_frontmatter = True
            continue

        if stripped_line == "---" and inside_frontmatter == True:
            inside_frontmatter = False
            frontmatter_ended = True
            continue

        if inside_frontmatter == True:
            if ":" in stripped_line:
                key, value = stripped_line.split(":", 1)
                metadata[key.strip()] = value.strip()

        if frontmatter_ended == True:
            body_lines.append(line)

    body_text = "".join(body_lines)

    return metadata, body_text


def load_all_documents():

    all_documents = []

    folders_to_read = [KB_STYLES_DIR, KB_GENERAL_DIR]

    for folder in folders_to_read:

        file_names = os.listdir(folder)
        file_names.sort()

        for file_name in file_names:

            if file_name.endswith(".md"):

                file_path = folder / file_name

                metadata, body_text = read_frontmatter(file_path)

                document = {
                    "text": body_text,
                    "metadata": metadata,
                    "filename": file_name
                }

                all_documents.append(document)

    return all_documents


def split_into_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []

    for doc in documents:

        text_pieces = splitter.split_text(doc["text"])

        for piece in text_pieces:

            chunk = {
                "text": piece,
                "metadata": doc["metadata"],
                "filename": doc["filename"]
            }

            all_chunks.append(chunk)

    return all_chunks


def build_vector_store(chunks):
    langchain_documents = []

    for chunk in chunks:

        # Combine frontmatter metadata + filename into one dictionary
        full_metadata = dict(chunk["metadata"])
        full_metadata["filename"] = chunk["filename"]

        doc = Document(
            page_content=chunk["text"],
            metadata=full_metadata
        )
        langchain_documents.append(doc)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma.from_documents(
        documents=langchain_documents,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DB_DIR)
    )

    return vector_store


# Quick test - only runs if this file is executed directly
if __name__ == "__main__":

    documents = load_all_documents()
    print("Total documents loaded:", len(documents))

    chunks = split_into_chunks(documents)
    print("Total chunks created:", len(chunks))

    print("\nBuilding vector store (downloads the embedding model on first run, may take a minute)...")
    vector_store = build_vector_store(chunks)
    print("Vector store built and saved to disk.")

    print("\nTest search: 'pointed arch'")
    results = vector_store.similarity_search("pointed arch", k=3)

    for r in results:
        style = r.metadata.get("style", "N/A")
        filename = r.metadata.get("filename", "unknown")
        print("-", style, "|", filename, "|", r.page_content[:80])