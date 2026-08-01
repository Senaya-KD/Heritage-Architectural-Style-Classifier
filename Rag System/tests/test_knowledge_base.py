import pytest
from src.knowledge_base import load_all_documents, split_into_chunks

@pytest.fixture(scope="module")
def documents():
    return load_all_documents()


@pytest.fixture(scope="module")
def chunks(documents):
    return split_into_chunks(documents)


def test_loads_exactly_twenty_documents(documents):
    assert len(documents) == 20


def test_every_document_has_text(documents):
    for doc in documents:
        assert len(doc["text"].strip()) > 0


def test_style_documents_have_a_style_in_metadata(documents):
    style_docs = [d for d in documents if "history" in d["filename"] or "features" in d["filename"]]

    assert len(style_docs) == 16

    for doc in style_docs:
        assert "style" in doc["metadata"]
        assert doc["metadata"]["style"] != ""


def test_general_documents_exist(documents):
    filenames = [d["filename"] for d in documents]

    assert "glossary.md" in filenames
    assert "conservation.md" in filenames
    assert "visiting.md" in filenames
    assert "about_nhpt.md" in filenames


def test_chunking_produces_chunks(chunks):
    assert len(chunks) > 20


def test_every_chunk_has_filename(chunks):
    for chunk in chunks:
        assert "filename" in chunk
        assert chunk["filename"] != ""


def test_no_chunk_is_empty(chunks):
    for chunk in chunks:
        assert len(chunk["text"].strip()) > 0


def test_chunks_stay_within_a_reasonable_size(chunks):
    for chunk in chunks:
        assert len(chunk["text"]) < 1000