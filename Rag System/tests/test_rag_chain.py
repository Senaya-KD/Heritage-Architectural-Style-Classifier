import pytest
from src.rag_chain import load_vector_store, retrieve_chunks, HeritageAssistant


@pytest.fixture(scope="module")
def vector_store():
    return load_vector_store()


@pytest.fixture
def assistant(vector_store):
 
    return HeritageAssistant(vector_store)


def test_retrieval_returns_chunks(vector_store):
    results = retrieve_chunks(vector_store, "What is Gothic architecture?")

    assert len(results) > 0


def test_retrieval_returns_requested_number_of_chunks(vector_store):
    results = retrieve_chunks(vector_store, "What is Gothic architecture?")

    assert len(results) == 4


def test_pointed_arch_search_returns_gothic_first(vector_store):
    results = retrieve_chunks(vector_store, "pointed arch")

    top_style = results[0].metadata.get("style", "")

    assert top_style == "Gothic"


def test_answer_includes_a_real_filename_citation(assistant):
    answer, chunks = assistant.ask("What makes Gothic architecture distinctive?")

    assert ".md" in answer


def test_out_of_scope_question_is_declined(assistant):
    answer, chunks = assistant.ask("What is the best restaurant near Durham Cathedral?")

    assert "do not have" in answer.lower() or "don't have" in answer.lower()


def test_memory_or_retrieval_answers_a_followup_correctly(assistant):
    assistant.ask("What makes Gothic architecture different from Romanesque?")

    answer, chunks = assistant.ask("Which one came first?")

    assert "romanesque" in answer.lower()

    # If chunks WERE retrieved, they should be relevant (Gothic or
    # Romanesque), not a random unrelated style - that would indicate
    # the original pronoun-retrieval bug, not correct behaviour.
    if len(chunks) > 0:
        styles_found = [c.metadata.get("style", "") for c in chunks]
        assert any(s in ["Gothic", "Romanesque"] for s in styles_found)


def test_ambiguous_followup_after_image_style_retrieves_correct_documents(assistant):
    assistant.ask("Tell me about Gothic architecture.")

    answer, chunks = assistant.ask(
        "What conservation challenges are typical for buildings in this style?"
    )

    filenames = [c.metadata.get("filename", "") for c in chunks]

    mentions_gothic = "gothic" in answer.lower()
    has_gothic_source = any("gothic" in f.lower() for f in filenames)

    assert mentions_gothic or has_gothic_source