from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory as ChatMessageHistory
from src.config import (
    EMBEDDING_MODEL,
    CHROMA_DB_DIR,
    RETRIEVAL_K,
    GEMINI_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE
)

RELEVANCE_CUTOFF = 1.0


def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vector_store = Chroma(
        persist_directory=str(CHROMA_DB_DIR),
        embedding_function=embedding_model
    )

    return vector_store


def retrieve_chunks(vector_store, question):
    results = vector_store.similarity_search(question, k=RETRIEVAL_K)

    return results


def build_context_text(chunks):

    context_parts = []

    for chunk in chunks:
        filename = chunk.metadata.get("filename", "unknown_file.md")
        context_parts.append("[" + filename + "]\n" + chunk.page_content)

    context_text = "\n\n".join(context_parts)

    return context_text


def print_sources(chunks):

    if len(chunks) == 0:
        print("Sources used: none (answered from conversation history)")
        return

    print("Sources used:")

    for i, chunk in enumerate(chunks):
        style = chunk.metadata.get("style", "N/A")
        doc_type = chunk.metadata.get("doc_type", "reference")
        filename = chunk.metadata.get("filename", "unknown file")

        print("  Source", i + 1, ":", style, "-", doc_type, "->", filename)


def ask_question(vector_store, question):
    chunks = retrieve_chunks(vector_store, question)

    context_text = build_context_text(chunks)

    prompt = f"""You are a helpful assistant for a UK heritage trust, explaining architecture to visitors.

Answer the question using ONLY the information in the CONTEXT below.
If the answer is not in the CONTEXT, say you do not have that information.

CITATION RULE - THIS IS MANDATORY:
After each fact you use, write the exact filename it came from in
brackets, directly in the sentence. Example:
"Gothic arches are pointed (03_gothic_features.md)."
NEVER write "Source 1", "Source 2", or any numbered source. ONLY use
the real filename shown in the [brackets] above each piece of context.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=LLM_TEMPERATURE
    )

    response = llm.invoke(prompt)
    answer_text = response.content[0]["text"]

    return answer_text, chunks


class HeritageAssistant:

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.history = ChatMessageHistory()

        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE
        )

    def format_history_text(self):

        if len(self.history.messages) == 0:
            return "(This is the start of the conversation - no earlier messages.)"

        history_lines = []

        for message in self.history.messages:

            if message.type == "human":
                history_lines.append("Visitor: " + message.content)
            else:
                history_lines.append("Assistant: " + message.content)

        return "\n".join(history_lines)

    def rewrite_question_using_history(self, question):
        if len(self.history.messages) == 0:
            # Nothing to resolve against yet - the question already
            # stands on its own
            return question

        history_text = self.format_history_text()

        rewrite_prompt = f"""Rewrite the NEW QUESTION below so it makes complete sense on its own,
without needing the conversation history to understand it. Replace
words like "this", "that", "it", or "the earlier one" with the actual
subject they refer to, based on the conversation.

If the NEW QUESTION already makes sense on its own, return it unchanged.

Return ONLY the rewritten question, nothing else - no explanation.

CONVERSATION SO FAR:
{history_text}

NEW QUESTION:
{question}

REWRITTEN STANDALONE QUESTION:"""

        response = self.llm.invoke(rewrite_prompt)
        rewritten = response.content[0]["text"].strip()

        return rewritten

    def ask(self, question):

        search_question = self.rewrite_question_using_history(question)

        results_with_scores = self.vector_store.similarity_search_with_score(
            search_question, k=RETRIEVAL_K
        )

        relevant_chunks = []
        for chunk, score in results_with_scores:
            if score < RELEVANCE_CUTOFF:
                relevant_chunks.append(chunk)

        history_text = self.format_history_text()

        if len(relevant_chunks) == 0:
            context_text = "(No closely matching documents were found for this question.)"
        else:
            context_text = build_context_text(relevant_chunks)

        prompt = f"""You are a helpful assistant for a UK heritage trust, explaining architecture to visitors.

You have two sources of information: the CONVERSATION SO FAR, and the CONTEXT below.

CITATION RULE - THIS IS MANDATORY:
If you use a fact from the CONTEXT, write the exact filename it came
from in brackets, directly in the sentence. Example:
"Gothic arches are pointed (03_gothic_features.md)."
NEVER write "Source 1", "Source 2", or any numbered source. ONLY use
the real filename shown in the [brackets] above each piece of context.

If the CONTEXT says no matching documents were found, but the answer was
already established earlier in the CONVERSATION SO FAR, answer using that
instead - no filename citation is needed in that case, since it is not
from a document. Say which earlier point you are drawing from instead.

Only say "I do not have that information" if NEITHER the CONTEXT nor the
CONVERSATION SO FAR contains the answer.

CONVERSATION SO FAR:
{history_text}

CONTEXT:
{context_text}

NEW QUESTION:
{question}

ANSWER:"""

        response = self.llm.invoke(prompt)
        answer_text = response.content[0]["text"]

        self.history.add_user_message(question)
        self.history.add_ai_message(answer_text)

        return answer_text, relevant_chunks


# Quick test - only runs if this file is executed directly
if __name__ == "__main__":

    vector_store = load_vector_store()

    assistant = HeritageAssistant(vector_store)

    conversation = [
        "What makes Gothic architecture different from Romanesque?",
        "Which one came first?",
        "Is Durham Cathedral an example of that earlier style?"
    ]

    for question in conversation:
        print("=" * 60)
        print("Visitor:", question)
        print()

        answer, chunks = assistant.ask(question)

        print("Assistant:", answer)
        print()
        print_sources(chunks)
        print()