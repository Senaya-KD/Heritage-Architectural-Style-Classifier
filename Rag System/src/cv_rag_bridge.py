from src.config import CONFIDENCE_THRESHOLD
from src.classifier import load_classifier, predict_style
from src.rag_chain import load_vector_store, HeritageAssistant, build_context_text


def build_question_from_prediction(prediction):
    style = prediction["style"]
    confidence = prediction["confidence"]

    if confidence >= CONFIDENCE_THRESHOLD:
        question = (
            "A visitor has photographed a building believed to be "
            + style + ". Explain what makes this style distinctive, "
            "including its history and key visual features."
        )
    else:
        top_3_names = [name for name, conf in prediction["top_3"]]
        style_list = ", ".join(top_3_names)

        question = (
            "A visitor has photographed a building. The most likely "
            "styles are: " + style_list + ". Explain the key "
            "differences between these styles so the visitor can "
            "identify which one they are actually looking at."
        )

    return question


def explain_uploaded_image(image_path, assistant, model):
    prediction = predict_style(model, image_path)

    if prediction["confidence"] >= CONFIDENCE_THRESHOLD:
        # Confident - single style, normal retrieval is fine.
        # assistant.ask() already uses filename-based inline citations.
        question = build_question_from_prediction(prediction)
        answer, chunks = assistant.ask(question)

    else:
        # Not confident - retrieve SEPARATELY for each candidate style,
        # so a strong match for one style cannot starve out the others.
        top_3_names = [name for name, conf in prediction["top_3"]]

        all_chunks = []
        for style_name in top_3_names:
            style_chunks = assistant.vector_store.similarity_search(
                "distinctive features of " + style_name,
                k=2,
                filter={"style": style_name.replace(" architecture", "")}
            )
            all_chunks.extend(style_chunks)

        question = build_question_from_prediction(prediction)

        context_text = build_context_text(all_chunks)
        history_text = assistant.format_history_text()

        prompt = f"""You are a helpful assistant for a UK heritage trust, explaining architecture to visitors.

The visitor's photo could not be confidently classified. Compare ALL of the
candidate styles shown in the CONTEXT below, giving each one fair coverage.

After each fact you use, write the exact filename it came from in
brackets, directly in the sentence - for example:
"Georgian buildings are symmetrical (04_georgian_features.md)."
Do NOT use "Source 1" style numbering. Use the real filename shown in
the [brackets] above each piece of context.

CONVERSATION SO FAR:
{history_text}

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

        response = assistant.llm.invoke(prompt)
        answer = response.content[0]["text"]

        assistant.history.add_user_message(question)
        assistant.history.add_ai_message(answer)

        chunks = all_chunks

    result = {
        "prediction": prediction,
        "question": question,
        "answer": answer,
        "chunks": chunks
    }

    return result


def print_result(result):
    prediction = result["prediction"]

    print("CV Model Prediction")
    print("  Style:", prediction["style"])
    print("  Confidence:", round(prediction["confidence"] * 100, 1), "%")

    if prediction["confidence"] < CONFIDENCE_THRESHOLD:
        print("  (Below the", int(CONFIDENCE_THRESHOLD * 100),
              "% threshold - showing top 3 candidates instead)")
        print("  Top 3:")
        for style, conf in prediction["top_3"]:
            print("   -", style, ":", round(conf * 100, 1), "%")

    print()
    print("Question sent to assistant:")
    print(" ", result["question"])
    print()

    print("Assistant's explanation:")
    print(result["answer"])


def save_result_to_file(result, filename="example_output.txt"):
    prediction = result["prediction"]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("CV Model Prediction\n")
        f.write("  Style: " + prediction["style"] + "\n")
        f.write("  Confidence: " + str(round(prediction["confidence"] * 100, 1)) + "%\n\n")

        if prediction["confidence"] < CONFIDENCE_THRESHOLD:
            f.write("  (Below threshold - top 3 shown instead)\n")
            for style, conf in prediction["top_3"]:
                f.write("   - " + style + ": " + str(round(conf * 100, 1)) + "%\n")
            f.write("\n")

        f.write("Question sent to assistant:\n  " + result["question"] + "\n\n")
        f.write("Assistant's explanation:\n" + result["answer"] + "\n")


# Quick test - only runs if this file is executed directly
if __name__ == "__main__":

    print("Loading CV model...")
    model = load_classifier()

    print("Loading knowledge base...")
    vector_store = load_vector_store()
    assistant = HeritageAssistant(vector_store)

    print("Ready.\n")
    print("=" * 60)

    TEST_IMAGE_PATH = "test_image.jpg"

    result = explain_uploaded_image(TEST_IMAGE_PATH, assistant, model)

    print_result(result)
    save_result_to_file(result)

    print("\nFull output also saved to example_output.txt")