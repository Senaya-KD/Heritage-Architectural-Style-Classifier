
from src.classifier import load_classifier
from src.rag_chain import load_vector_store, HeritageAssistant
from src.cv_rag_bridge import explain_uploaded_image, print_result


def format_sources_for_file(chunks):
    if len(chunks) == 0:
        return "  (answered from conversation history - no new documents retrieved)\n"

    lines = ""
    for i, chunk in enumerate(chunks):
        style = chunk.metadata.get("style", "N/A")
        doc_type = chunk.metadata.get("doc_type", "reference")
        filename = chunk.metadata.get("filename", "unknown file")
        lines += "  Source " + str(i + 1) + ": " + style + " - " + doc_type + " -> " + filename + "\n"

    return lines


print("Loading CV model...")
model = load_classifier()

print("Loading knowledge base...")
vector_store = load_vector_store()

assistant = HeritageAssistant(vector_store)

print("Ready.\n")
print("=" * 60)

TEST_IMAGE_PATH = "test_image_3.jpg"

result = explain_uploaded_image(TEST_IMAGE_PATH, assistant, model)
print_result(result)

with open("low_confidence_output.txt", "w", encoding="utf-8") as f:

    prediction = result["prediction"]

    f.write("Image: " + TEST_IMAGE_PATH + " (Chicago school - NOT a trained class)\n\n")
    f.write("CV Prediction (forced choice among the 8 trained styles):\n")
    f.write("  Top pick: " + prediction["style"] + "\n")
    f.write("  Confidence: " + str(round(prediction["confidence"] * 100, 1)) + "%\n\n")

    f.write("Top 3 candidates:\n")
    for style, conf in prediction["top_3"]:
        f.write("  " + style + ": " + str(round(conf * 100, 1)) + "%\n")
    f.write("\n")

    f.write("Question sent to assistant:\n  " + result["question"] + "\n\n")
    f.write("Assistant's explanation:\n" + result["answer"] + "\n\n")

    f.write("Sources:\n")
    f.write(format_sources_for_file(result["chunks"]))

print("\n\nFull output saved to low_confidence_output.txt")
