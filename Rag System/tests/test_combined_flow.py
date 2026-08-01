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


#  Step 1: visitor uploads a photo 
print("STEP 1: Visitor uploads a photo\n")

TEST_IMAGE_PATH = "test_image.jpg"

result = explain_uploaded_image(TEST_IMAGE_PATH, assistant, model)
print_result(result)


#  Step 2: visitor asks a typed follow-up question 
print("\n" + "=" * 60)
print("STEP 2: Visitor asks a follow-up question\n")

follow_up_question = "How old is this architectural style, roughly?"

print("Visitor:", follow_up_question)
print()

follow_up_answer, follow_up_chunks = assistant.ask(follow_up_question)

print("Assistant:", follow_up_answer)


# Step 3: one more follow-up, to really test memory depth 
print("\n" + "=" * 60)
print("STEP 3: A second follow-up, testing deeper memory\n")

second_follow_up = "Which UK building did you mention as an example of it?"

print("Visitor:", second_follow_up)
print()

second_answer, second_chunks = assistant.ask(second_follow_up)

print("Assistant:", second_answer)


# Save the whole thing to a file, WITH real filenames for every source 
with open("combined_flow_output.txt", "w", encoding="utf-8") as f:

    f.write("STEP 1: Image upload\n")
    f.write("Style: " + result["prediction"]["style"] + "\n")
    f.write("Confidence: " + str(round(result["prediction"]["confidence"] * 100, 1)) + "%\n")
    f.write("Question: " + result["question"] + "\n")
    f.write("Answer: " + result["answer"] + "\n")
    f.write("Sources:\n")
    f.write(format_sources_for_file(result["chunks"]))
    f.write("\n")

    f.write("STEP 2: Follow-up question\n")
    f.write("Visitor: " + follow_up_question + "\n")
    f.write("Answer: " + follow_up_answer + "\n")
    f.write("Sources:\n")
    f.write(format_sources_for_file(follow_up_chunks))
    f.write("\n")

    f.write("STEP 3: Second follow-up\n")
    f.write("Visitor: " + second_follow_up + "\n")
    f.write("Answer: " + second_answer + "\n")
    f.write("Sources:\n")
    f.write(format_sources_for_file(second_chunks))

print("\n\nFull combined conversation saved to combined_flow_output.txt")