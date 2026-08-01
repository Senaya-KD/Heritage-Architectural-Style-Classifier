import streamlit as st
from src.classifier import load_classifier
from src.rag_chain import load_vector_store, HeritageAssistant
from src.cv_rag_bridge import explain_uploaded_image

# Page setup 

st.set_page_config(
    page_title="NHPT Heritage Assistant",
    page_icon="🏛️",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f4f1ea;
    }
    h1, h2, h3 {
        color: #5a3921;
    }
    .stButton button {
        background-color: #8b4513;
        color: white;
        border-radius: 6px;
        border: none;
    }
    .stButton button:hover {
        background-color: #6b3410;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Header 

st.title("🏛️ National Heritage Preservation Trust")
st.caption("Ask about architectural styles, or attach a photo of a building to have it identified.")


#  Sidebar 

with st.sidebar:
    st.header("About this Assistant")
    st.write(
        "This assistant recognises **8 architectural styles** found "
        "across UK heritage sites, using a trained image classifier "
        "combined with a curated knowledge base."
    )
    st.markdown("**Styles covered:**")
    st.markdown(
        "- Romanesque\n"
        "- Byzantine\n"
        "- Gothic\n"
        "- Georgian\n"
        "- Tudor Revival\n"
        "- Baroque\n"
        "- Art Deco\n"
        "- Art Nouveau"
    )
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.assistant = HeritageAssistant(vector_store) if "vector_store" in dir() else None
        st.rerun()


#  Load everything ONCE

@st.cache_resource
def load_everything():
    model = load_classifier()
    vector_store = load_vector_store()
    return model, vector_store


with st.spinner("Loading model and knowledge base..."):
    model, vector_store = load_everything()


# Session state 

if "assistant" not in st.session_state:
    st.session_state.assistant = HeritageAssistant(vector_store)

if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role", "content", "image"(optional)}


#  Display the full chat history 

for message in st.session_state.messages:

    if message["role"] == "visitor":
        with st.chat_message("user"):
            if message.get("image") is not None:
                st.image(message["image"], width=250)
            if message.get("content"):
                st.write(message["content"])

    else:
        with st.chat_message("assistant", avatar="🏛️"):

            # If this assistant message includes a CV prediction,
            # show it as a small highlighted panel above the answer.
            if message.get("prediction") is not None:
                prediction = message["prediction"]
                confidence_percent = round(prediction["confidence"] * 100, 1)

                st.markdown("**Predicted style: " + prediction["style"] + "**")
                st.progress(prediction["confidence"], text=str(confidence_percent) + "% confidence")

                if confidence_percent < 70:
                    st.warning("Not fully certain — top candidates:")
                    for style_name, style_confidence in prediction["top_3"]:
                        pct = round(style_confidence * 100, 1)
                        st.write("• " + style_name + ": " + str(pct) + "%")

            st.write(message["content"])


#  Single input row: text + file attach, like Claude 

user_input = st.chat_input(
    "Ask about an architectural style, or attach a photo...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"]
)

if user_input:

    typed_text = user_input.text
    attached_file = user_input.files[0] if user_input.files else None

    # ---- Case 1: an image was attached ----
    if attached_file is not None:

        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(attached_file.getbuffer())

        st.session_state.messages.append({
            "role": "visitor",
            "content": typed_text,   # may be empty if they only attached a photo
            "image": attached_file
        })

        with st.spinner("Analysing the photograph..."):
            result = explain_uploaded_image(
                temp_path,
                st.session_state.assistant,
                model
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "prediction": result["prediction"]
        })

    #Case 2: plain typed question, no image
    else:

        st.session_state.messages.append({
            "role": "visitor",
            "content": typed_text
        })

        with st.spinner("Thinking..."):
            answer, chunks = st.session_state.assistant.ask(typed_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    st.rerun()