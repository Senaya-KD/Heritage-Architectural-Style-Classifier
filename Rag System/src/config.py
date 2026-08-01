import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Check that .env exists in the project "
        "root and contains GEMINI_API_KEY=your_key"
    )

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KB_STYLES_DIR = PROJECT_ROOT / "knowledge_base" / "styles"
KB_GENERAL_DIR = PROJECT_ROOT / "knowledge_base" / "general"
KB_PDFS_DIR = PROJECT_ROOT / "knowledge_base" / "pdfs"

CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
MODEL_PATH = PROJECT_ROOT / "models" / "final_model.keras"

# LLM 
LLM_MODEL = "gemini-flash-lite-latest"
LLM_TEMPERATURE = 0.2

# Embedding
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval 
RETRIEVAL_K = 4

# CV confidence threshold 
CONFIDENCE_THRESHOLD = 0.70

# Class names, alphabetical order (matches Keras's index assignment)
CLASS_NAMES = [
    "Art Deco architecture",
    "Art Nouveau architecture",
    "Baroque architecture",
    "Byzantine architecture",
    "Georgian architecture",
    "Gothic architecture",
    "Romanesque architecture",
    "Tudor Revival architecture",
]

IMG_SIZE = (224, 224)