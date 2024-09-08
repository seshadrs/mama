import os


for flag in ("OPENAI_API_KEY", "PINECONE_API_KEY"):
    if not os.environ.get(flag):
        raise ValueError(f"{flag} is not set")

OPENAI_MODEL_NAME = "gpt-4o-mini"

PINECONE_INDEX = "mama-medical-docs"
PINECONE_INDEX_NAMESPACE = "sesh"
MEDICAL_DOCS_DIRPATH = "data/medical_documents/sesh"

METADATA_TEXT_FIELD = "clean_content"
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
EMBEDDING_MODEL_DIMENSIONALITY = 1536
