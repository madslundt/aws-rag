from langchain_community.embeddings.ollama import OllamaEmbeddings
from env import OLLAMA_EMBEDDING_MODEL


def get_embedding_function() -> OllamaEmbeddings:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
    return embeddings
