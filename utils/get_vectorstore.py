from env import CHROMA_COLLECTION_NAME, CHROMA_PATH
import chromadb
from utils.get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


def get_vectorstore() -> Chroma:
    persistent_client = chromadb.PersistentClient(
        path=CHROMA_PATH,
    )

    db = Chroma(
        client=persistent_client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_function()
    )

    return db
