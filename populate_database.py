import argparse
import hashlib
import os
import shutil
from typing import Optional
from env import CHROMA_PATH, DOCSTORE_PATH, DOCSTORE_TABLE_NAME, DOCUMENT_HASHES_TABLE_NAME, DOCUMENTS_PATH, PARENT_CHUNK_SIZE, PARENT_DOC_ID, CHILD_CHUNK_SIZE
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma.vectorstores import VectorStore
from utils import get_sqlitestore, get_vectorstore, verbose_print

def main(path: str) -> None:
    """
    Main function to handle database reset and document processing.

    Args:
        path (str): The root directory path where documents are located.
    """
    args = parse_arguments()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Load, split, and add documents to the database
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)

                local_file_hash = get_file_hash(file_path)
                document_hash = get_document_hash_from_store(file_path)

                if local_file_hash == document_hash:
                    print(f"Skipping {file_path} because it already exists in the database")
                    continue
                elif not document_hash:
                    print(f"{file_path} does not exist in the database")
                else:
                    print(f"{file_path} exists in the database but an older different version.")

                documents = load_documents(file_path)

                print("Splitting document into chunks...")
                docs, sub_docs = split_documents(documents, parent_chunk_size=PARENT_CHUNK_SIZE, child_chunk_size=CHILD_CHUNK_SIZE)

                print("Adding document and chunks to Chroma and doc store...")
                add_documents_to_store(docs, sub_docs)

                add_document_hash(file_path)

def get_document_hashes_store():
    """
    Retrieve the SQLite store for document hashes.

    Returns:
        SQLite store instance configured for document hashes.
    """
    store = get_sqlitestore(DOCSTORE_PATH, DOCUMENT_HASHES_TABLE_NAME)
    return store

def add_document_hash(file_path: str) -> None:
    """
    Add a document hash to the database.

    Args:
        file_path (str): The file path of the document to hash and store.
    """
    store = get_document_hashes_store()
    store.mset([
        (file_path, get_file_hash(file_path))
    ])

def get_document_hash_from_store(file_path: str) -> Optional[str]:
    """
    Retrieve the hash of a document from the database.

    Args:
        file_path (str): The file path of the document.

    Returns:
        Optional[str]: The hash of the document if it exists in the store, otherwise None.
    """
    store = get_document_hashes_store()
    file_hashes = store.mget([file_path])
    if file_hashes:
        return file_hashes[0]

    return None

def get_file_hash(file_path: str) -> str:
    """
    Calculate the SHA-1 hash of a file.

    Args:
        file_path (str): Path to the file to hash.

    Returns:
        str: The SHA-1 hash of the file.
    """
    sha1 = hashlib.sha1()
    buffer_size = 65536
    with open(file_path, 'rb') as file:
        while chunk := file.read(buffer_size):
            sha1.update(chunk)

    return sha1.hexdigest()

def parse_arguments() -> None:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    return parser.parse_args()

def load_documents(path: str) -> list[Document]:
    """
    Load all PDF files in a given path.

    Args:
        path (str): Path to the PDF file.

    Returns:
        list[Document]: A list containing the loaded documents.
    """
    loader = PyPDFLoader(path)
    document = loader.load()

    return document

def split_documents(
    documents: list[Document],
    parent_chunk_size: int = 400,
    child_chunk_size: int = 0
) -> tuple[list[Document], list[Document]]:
    """
    Split documents into chunks and sub-chunks.

    If child_chunk_size is > 0, split the documents into sub-chunks as well.
    If child_chunk_size is 0, only split the documents into chunks.

    Args:
        documents (list[Document]): List of documents to split.
        parent_chunk_size (int, default 400): Size of parent chunks.
        child_chunk_size (int, default 0): Size of child chunks.

    Returns:
        tuple[list[Document], list[Document]]: Tuple containing lists of parent and sub-documents.
    """
    parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)

    new_documents = generate_documents_with_metadata(parent_text_splitter.split_documents(documents))
    sub_documents = []

    child_text_splitter = None
    if child_chunk_size > 0:
        child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)

    for idx, document in enumerate(new_documents):
        if child_text_splitter:
            _sub_documents = generate_documents_with_metadata(
                child_text_splitter.split_documents([document]),
                idx
            )

            for sub_document in _sub_documents:
                sub_document.metadata[PARENT_DOC_ID] = document.metadata.get("id")
                sub_documents.append(sub_document)

    return new_documents, sub_documents

def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """
    Divide a list into chunks of specified size.

    Args:
        lst (list): List to be chunked.
        chunk_size (int): Max size of each chunk.

    Returns:
        list[list]: List of chunked lists.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def add_documents_to_store(
        documents: list[Document],
        sub_documents: list[Document] = [],
        chunk_size: int = 500
) -> None:
    """
    Add documents to the vectorstore.

    If sub_documents is an empty list, then add the documents to the vectorstore.
    Else add sub_documents to the vectorstore and add documents to the bytestore.

    Args:
        documents (list[Document]): List of documents to add to the bytestore (if sub_documents is empty, then they'll be added to the vectorstore).
        sub_documents (list[Document], optional): List of sub-documents to add to the vectorstore. Default is an empty list.
        chunk_size (int, default 500): Number of documents to add in each batch.
    """
    vectorstore = get_vectorstore()
    vectorstore_documents = sub_documents if sub_documents else documents
    docstore = get_sqlitestore(DOCSTORE_PATH, DOCSTORE_TABLE_NAME)

    if sub_documents:
        docstore.mset(list(zip([doc.metadata["id"] for doc in documents], documents)))

    existing_ids = set(vectorstore.get(include=[])["ids"])
    verbose_print(f"\tNumber of existing documents in DB: {len(existing_ids)}")

    new_chunks, updated_chunks = get_documents_to_add_or_update(vectorstore_documents, existing_ids, vectorstore)

    if new_chunks:
        verbose_print(f"\t👉 Adding new documents: {len(new_chunks)}")
        add_or_update_documents_to_vectorstore(new_chunks, vectorstore, chunk_size)
    else:
        verbose_print("\t✅ No new documents to add")

    if updated_chunks:
        verbose_print(f"\t👉 Updating documents: {len(updated_chunks)}")
        add_or_update_documents_to_vectorstore(updated_chunks, vectorstore, chunk_size)
    else:
        verbose_print("\t✅ All documents are up-to-date")

def get_documents_to_add_or_update(
        documents: list[Document],
        existing_ids: list[str],
        vectorstore: VectorStore
) -> tuple[list[Document], list[Document]]:
    """
    Get documents that need to be added or updated.

    A document is added if its ID is not present in the vectorstore. A document is updated if the hash is different.
    Documents that are already present and have the same hash are ignored.

    Args:
        documents (list[Document]): List of documents with IDs.
        existing_ids (list[str]): List of existing document IDs in the vectorstore.
        vectorstore (VectorStore): Vectorstore instance.

    Returns:
        tuple[list[Document], list[Document]]: Tuple containing lists of new and updated documents.
    """
    new_documents = []
    updated_documents = []

    for document in documents:
        id = document.metadata["id"]
        hash = document.metadata["hash"]

        if id not in existing_ids:
            new_documents.append(document)
        else:
            existing_doc = vectorstore.get(ids=[id])
            if existing_doc and existing_doc["metadatas"][0].get("hash") != hash:
                updated_documents.append(document)

    return new_documents, updated_documents

def add_or_update_documents_to_vectorstore(
        documents: list[Document],
        vectorstore: VectorStore,
        chunk_size: int = 500
) -> None:
    """
    Add or update documents in the vectorstore in batches.

    Args:
        documents (list[Document]): List of documents to add or update.
        vectorstore (VectorStore): Vectorstore instance.
        chunk_size (int, default 500): Number of documents to add in each batch.
    """
    for idx, chunk_group in enumerate(chunk_list(documents, chunk_size)):
        vectorstore.add_documents(chunk_group, ids=[chunk.metadata["id"] for chunk in chunk_group])
        verbose_print(f"\t👉 Added: {chunk_size * idx + len(chunk_group)}")

def generate_hash(text: str) -> str:
    """
    Generate a SHA-256 hash for the given text.

    Args:
        text (str): Text to generate hash for.

    Returns:
        str: The SHA-256 hash of the text.
    """
    return hashlib.sha256(text.encode()).hexdigest()

def generate_documents_with_metadata(documents: list[Document], source_chunk_idx: Optional[int] = None) -> list[Document]:
    """
    Generate metadata for documents, including unique IDs and hash.

    Args:
        documents (list[Document]): List of documents to generate metadata for.
        source_chunk_idx (Optional[int], default None): Index of the source chunk, if applicable.

    Returns:
        list[Document]: List of documents with updated metadata.
    """
    last_page_id = None
    current_chunk_index = 0

    for document in documents:
        source = document.metadata.get("source")
        page = document.metadata.get("page")
        current_page_id = f"{source}"
        if source_chunk_idx is not None:
            current_page_id += f":{source_chunk_idx}"

        current_page_id += f":{page or 0}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        id = f"{current_page_id}:{current_chunk_index}"
        document.metadata["id"] = id
        document.metadata["hash"] = generate_hash(document.page_content)
        last_page_id = current_page_id

    return documents

def clear_database() -> None:
    """
    Clear both the Chroma and docstore databases.

    This removes all data from the specified database paths.
    """
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    if os.path.exists(DOCSTORE_PATH):
        os.remove(DOCSTORE_PATH)

if __name__ == "__main__":
    main(DOCUMENTS_PATH)
