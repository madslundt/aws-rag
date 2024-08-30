import argparse
import os
from typing import Optional
from langchain.retrievers.multi_query import LineListOutputParser
from env import CONFIG_PATH, DOCSTORE_PATH, DOCSTORE_TABLE_NAME, DOCUMENTS_PATH, OLLAMA_MODEL, PARENT_DOC_ID
from utils import get_sqlitestore, load_json_file, verbose_print
from utils.get_vectorstore import get_vectorstore
from langchain_core.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

def main() -> None:
    """
    Main function to handle command-line arguments and interactive querying.

    Parses command-line arguments to determine if a query should be executed directly
    or if the user should be prompted for input in an interactive loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    if query_text:
        query_rag(query_text)
    else:
        interactive_query_loop()

def interactive_query_loop() -> None:
    """
    Runs an interactive loop to accept queries from the user until 'exit' or 'q' is entered.

    The user is prompted to enter queries, which are then processed to retrieve and display relevant information.
    """
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"exit", "q"}:
            break
        if query:
            query_rag(query)

def get_prompt(template: str, input_variables: list[str]) -> PromptTemplate:
    """
    Creates a PromptTemplate with a given template and input variables.

    Args:
        template (str): The template string used to structure the prompt.
        input_variables (list[str]): List of variable names used in the template.

    Returns:
        PromptTemplate: A configured PromptTemplate object.
    """
    return PromptTemplate(template=template, input_variables=input_variables)

def get_metadata_field_info() -> list[AttributeInfo]:
    """
    Returns metadata field information for document retrieval.

    This includes details like unique ID, source, hash, and page number for document chunks.

    Returns:
        list[AttributeInfo]: List of AttributeInfo objects describing metadata fields.
    """
    return [
        AttributeInfo(
            name="id",
            description="Unique id of the document chunk",
            type="string",
        ),
        AttributeInfo(
            name="source",
            description="Source name of the document from which the information was extracted.",
            type="string",
        ),
        AttributeInfo(
            name="hash",
            description="SHA1 Hash of the document chunks",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="Page the chunk appears on in the main document",
            type="int",
        )
    ]

def get_keywords_from_config() -> list[str]:
    """
    Extracts a list of keywords from the configuration file.

    Returns:
        list[str]: A list of unique keywords found in the configuration file.
        None: If the configuration file or keyword entries are missing.
    """
    config: Optional[dict[str, dict]] = load_json_file(CONFIG_PATH)
    if not config or not config['documents']:
        return None

    result: list[str] = list(set(keyword for value in config['documents'] for keyword in value['keywords']))

    return result

def get_filenames_based_on_keywords_from_config(keywords: list[str]) -> list[str]:
    """
    Retrieves a list of filenames from the configuration that match given keywords.

    Args:
        keywords (list[str]): A list of keywords to match against document entries in the configuration.

    Returns:
        list[str]: List of filenames that match the provided keywords.
        None: If no matching documents are found or the configuration is missing.
    """
    config: Optional[dict[str, dict]] = load_json_file(CONFIG_PATH)
    if not config or not config['documents']:
        return None

    documents = [document for document in config['documents'] if any(keyword in document['keywords'] for keyword in keywords)]
    result = [os.path.join(DOCUMENTS_PATH, pdf['filename']) for document in documents for pdf in document['pdfs']]

    return result

def query_rag(query_text: str) -> None:
    """
    Handles the query process, from generating alternative queries to retrieving relevant documents and generating a response.

    This function orchestrates the entire query-response process including generating variations of the user query,
    finding relevant documents, extracting relevant information, and constructing a final response.

    Args:
        query_text (str): The text of the user's query.
    """
    try:
        vectorstore = get_vectorstore()
        docstore = get_sqlitestore(DOCSTORE_PATH, DOCSTORE_TABLE_NAME)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=PARENT_DOC_ID,
        )

        llm = Ollama(model=OLLAMA_MODEL)
        query_output_parser = LineListOutputParser()

        query_prompt = get_prompt(
            template="""You are an AI language model assistant. Your task is to generate five
            different versions of the given user question to retrieve relevant documents from a vector
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search.
            Provide these and only these alternative questions separated by newlines.
            Original question: {question}""",
            input_variables=["question"]
        )

        model = query_prompt | llm | query_output_parser
        questions = model.invoke({"question": query_text})[1:]

        verbose_print("\n".join(questions), "\n")

        keywords_prompt = get_prompt(
            template="""You are an AI language model assistant. Your task is to identify related keywords for the provided user questions.

            Keywords:
            {keywords}

            ---------

            Your goal is to find related keywords that will help improve the results of distance-based similarity searches. Provide only the related keywords in a single list, each on a new line.

            Questions:
            {questions}""",
            input_variables=["keywords", "questions"]
        )

        keywords = " - " + "\n - ".join(get_keywords_from_config())
        questions_str = " - " + "\n - ".join(questions)

        model = keywords_prompt | llm | query_output_parser
        relevant_keywords = model.invoke({"keywords": keywords, "questions": questions_str})[1:]
        relevant_sources = get_filenames_based_on_keywords_from_config(relevant_keywords)

        relevant_keywords_str = "\n".join(relevant_keywords)
        verbose_print(f"Relevant keywords:\n{relevant_keywords_str}")

        relevant_docs, source_pages = retrieve_relevant_docs(questions, retriever, relevant_sources)
        response_text = generate_response(query_text, relevant_docs)

        print(f"Response:\n{response_text}\n\nSources: {source_pages}")
        return response_text

    except Exception as e:
        print(f"An error occurred: {e}")

def retrieve_relevant_docs(questions: list[str], retriever: BaseRetriever, relevant_sources: list[str]) -> tuple[list, list]:
    """
    Retrieves relevant documents based on generated questions and relevant sources.

    Args:
        questions (list[str]): List of alternative questions generated from the original query.
        retriever (BaseRetriever): The retriever instance used to find relevant documents.
        relevant_sources (list[str]): List of source file paths to narrow down the search.

    Returns:
        tuple[list, list]: A tuple containing a list of relevant documents and a list of source page references.
    """
    relevant_docs = []
    source_ids = set()
    source_pages = []

    for search in questions:
        _relevant_docs = [
            doc for doc in retriever.invoke(search, where={"source": {"$in": relevant_sources}}) if doc.metadata.get("id") not in source_ids
        ]
        relevant_docs.extend(_relevant_docs)
        source_ids.update(doc.metadata.get("id") for doc in _relevant_docs)
        source_pages.extend(f"{doc.metadata.get('source', 'unknown')} page {doc.metadata.get('page', 'unknown')}" for doc in _relevant_docs)

    return relevant_docs, source_pages

def generate_response(query_text: str, relevant_docs: list[Document]) -> str:
    """
    Generates a response based on the context provided by relevant documents.

    Args:
        query_text (str): The original query text from the user.
        relevant_docs (list[Document]): A list of documents that provide context for answering the query.

    Returns:
        str: The generated response text.
    """
    context_text = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = get_prompt(
        template="""Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}""",
        input_variables=["context", "question"]
    )

    llm = Ollama(model=OLLAMA_MODEL)
    model = prompt | llm
    response_text = model.invoke({"context": context_text, "question": query_text})
    return response_text

if __name__ == "__main__":
    main()
