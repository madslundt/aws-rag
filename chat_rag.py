import argparse
from env import DOCSTORE_PATH, DOCSTORE_TABLE_NAME, OLLAMA_MODEL, PARENT_DOC_ID
from utils import get_sqlitestore
from utils.get_vectorstore import get_vectorstore
from langchain_community.llms.ollama import Ollama
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables import Runnable
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever

def main() -> None:
    """
    Main function to handle command-line arguments and interactive querying.

    Parses command-line arguments to determine if a query should be executed directly
    or if the user should be prompted for input in an interactive loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_text", type=str, help="The query text.")
    interactive_query_loop()

def get_contextualize_question_prompt() -> ChatPromptTemplate:
    """
    Sets up a question contextualization prompt that reformulates user questions based on chat history.

    Returns:
        ChatPromptTemplate: The prompt template for contextualizing questions.
    """
    contextualize_q_system_prompt = """
        Given a chat history and the latest user question
        which might reference context in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do NOT answer the question, just
        reformulate it if needed and otherwise return it as is.
    """

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return contextualize_q_prompt

def get_question_answering_prompt() -> ChatPromptTemplate:
    """
    Sets up a question-answering prompt that uses the retrieved context to answer the user's question concisely.

    Returns:
        ChatPromptTemplate: The prompt template for answering questions.
    """
    qa_system_prompt = """
        You are an assistant for question-answering tasks. Use
        the following pieces of retrieved context to answer the
        question. If you don't know the answer, just say that you
        don't know. Use three sentences maximum and keep the answer
        concise.

        {context}
    """

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    return qa_prompt

def get_rag_chain() -> Runnable:
    """
    Creates a Retrieval-Augmented Generation (RAG) chain for answering questions with retrieved context.

    This function performs the following steps:
    1. Initializes an LLM and the vector store for retrieving documents.
    2. Sets up a retriever that combines a vector store and a document store, using similarity search.
    3. Creates a history-aware retriever that uses the LLM to reformulate questions.
    4. Combines the history-aware retriever with the question-answering system to create the final RAG chain.

    Returns:
        Runnable: A RAG chain that can be used for answering questions with retrieved context.
    """
    llm = get_llm()

    vectorstore = get_vectorstore()
    docstore = get_sqlitestore(DOCSTORE_PATH, DOCSTORE_TABLE_NAME)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=PARENT_DOC_ID,
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    contextualize_q_prompt = get_contextualize_question_prompt()

    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = get_question_answering_prompt()

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

def interactive_query_loop() -> None:
    """
    Runs an interactive loop to accept queries from the user until 'exit' or 'q' is entered.
    Clear previous chat messages by entering 'reset' or 'r'
    View chat history by entering 'change_history', 'ch' or 'history'

    The user is prompted to enter queries, which are then processed to retrieve and display relevant information based on previous chat messages.
    """
    rag_chain = get_rag_chain()

    chat_history: list[BaseMessage] = []
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in {"exit", "q"}:
            break
        if query.lower() in {"reset", "r"}:
            chat_history = []
            print(chr(27) + "[2J") # Clear terminal
            print("Chat history has been reset")
            continue
        if query.lower() in {"chat_history", "ch", "history"}:
            messages = "\n".join([f"{'You' if type(message) == HumanMessage else 'AI'}: \"{message.content}\"" for message in chat_history])
            print(chr(27) + "[2J") # Clear terminal
            print(f"Chat history:\n\033[92m{messages}\033[0m\n")
            continue
        if query:
            result = rag_chain.invoke({"input": query, "chat_history": chat_history})

            print(result['answer'])

            chat_history.append(HumanMessage(content=query))
            chat_history.append(SystemMessage(content=result["answer"]))


def get_llm():
    return Ollama(model=OLLAMA_MODEL)


if __name__ == "__main__":
    main()
