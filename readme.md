# AWS RAG

## Overview

This project consists of four Python scripts designed for different functionalities:

1. **Download documents**: Downloads and manages documents from remote sources based on a provided configuration file.
2. **Database and Document Processing**: Handles document processing, including splitting and adding documents to a database.
3. **Query and Retrieval System**: Manages querying, interactive loops, and retrieval of relevant documents using a multi-vector retrieval approach.
4. **Chat and Retrieval System**: Manages chatting, interactive loops, and retrieval of relevant documents using a multi-vector retrieval approach.

These scripts work together to manage document processing, querying, and downloading, creating a cohesive system for document management and information retrieval.

## Table of Contents

1. [Download documents](#download-documents)
2. [Database and Document Processing](#database-and-document-processing)
3. [Query and Retrieval System](#query-and-retrieval-system)
3. [Chat and Retrieval System](#chat-and-retrieval-system)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)
6. [Configuration](#configuration)

---

## Download documents

### Description

This script manages the downloading of documents from remote sources. It supports checking if the remote file has been updated before downloading and handles concurrent downloads using threading.

### Key Functions

- **`main(file_path: str, download_path: str) -> None`**: The entry point for the document download process.
- **`load_json_file(file_path: str) -> Optional[dict]`**: Loads and parses a JSON configuration file.
- **`download_file(...) -> None`**: Downloads a file from a specified URL to a local path.
- **`is_remote_file_updated(...) -> bool`**: Checks if a remote file is newer or different in size compared to the local file.
- **`download_docs(...) -> None`**: Manages downloading multiple documents based on configuration settings.

---

## Database and Document Processing

### Description

This script handles the processing of documents, including splitting them into chunks and managing document hashes. The documents are then stored in a vectorstore and bytestore for efficient retrieval.

### Key Functions

- **`main(path: str) -> None`**: Handles database reset and document processing.
- **`add_document_hash(file_path: str) -> None`**: Adds a hash of the document to the database to track changes.
- **`load_documents(path: str) -> list[Document]`**: Loads PDF documents from the specified path.
- **`split_documents(...)`**: Splits documents into parent and child chunks based on specified sizes.
- **`add_documents_to_store(...)`**: Adds documents to vector and bytestore databases.
- **`clear_database() -> None`**: Clears all stored data in both vector and bytestore databases.

---

## Query and Retrieval System

### Description

This script provides functionality to perform queries against a stored set of documents using multi-vector retrieval. It allows for both direct queries and an interactive query loop. The retrieval process leverages self-querying and hybrid search methods, combining keyword-based searches with vector similarity to provide more accurate and relevant results.

### Key Features

- **Self-Querying**: Automatically generates alternative queries from the user's input to enhance retrieval performance.
- **Hybrid Search**: Utilizes both keyword-based and vector-based similarity searches to maximize the relevance of the retrieved documents.

### Key Functions

- **`main() -> None`**: Initializes the command-line interface for querying.
- **`interactive_query_loop() -> None`**: Provides an interactive loop for continuous user queries.
- **`query_rag(query_text: str) -> None`**: Main function that handles the entire query process, including generating alternative questions, retrieving documents, and generating responses using self-querying and hybrid search techniques.
- **`generate_response(...) -> str`**: Generates a response based on the context of relevant documents.
- **`retrieve_relevant_docs(...) -> tuple[list, list]`**: Retrieves relevant documents based on provided questions and sources.

---

## Chat and Retrieval System

### Description

This script provides functionality to chat with a stored set of documents using multi-vector retrieval. The retrieval process leverages self-querying and previous chat messages, using vector similarity to provide more accurate and relevant results.

### Key Features

- **Self-Querying**: Automatically generates alternative queries from the user's input to enhance retrieval performance.

### Key Functions

- **`main() -> None`**: Initializes the command-line interface for querying.
- **`interactive_query_loop() -> None`**: Provides an interactive loop for continuous user chats.

---

## Setup and Installation

1. **Clone the Repository**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/madslundt/aws-rag.git
   cd aws-rag
   ```

2. **Install Dependencies**: Use `pip` or `pipenv` to install the necessary packages.
   ```bash
   pipenv install
   ```

3. **Activate the Virtual Environment**: If you are using `pipenv`, enter the virtual environment shell.
   ```bash
   pipenv shell
   ```

## Usage

### Running the Scripts

1. **Download documents**:
   ```bash
   python download_docs.py
   ```

2. **Database and Document Processing**:
   ```bash
   python populate_database.py [--reset]
   ```

3. **Query or chat**:
   Query with the RAG in interactive mode:
   ```bash
   python query_rag.py
   ```

   Chat with the RAG in interactive mode:
   ```bash
   python chat_rag.py
   ```

### Command-Line Options

- **`--reset`**: Clears the existing database before processing new documents.
- **`--query_text`**: Directly queries the system without entering interactive mode.

In interactive mode the following keywords can be used:
- **`q` or `exit`**: Terminate interactive mode
- **`r` or `reset`**: Resets the chat history
- **`ch` or `history` or `chat_history`**: Shows the chat history

## Configuration

- **`CONFIG_PATH`**: Path to the configuration JSON file.
- **`DOCSTORE_PATH`**: Directory path for storing document databases.
- **`DOCUMENTS_PATH`**: Directory path where documents are downloaded and stored.
- **`OLLAMA_MODEL`**: Specifies the language model to be used. The default is `llama3.1`.
- **`EMBEDDING_MODEL`**: Specifies the embedding model to be used. The default is `nomic-embed-text` via Ollama.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
