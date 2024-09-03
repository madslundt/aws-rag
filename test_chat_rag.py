from env import OLLAMA_MODEL
from chat_rag import get_rag_chain
from langchain_community.llms.ollama import Ollama
from langchain_core.messages import BaseMessage, HumanMessage

from utils import verbose_print

EVAL_PROMPT = """
Does the actual response match the expected response? (Answer with 'true' or 'false')
---
Expected Response: {expected_response}
Actual Response: {actual_response}
"""


def test_s3_limited_to_a_region():
    assert query_and_validate(
        question="Is S3 limited to a specific region? (answer with 'yes' or 'no')",
        expected_response="""No"""
    )

def test_s3_limited_with_chat_history():
    chat_history = [
        HumanMessage(content="Is Lambda limited to a specific region? (answer with 'yes' or 'no')"),
    ]
    assert query_and_validate(
        question="Is S3 also limited? (answer with 'yes' or 'no')",
        expected_response="""No""",
        chat_history=chat_history,
    )

def test_rds_postgres_support():
    assert query_and_validate(
        question="Does AWS RDS support Postgres? (answer with 'yes' or 'no')",
        expected_response="""Yes"""
    )

def test_lambda_runtime_supports_python():
    assert query_and_validate(
        question="Is Python supported in AWS Lambda runtime? (answer with 'yes' or 'no')",
        expected_response="""Yes"""
    )

def test_regions_include_denmark_as_incorrect():
    assert not query_and_validate(
        question="What regions are available for EC2?",
        expected_response="""Denmark"""
    )

def test_chat_with_chat_history():
    chat_history = [
        HumanMessage(content=f"Tell me about AWS RDS."),
    ]
    assert query_and_validate(
        question="What are we talking about?",
        expected_response="""AWS RDS""",
        chat_history=chat_history,
    )

def test_chat_with_missing_chat_history():
    assert not query_and_validate(
        question="What are we talking about?",
        expected_response="""AWS RDS""",
    )


def query_and_validate(question: str, expected_response: str, chat_history: list[BaseMessage] = []) -> bool:
    """
    Queries the RAG system with a given query and compares the actual response with the expected response.

    Args:
        question (str): The question to query.
        expected_response (str): The expected response for validation.
        chat_history (list[BaseMessage]): Chat history if any (default to empty list)

    Returns:
        bool: True if actual response matches the expected response, otherwise False.
    """
    try:
        rag_chain = get_rag_chain()
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        response_text = response['answer'].strip().lower()
        formatted_prompt = EVAL_PROMPT.format(
            expected_response=expected_response.strip().lower(), actual_response=response_text
        )

        model = Ollama(model=OLLAMA_MODEL)
        evaluation_result = model.invoke(formatted_prompt).strip().lower()

        verbose_print(formatted_prompt)
        return process_evaluation_result(evaluation_result)
    except Exception as e:
        verbose_print(f"Error during validation: {str(e)}")
        raise

def process_evaluation_result(evaluation_result: str) -> bool:
    """
    Processes the evaluation result string to determine if the actual response matches the expected response.

    Args:
        evaluation_result (str): The evaluation result string from the model.

    Returns:
        bool: True if the result indicates a match, False otherwise.
    """
    if "true" in evaluation_result:
        verbose_print("\033[92mResponse: True\033[0m")
        return True
    elif "false" in evaluation_result:
        verbose_print("\033[91mResponse: False\033[0m")
        return False
    else:
        raise ValueError("Invalid evaluation result. Expected 'true' or 'false'.")
