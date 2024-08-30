from env import OLLAMA_MODEL
from query_rag import query_rag
from langchain_community.llms.ollama import Ollama

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


def query_and_validate(question: str, expected_response: str) -> bool:
    """
    Queries the RAG system with a given question and compares the actual response with the expected response.

    Args:
        question (str): The question to query.
        expected_response (str): The expected response for validation.

    Returns:
        bool: True if actual response matches the expected response, otherwise False.
    """
    try:
        response_text = query_rag(question).strip().lower()
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
