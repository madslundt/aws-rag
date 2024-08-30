import json
from typing import Any, Generic, Optional, TypeVar

V = TypeVar("V")

def load_json_file(file_path: str) -> Optional[dict[str, Generic[V]]]:
    """
    Load and parse a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Optional[dict[str, V]]: Parsed JSON data as a dictionary, or None if an error occurs.
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        print(f"JSON decode error: {str(e)}")
    return None
