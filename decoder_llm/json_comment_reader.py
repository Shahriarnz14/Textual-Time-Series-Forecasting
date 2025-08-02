import json
import re

def read_json_with_comments(file_path):
    """Reads a JSON file that may contain // comments and returns the parsed JSON.

    Args:
        file_path: The path to the JSON file.

    Returns:
        A Python dictionary or list representing the parsed JSON data, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove single-line comments (//...)
        content = re.sub(r'//.*', '', content)

        # Remove multi-line comments (/* ... */)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Parse the cleaned JSON
        data = json.loads(content)
        return data

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None