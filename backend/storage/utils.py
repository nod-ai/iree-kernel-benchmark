from pathlib import Path
import os
from typing import Union, Any


def convert_dict_case(data: dict[Any, Any], to_camel=True):
    def snake_to_camel(snake_str: str):
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    def camel_to_snake(camel_str: str):
        result = []
        for i, char in enumerate(camel_str):
            if char.isupper() and i > 0:
                result.append("_")
                result.append(char.lower())
            else:
                result.append(char.lower())
        return "".join(result)

    converter = snake_to_camel if to_camel else camel_to_snake

    return {converter(k) if isinstance(k, str) else k: v for k, v in data.items()}


def get_nested_files(local_path: Union[os.PathLike, str], file_type: str) -> list[Path]:
    """
    Returns a list of Path objects for all files with the given extension in a directory tree.

    Args:
        local_path: The root directory path to search in
        file_type: The file extension to search for (e.g., '.txt', 'txt', '*.txt')

    Returns:
        A list of Path objects for all matching files
    """
    root_path = Path(local_path)

    if not root_path.exists():
        raise ValueError(f"Path does not exist: {local_path}")
    if not root_path.is_dir():
        raise ValueError(f"Path is not a directory: {local_path}")

    if not file_type.startswith("."):
        file_type = "." + file_type
    if file_type.startswith("*."):
        file_type = file_type[1:]

    pattern = f"**/*{file_type}"
    matching_files = list(root_path.glob(pattern))

    return matching_files
