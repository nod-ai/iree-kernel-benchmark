import json
import requests
from datetime import datetime
import tempfile
import os
from backend.storage.auth import get_gist_token


def create_gist(data, filename=None, description=None, github_token=None):
    """
    Upload JSON data to GitHub Gist and return the raw URL.

    Args:
        data: Python object to be converted to JSON
        filename: Name for the JSON file (default: generated timestamp)
        description: Gist description (default: "Temporary JSON storage")
        github_token: GitHub personal access token (optional, but recommended)

    Returns:
        dict: Contains 'raw_url', 'gist_url', and 'gist_id'
    """
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.json"

    # Ensure filename ends with .json
    if not filename.endswith(".json"):
        filename += ".json"

    # Convert data to JSON string
    json_content = json.dumps(data, indent=2)

    # Prepare the gist data
    gist_data = {
        "description": description or "Temporary JSON storage",
        "public": True,
        "files": {filename: {"content": json_content}},
    }

    # Headers
    headers = {"Accept": "application/vnd.github.v3+json"}

    # Add authorization header if token provided
    if github_token:
        headers["Authorization"] = f"token {github_token}"

    # Create the gist
    response = requests.post(
        "https://api.github.com/gists", headers=headers, json=gist_data
    )

    if response.status_code == 201:
        gist_info = response.json()
        raw_url = gist_info["files"][filename]["raw_url"]

        return {
            "raw_url": raw_url,
            "gist_url": gist_info["html_url"],
            "gist_id": gist_info["id"],
            "success": True,
        }
    else:
        return {
            "success": False,
            "error": f"Failed to create gist: {response.status_code} - {response.text}",
        }


# Example usage:
if __name__ == "__main__":
    # Sample data
    sample_data = {
        "name": "John Doe",
        "age": 30,
        "city": "New York",
        "hobbies": ["reading", "gaming", "hiking"],
    }

    # Using GitHub Gist (persistent until manually deleted)
    print("=== GitHub Gist Method ===")
    result = create_gist(
        data=sample_data,
        filename="my_data.json",
        description="Test JSON upload",
        github_token=get_gist_token(),
    )

    if result["success"]:
        print(f"Raw URL: {result['raw_url']}")
        print(f"Gist URL: {result['gist_url']}")

        # Verify by downloading
        response = requests.get(result["raw_url"])
        print(f"Retrieved data: {response.json()}")
    else:
        print(f"Error: {result['error']}")
