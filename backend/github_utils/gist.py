from .auth import get_gist_token
from dataclasses import dataclass
import logging
from typing import Any, Optional
from datetime import datetime
import json
import requests

logger = logging.getLogger(__name__)


@dataclass
class Gist:
    raw_url: str
    gist_url: str
    gist_id: str
    filename: str


def create_gist(
    data: Any, filename: str = None, description: str = None
) -> Optional[Gist]:
    github_token = get_gist_token()

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    json_content = json.dumps(data, indent=2)

    gist_data = {
        "description": description or "Temporary JSON storage",
        "public": True,
        "files": {filename: {"content": json_content}},
    }

    headers = {"Accept": "application/vnd.github.v3+json"}

    headers["Authorization"] = f"token {github_token}"

    response = requests.post(
        "https://api.github.com/gists", headers=headers, json=gist_data
    )

    if response.status_code == 201:
        gist_info = response.json()
        raw_url = gist_info["files"][filename]["raw_url"]
        return Gist(raw_url, gist_info["html_url"], gist_info["id"], filename)

    else:
        logger.error(f"Failed to create gist: {response.status_code} - {response.text}")
        return None


def update_gist(
    gist_id: str, data: Any, filename: str = None, description: str = None
) -> Optional[Gist]:
    github_token = get_gist_token()

    # First, get the existing gist to find the filename if not provided
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }

    # Get existing gist details
    get_response = requests.get(
        f"https://api.github.com/gists/{gist_id}", headers=headers
    )

    if get_response.status_code != 200:
        return {
            "success": False,
            "error": f"Failed to retrieve gist: {get_response.status_code} - {get_response.text}",
        }

    existing_gist = get_response.json()

    # If filename not provided, use the first JSON file from the gist
    if filename is None:
        json_files = [f for f in existing_gist["files"].keys() if f.endswith(".json")]
        if json_files:
            filename = json_files[0]
        else:
            # If no JSON files exist, create a new filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data_{timestamp}.json"

    if not filename.endswith(".json"):
        filename += ".json"

    json_content = json.dumps(data, indent=2)

    # Prepare update data
    update_data = {"files": {filename: {"content": json_content}}}

    # Add description if provided
    if description is not None:
        update_data["description"] = description

    # Update the gist
    response = requests.patch(
        f"https://api.github.com/gists/{gist_id}", headers=headers, json=update_data
    )

    if response.status_code == 200:
        gist_info = response.json()
        raw_url = gist_info["files"][filename]["raw_url"]
        return Gist(raw_url, gist_info["html_url"], gist_info["id"], filename)

    else:
        logger.error(f"Failed to update gist: {response.status_code} - {response.text}")
        return None


def load_gist(raw_url: str) -> Any:
    response = requests.get(raw_url)
    return response.json()
