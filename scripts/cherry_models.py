import json
from pathlib import Path

import requests
import yaml

# Load configuration from fastagent.config.yaml
config_path = Path(__file__).parent.parent / "fastagent.config.yaml"
if not config_path.exists():
    print(f"Error: Configuration file not found at {config_path}")
    exit(1)

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Get base_url and api_key from generic section
generic_config = config.get("generic", {})
base_url = generic_config.get("base_url", "http://localhost:23333/v1")
api_key = generic_config.get("api_key", "")

if not api_key:
    print("Error: api_key not found in configuration file")
    exit(1)

# Construct URL and headers
url = f"{base_url.rstrip('/')}/models"
headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get(url, headers=headers)

# Check if response is successful and valid JSON
if response.status_code == 200:
    try:
        data = response.json()
        # Print model IDs
        if "data" in data:
            for model in data["data"]:
                print(model.get("id", ""))
        # Print total
        if "total" in data:
            print(f"\nTotal: {data['total']}")
    except json.JSONDecodeError:
        print("Response is not valid JSON. Content:")
        print(response.text)
else:
    print(f"Request failed with status {response.status_code}")
    print(f"Response content: {response.text}")
