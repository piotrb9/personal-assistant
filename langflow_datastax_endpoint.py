import json
import requests
import os


def query_hosted_langflow(input_value):
    # Load config values (you can store the token and URL here)
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    with open(config_path, 'r') as fd:
        config = json.load(fd)

    url = config['langflow_url']  # Hosted Langflow URL from config
    token = config['langflow_token']  # Application token from config

    payload = {
        "input_value": input_value,
        "output_type": "chat",
        "input_type": "chat"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    try:
        # Send API request
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        # Parse and return AI-generated text
        response_json = response.json()
        return response_json["outputs"][0]["outputs"][0]["results"]["message"]["text"]

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except (ValueError, KeyError) as e:
        print(f"Error parsing response: {e}")


if __name__ == "__main__":
    input_value = "Tell me a joke about cats"
    result = query_hosted_langflow(input_value)
    if result:
        print(result)
