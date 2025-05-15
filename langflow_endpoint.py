import json
import os

import requests


def query_langflow(input_value):
    # Load url from config file
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.json')
    with open(config_path, 'r') as fd:
        config = json.load(fd)

    url = config['langflow_url']

    # Request payload configuration
    payload = {
        "input_value": input_value,  # The input value to be processed by the flow
        "output_type": "chat",  # Specifies the expected output format
        "input_type": "chat"  # Specifies the input format
    }

    # Request headers
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        # Print response
        # print(response.text)
        response_json = response.json()  # Parse the JSON response
        return response_json["outputs"][0]["outputs"][0]["results"]["message"]["text"]  # Return the text

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")


if __name__ == "__main__":
    # Example usage
    input_value = "What is the capital of France?"
    response = query_langflow(input_value)
    print(response)  # Print the response from the API