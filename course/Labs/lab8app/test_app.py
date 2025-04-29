import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample wine features (13 features as per the wine dataset)
sample_features = [
    14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065
]

# Prepare the request data
data = {
    "features": sample_features
}

# Make the request
response = requests.post(url, json=data)

# Print the response
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))
