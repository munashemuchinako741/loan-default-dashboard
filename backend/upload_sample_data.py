import requests

url = "http://localhost:8000/predict/"
file_path = "backend/uploads/sample_prediction_data.csv"

with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "text/csv")}
    response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
