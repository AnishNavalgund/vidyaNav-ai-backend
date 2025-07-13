import requests

with open("data/book.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/generate-worksheet/",
        files={"file": ("data/book.png", f)},
        data={"grades": "1,2,3", "language": "English"}
    )

print("Status Code:", response.status_code)
print("Raw Response:", response.text)

try:
    print("JSON:", response.json())
except Exception as e:
    print("Failed to parse JSON:", e)
