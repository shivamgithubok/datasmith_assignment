import requests

# Test text input
response = requests.post("http://localhost:8000/process", data={'query': 'summarize', 'text': 'This is a test text for summarization.'})
print(response.json())

# For file, need to upload, but for test, assume.

print("Test completed.")