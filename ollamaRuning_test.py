import requests

res = requests.get("http://localhost:11434")
print(res.status_code)
print(res.text)