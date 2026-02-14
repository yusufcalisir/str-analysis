import urllib.request
import json
import sys

url = "http://localhost:8000/profile/analyze"
data = {
    "profile_id": "test-profile-eu",
    "population": "European"
}

req = urllib.request.Request(
    url,
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    print(f"Sending POST request to {url}...")
    with urllib.request.urlopen(req) as response:
        print(f"Status: {response.status}")
        # print(response.read().decode('utf-8')[:500]) # truncated
        print("Success")
except urllib.error.HTTPError as e:
    print(f"HTTPError: {e.code}")
    print(e.read().decode('utf-8'))
except urllib.error.URLError as e:
    print(f"URLError: {e.reason}")
except Exception as e:
    print(f"Error: {e}")
