import json
import os
import urllib.request

# Get dataset2 from i2Nav-WHU/KF-GINS-Matlab repo

REPO = "i2Nav-WHU/KF-GINS-Matlab"
FOLDER = "dataset2"
BRANCH = "main"
OUTPUT_DIR = "../data/KF_GINS"

def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def fetch_bytes(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def download_folder(repo, folder, branch, output_dir):
    url = f"https://api.github.com/repos/{repo}/contents/{folder}?ref={branch}"
    entries = fetch_json(url)

    for entry in entries:
        if entry["type"] == "dir":
            download_folder(repo, entry["path"], branch, output_dir)
        elif entry["type"] == "file":
            local_path = os.path.join(output_dir, entry["name"])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading {entry['path']}")
            data = fetch_bytes(entry["download_url"])
            with open(local_path, "wb") as f:
                f.write(data)


download_folder(REPO, FOLDER, BRANCH, OUTPUT_DIR)
