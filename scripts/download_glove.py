"""Download GloVe embeddings (300d, 6B tokens)."""
import os
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = os.path.join("data", "glove")
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.300d.txt")


def main():
    if os.path.exists(GLOVE_FILE):
        print(f"GloVe already exists at {GLOVE_FILE}")
        return

    os.makedirs(GLOVE_DIR, exist_ok=True)
    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")

    print(f"Downloading GloVe from {GLOVE_URL}...")
    print("This is ~862MB and may take a few minutes.")
    urllib.request.urlretrieve(GLOVE_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("glove.6B.300d.txt", GLOVE_DIR)

    os.remove(zip_path)
    print(f"Done. GloVe saved to {GLOVE_FILE}")


if __name__ == "__main__":
    main()
