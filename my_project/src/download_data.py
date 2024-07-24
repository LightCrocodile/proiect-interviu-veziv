# src/download_data.py
import os
import requests
import zipfile

def download_data(url, extract_to='.'):
    local_filename = url.split('=')[-1] + '.zip'  # Assign a valid filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(local_filename)

if __name__ == "__main__":
    download_data('https://drive.google.com/uc?export=download&id=1CEx_TnzAlTMEGqqq5O2QXKoz8fPOUjW8', './data')
