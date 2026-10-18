"""Download DDXPlus (English) from figshare, check every file's MD5, unzip the patient files.

    python get_data.py

DDXPlus: Fansi Tchango, Goel, Wen, Martel, Ghosn. "DDXPlus: A New Dataset For Automatic Medical Diagnosis".
NeurIPS 2022 Datasets and Benchmarks. arXiv:2205.09148. https://github.com/mila-iqia/ddxplus
Licence: CC BY 4.0. The patients are SYNTHETIC.
"""
import hashlib
import json
import os
import urllib.request
import zipfile

ROOT = os.environ.get("TRIAGE_ROOT") or os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data_raw", "ddxplus_en")
UNZ = os.path.join(ROOT, "data_raw", "ddxplus_en_unzipped")
ARTICLE = "https://api.figshare.com/v2/articles/22687585"     # DDXPlus Dataset (English)
MD5 = {"release_evidences.json": "0660962abe0bbf99e1fb77188c253597",
       "release_conditions.json": "a992eadcff3099dd72e1820fbd2115aa",
       "release_test_patients.zip": "6fd53d331cf19ed4439bf4f46b92ee94",
       "release_validate_patients.zip": "e4db88e9082f18410de6fd0cb7e74d18",
       "release_train_patients.zip": "f75a78420ccb061951daa4db2d62139b"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


os.makedirs(RAW, exist_ok=True)
files = {f["name"]: f["download_url"] for f in json.load(urllib.request.urlopen(ARTICLE))["files"]}
for name, want in MD5.items():
    path = os.path.join(RAW, name)
    if not (os.path.exists(path) and md5(path) == want):
        print("downloading", name)
        urllib.request.urlretrieve(files[name], path)
    got = md5(path)
    assert got == want, f"{name}: md5 {got}, expected {want}"
    print("ok", name)
    if name.endswith(".zip"):
        zipfile.ZipFile(path).extractall(UNZ)
print("DDXPlus ready in", RAW, "and", UNZ)
