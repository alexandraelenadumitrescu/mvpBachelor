import os
import pickle
import tempfile

import requests
from deepface import DeepFace

MOCK_EMPLOYEES = [
    {"name": "Alice", "email": "alice@company.com", "photo_url": "https://randomuser.me/api/portraits/women/1.jpg"},
    {"name": "Bob",   "email": "bob@company.com",   "photo_url": "https://randomuser.me/api/portraits/men/2.jpg"},
    {"name": "Carol", "email": "carol@company.com", "photo_url": "https://randomuser.me/api/portraits/women/3.jpg"},
]


def build_db(employees: list[dict]) -> dict[str, list[float]]:
    """Download each employee's photo and compute a FaceNet embedding."""
    db = {}
    for emp in employees:
        name, email, url = emp["name"], emp["email"], emp["photo_url"]
        print(f"  Processing {name} ({email})...")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Write to a temp file — DeepFace needs a file path, not bytes
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = tmp.name

            try:
                result = DeepFace.represent(img_path=tmp_path, model_name="Facenet")
                db[email] = result[0]["embedding"]
                print(f"    ✓ embedding computed ({len(db[email])} dims)")
            finally:
                os.unlink(tmp_path)

        except Exception as exc:
            print(f"    ✗ skipped: {exc}")

    return db


def save_db(db: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(db, f)
    print(f"DB saved → {path} ({len(db)} entries)")


def load_db(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
