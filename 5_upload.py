"""Step 5 (optional): Push splits to your HF dataset repo."""
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo


load_dotenv()
username = os.getenv("HF_USERNAME")
repo_name = os.getenv("HF_DATASET_REPO", "ai-reviews-augmented")
token = os.getenv("HF_TOKEN")
assert username and token, "Set HF_USERNAME and HF_TOKEN in .env"

repo_id = f"{username}/{repo_name}"
api = HfApi(token=token)

try:
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
except Exception as e:
    print(f"Repo note: {e}")

api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["*.parquet"],
    token=token,
)
print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")
