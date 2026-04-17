from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()
REPO_ID = "foadnamjoo/TruthfulQA-Audited"
REPO_TYPE = "dataset"

# Create the repo
api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE,
                exist_ok=True, private=False)

# Upload all files preserving subfolder structure
base = Path("data/subsets/TruthfulQA-Audited")
for f in sorted(base.rglob("*")):
    if f.is_file():
        hub_path = str(f.relative_to(base))
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=hub_path,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            commit_message=f"Add {hub_path}",
        )
        print(f"uploaded: {hub_path}")
