#!/usr/bin/env python3
"""Upload local truthfulqaPro/ (CSVs + pair_ids) and the Hub dataset card to the target HF dataset (default foadnamjoo/TruthfulQAPro).

Requires:
  pip install -r requirements-hub.txt

Authenticate (run alone; no trailing text on the same line):
  hf auth login

Check the active account (must own the repo namespace, e.g. foadnamjoo for foadnamjoo/TruthfulQAPro):
  hf auth whoami

Use a personal write token for your Hub user. A Cursor- or tool-issued fine-grained token may not
have write access to your namespace. Or set HF_TOKEN to such a token.

When pasting commands, copy only the command line — not the shell prompt
(e.g. lines containing "user@hostname"), or zsh may report a parse error near "@".

Default target is your personal user namespace (foadnamjoo/…), not an org you belong to.
`hf auth whoami` listing orgs (e.g. PhillipsLab) does not change that. To publish under an org,
you must pass --repo-id OrgName/repo explicitly.

Usage:
  python scripts/push_truthfulqa_pro_to_hub.py
  python scripts/push_truthfulqa_pro_to_hub.py --repo-id YOUR_NAME/TruthfulQAPro
  python scripts/push_truthfulqa_pro_to_hub.py --no-create-repo
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_DATASET_REPO", "foadnamjoo/TruthfulQAPro"),
        help=(
            "Dataset repo id USER_OR_ORG/name (default: HF_DATASET_REPO env or foadnamjoo/TruthfulQAPro). "
            "Uses your personal user by default, not an org membership."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Skip create_repo (use after creating an empty dataset on huggingface.co)",
    )
    args = parser.parse_args()
    root = args.root or Path(__file__).resolve().parents[1]

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        print("Install huggingface_hub: pip install -r requirements-hub.txt", file=sys.stderr)
        sys.exit(1)

    card = root / "hf_datasets" / "truthfulqaPro" / "README.md"
    data_dir = root / "truthfulqaPro"
    if not card.is_file():
        print(f"Missing dataset card: {card}", file=sys.stderr)
        sys.exit(1)
    if not data_dir.is_dir():
        print(f"Missing data directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    api = HfApi()
    if not args.no_create_repo:
        repo_exists = False
        try:
            api.repo_info(repo_id=args.repo_id, repo_type="dataset")
            repo_exists = True
        except RepositoryNotFoundError:
            pass
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 401:
                print("Not authenticated for this repo. Run: hf auth login", file=sys.stderr)
            raise

        if not repo_exists:
            try:
                api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
            except HfHubHTTPError as e:
                if getattr(e.response, "status_code", None) == 403:
                    owner, _, name = args.repo_id.partition("/")
                    if not name:
                        name = args.repo_id
                        owner = "(your Hub user)"
                    print(
                        f"Cannot create dataset {args.repo_id} via API (403 — token often lacks "
                        "'create repo' on fine-grained tokens).\n\n"
                        "Create an empty dataset once in the browser. Pick your personal account as "
                        f"owner (not an org unless you want {args.repo_id} under that org):\n"
                        "  https://huggingface.co/new-dataset\n"
                        f"  Owner: {owner}    Name: {name}\n\n"
                        "Then run:\n"
                        "  python scripts/push_truthfulqa_pro_to_hub.py --no-create-repo\n\n"
                        "Or use a classic/write token and re-run without --no-create-repo.\n"
                        "Fine-grained tokens need repository write access to this dataset after it exists.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                raise

    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    for csv_path in sorted(data_dir.glob("*.csv")):
        api.upload_file(
            path_or_fileobj=str(csv_path),
            path_in_repo=csv_path.name,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    pair_dir = data_dir / "pair_ids"
    for json_path in sorted(pair_dir.glob("*.json")):
        api.upload_file(
            path_or_fileobj=str(json_path),
            path_in_repo=f"pair_ids/{json_path.name}",
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"Uploaded dataset card + CSVs + pair_ids to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
