"""
Deploy DS4A Loan Predictor to Hugging Face Spaces.

Usage:
    python deploy_to_hf.py --token hf_xxxYOURTOKENxxx

Get your token at: https://huggingface.co/settings/tokens
(needs write access)
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

SPACE_ID   = "sazkicher/ds4a-loan-predictor"
SPACE_NAME = "DS4A Loan Payment Predictor"
REPO_DIR   = Path(__file__).parent

FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "README.md",
    "docs/images/confusion_matrix.png",
    "docs/images/feature_importance.png",
    "docs/images/distributions.png",
    "docs/images/default_rates.png",
]

def deploy(token: str) -> None:
    api = HfApi(token=token)

    print(f"Creating / updating Space: {SPACE_ID}")
    create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        token=token,
        private=False,
    )

    print("Uploading files…")
    for rel_path in FILES_TO_UPLOAD:
        local = REPO_DIR / rel_path
        if not local.exists():
            print(f"  ⚠️  Skipping (not found): {rel_path}")
            continue
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=rel_path,
            repo_id=SPACE_ID,
            repo_type="space",
            token=token,
            commit_message=f"Upload {rel_path}",
        )
        print(f"  ✅  {rel_path}")

    print(f"\n🚀 Done! Your Space is live at:\n   https://huggingface.co/spaces/{SPACE_ID}\n")
    print("It may take ~2 minutes to build on first deploy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--token", required=True, help="HF write token")
    args = parser.parse_args()
    deploy(args.token)
