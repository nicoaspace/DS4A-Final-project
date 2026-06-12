"""
Deploy DS4A Loan Predictor to Hugging Face Spaces.

Usage:
    python deploy_to_hf.py --token hf_xxxYOURTOKENxxx
    python deploy_to_hf.py --token hf_xxxYOURTOKENxxx --force

Get your token at: https://huggingface.co/settings/tokens
(needs write access)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

SPACE_ID = "sazkicher/ds4a-loan-predictor"
REPO_DIR = Path(__file__).parent

FILES_TO_UPLOAD = [
    "app.py",
    "requirements.txt",
    "README.md",
    "docs/images/confusion_matrix.png",
    "docs/images/feature_importance.png",
    "docs/images/distributions.png",
    "docs/images/default_rates.png",
]


def deploy(token: str, force: bool = False) -> None:
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

    print("Uploading files in a single commit…")
    upload_folder(
        folder_path=str(REPO_DIR),
        repo_id=SPACE_ID,
        repo_type="space",
        token=token,
        commit_message="Deploy DS4A loan predictor app",
        allow_patterns=FILES_TO_UPLOAD,
    )
    for rel_path in FILES_TO_UPLOAD:
        print(f"  ✅  {rel_path}")

    if force:
        print("Forcing Space restart…")
        api.restart_space(repo_id=SPACE_ID, token=token)

    print(f"\n🚀 Done! Your Space is live at:\n   https://huggingface.co/spaces/{SPACE_ID}\n")
    print("Wait ~2 minutes for the build to finish.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to Hugging Face Spaces")
    parser.add_argument("--token", required=True, help="HF write token")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Restart the Space after upload",
    )
    args = parser.parse_args()
    deploy(args.token, force=args.force)
