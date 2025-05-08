import time
from dotenv import load_dotenv
from utils.github_utils import create_github_repo, commit_and_push_changes

load_dotenv()

class DeploymentAgent:
    def __init__(self):
        ts = int(time.time())
        self.repo_name = f"xalt-chatbot-repo-{ts}"
        self.github_repo_url = None

    def deploy_now(self) -> str:
        # 1️⃣ Create GitHub repo
        print("[1/2] Creating GitHub repository…")
        self.github_repo_url = create_github_repo(self.repo_name)
        print(f"[✅] GitHub repo created: {self.github_repo_url}")

        # 2️⃣ Push code
        print("[2/2] Pushing project files to GitHub…")
        commit_and_push_changes(self.github_repo_url, self.repo_name)
        print("[✅] Code pushed to GitHub")

        return self.github_repo_url
