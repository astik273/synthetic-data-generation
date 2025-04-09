import os
from git import Repo

class RepoManager:
    def __init__(self, clone_dir="repos"):
        self.clone_dir = clone_dir
        if not os.path.exists(clone_dir):
            os.makedirs(clone_dir)

    def clone_repos(self, repo_urls):
        repo_paths = []
        for url in repo_urls:
            repo_name = url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(self.clone_dir, repo_name)
            try:
                Repo.clone_from(url, repo_path, depth=1)
                repo_paths.append(repo_path)
            except Exception as e:
                print(f"Failed to clone {url}: {e}")
        return repo_paths

    def get_commit_hash(self, repo_path):
        repo = Repo(repo_path)
        return repo.head.commit.hexsha