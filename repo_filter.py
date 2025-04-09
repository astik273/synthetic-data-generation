import os
from git import Repo
from utils import count_commits_in_last_month, get_star_count, analyze_readme

class RepoFilter:
    def __init__(self, min_stars=1000, min_commits=10):
        self.min_stars = min_stars
        self.min_commits = min_commits

    def filter_repos(self, repo_paths):
        filtered = []
        for repo_path in repo_paths:
            repo = Repo(repo_path)
            star_count = get_star_count(repo_path)
            if star_count < self.min_stars:
                continue
            commit_count = count_commits_in_last_month(repo)
            if commit_count < self.min_commits:
                continue
            if not self.is_library_or_framework(repo_path):
                continue
            filtered.append(repo_path)
        return filtered

    def is_library_or_framework(self, repo_path):
        readme_path = os.path.join(repo_path, "README.md")
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return analyze_readme(content)
        return False
