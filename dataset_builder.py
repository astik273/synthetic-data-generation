import pandas as pd
from repo_manager import RepoManager

class DatasetBuilder:
    def __init__(self):
        self.data = []

    def add_entry(self, repo_path, queries, answers):
        commit_hash = RepoManager().get_commit_hash(repo_path)
        for query in queries:
            for answer in answers:
                self.data.append({
                    "github_url": repo_path,
                    "commit": commit_hash,
                    "query": query,
                    "answer": answer
                })

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)