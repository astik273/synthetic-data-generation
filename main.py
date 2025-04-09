import argparse
from repo_manager import RepoManager
from repo_filter import RepoFilter
from code_analyzer import CodeAnalyzer
from query_generator import QueryGenerator
from dataset_builder import DatasetBuilder

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from GitHub repositories.")
    parser.add_argument("--repo_list", required=True, help="Path to the text file containing repository URLs.")
    parser.add_argument("--top_files", type=int, default=50, help="Number of top files to select per repository.")
    args = parser.parse_args()

    with open(args.repo_list, 'r') as f:
        repo_urls = [line.strip() for line in f if line.strip()]

    repo_manager = RepoManager()
    repo_filter = RepoFilter()
    code_analyzer = CodeAnalyzer()
    query_generator = QueryGenerator()
    dataset_builder = DatasetBuilder()

    repo_paths = repo_manager.clone_repos(repo_urls)
    filtered_repos = repo_filter.filter_repos(repo_paths)

    for repo_path in filtered_repos:
        top_files = code_analyzer.analyze_repo(repo_path, top_n=args.top_files)
        for file_path in top_files:
            entities = code_analyzer.extract_entities(file_path)
            for entity in entities:
                queries = query_generator.generate_queries(entity)
                answers = code_analyzer.get_related_answers(entity)
                dataset_builder.add_entry(repo_path, queries, answers)

    dataset_builder.save_to_csv("dataset.csv")

if __name__ == "__main__":
    main()