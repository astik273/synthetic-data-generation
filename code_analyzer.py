import os
import logging
from utils import count_python_files, get_file_lines, parse_definitions

# Configure logging to track code analysis
logging.basicConfig(level=logging.INFO)

class CodeAnalyzer:
    def __init__(self):
        """Initializes the CodeAnalyzer."""
        pass

    def analyze_repo(self, repo_path, top_n=50):
        """
        Analyzes the repository and returns the top N Python files based on line count.

        Args:
            repo_path (str): Path to the repository.
            top_n (int): Number of top files to return (default is 50).

        Returns:
            list: List of paths to the top N Python files.
        """
        try:
            python_files = []
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        line_count = get_file_lines(file_path)
                        python_files.append((file_path, line_count))
            
            # Sort by line count in descending order and take top N
            python_files.sort(key=lambda x: x[1], reverse=True)
            top_files = [file[0] for file in python_files[:top_n]]
            logging.info(f"Found {len(top_files)} top Python files in {repo_path}")
            return top_files
        except Exception as e:
            logging.error(f"Error analyzing repository {repo_path}: {e}")
            return []

    def extract_entities(self, file_path):
        """
        Extracts class and function definitions from a Python file.

        Args:
            file_path (str): Path to the Python file.

        Returns:
            list: List of entity names (classes and functions).
        """
        try:
            entities = parse_definitions(file_path)
            logging.info(f"Extracted {len(entities)} entities from {file_path}")
            return entities
        except Exception as e:
            logging.error(f"Error extracting entities from {file_path}: {e}")
            return []

    def get_related_answers(self, entity, repo_path):
        """
        Finds files related to the given entity within the repository.

        Args:
            entity (str): Name of the entity (class or function).
            repo_path (str): Path to the repository.

        Returns:
            list: List of file paths related to the entity.
        """
        try:
            related_files = []
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if entity in content:
                            related_files.append(file_path)
            logging.info(f"Found {len(related_files)} related files for entity {entity} in {repo_path}")
            return related_files
        except Exception as e:
            logging.error(f"Error finding related files for {entity} in {repo_path}: {e}")
            return []