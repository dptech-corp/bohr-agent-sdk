import os
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import arxiv
import json
from typing import List
import time

PAPER_DIR = "output/papers"

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# Use model from environment or default to deepseek
model_type = os.getenv('MODEL', 'deepseek/deepseek-chat')


def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)

    Returns:
        List of paper IDs found in the search
    """
    time.sleep(600)
    # Use arxiv to find the papers
    client = arxiv.Client()
    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids

def extract_info(paper_id: str) -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."

def read_file(file_path: str):
    """
    Args:
        file_path: 文件路径
    Returns:
        The content of the file
    """
    with open(file_path, "r") as file:
        return file.read()

def create_agent(ak: str = None, app_key: str = None, project_id: int = None):
    """动态创建 agent - SDK 标准接口
    
    Args:
    
        ak: 可选的 AK 参数
        app_key: 可选的 app_key 参数
        project_id: 可选的 project_id 参数

    """
    
    agent_ak = ak
    agent_app_key = app_key
    agent_project_id = project_id
    
    def show_ak():
        """
        Show the AK
        Args:
            None
        Returns:
            The AK
        """
        if agent_ak:
            return f"My AK is: {agent_ak}"
        else:
            return "I don't have an AK (temporary user)"
    
    def show_app_key():
        """
        Show the app_key
        Args:
            None
        Returns:
            The app_key
        """
        if agent_app_key:
            return f"My app_key is: {agent_app_key}"
        else:
            return "I don't have an app_key (temporary user)"
        
    def show_project_id():
        """
        Show the project_id
        Args:
            None
        Returns:
            The project_id
        """
        if agent_project_id:
            return f"My project_id is: {agent_project_id}, type: {type(agent_project_id)}"
        else:
            return f"I don't have an project_id (temporary user), type: {type(agent_project_id)}"
    
    return Agent(
        name="mcp_sse_agent",
        model=LiteLlm(model=model_type),
        instruction="You are an intelligent assistant capable of using external tools.",
        tools=[show_ak, show_app_key, show_project_id, search_papers, extract_info, read_file]
    )
