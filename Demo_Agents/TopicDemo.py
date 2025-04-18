# Example usage
import json 
import logging
import asyncio
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.path_utils import setup_project_root

# Set up project root path
setup_project_root()

from Agents.TopicAgent import TopicAgent

async def demo():
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    topic_agent = TopicAgent(groq_api_key)
    
    # Test with a sample topic
    topic = "How Python is used in AI"
    tone = "educational"
    
    analysis = await topic_agent.analyze_topic(topic, tone)
    print(json.dumps(analysis, indent=2))
    
    structure = await topic_agent.suggest_blog_structure(analysis)
    print(json.dumps(structure, indent=2))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())