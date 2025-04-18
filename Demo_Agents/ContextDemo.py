# Example usage

import os
import logging
import asyncio
import sys
from dotenv import load_dotenv
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)
from utils.path_utils import setup_project_root

# Set up project root path
setup_project_root()

from Agents.ContextAgent import ContextAgent

async def demo():
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load environment variables
    load_dotenv()
    newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not newsdata_api_key or not groq_api_key:
        print("Error: API keys not found in environment variables")
        return
    
    # Create ContextAgent
    context_agent = ContextAgent(newsdata_api_key, groq_api_key)
    
    try:
        # Test with a sample topic and subtopics
        topic = "Python in AI Development"
        subtopics = [
            "Machine Learning Libraries in Python",
            "Natural Language Processing with Python",
            "Python for Data Preprocessing",
            "Deploying AI Models with Python"
        ]
        
        print(f"\n--- Gathering context for topic: '{topic}' ---\n")
        
        # Get enriched context
        context = await context_agent.prepare_enriched_context(topic, subtopics)
        
        # Display results
        print("\n=== CONTEXT SUMMARY ===\n")
        
        print(f"Topic: {context['topic']}")
        print(f"Timestamp: {context['timestamp']}")
        
        print("\n--- Top News Articles ---")
        for i, article in enumerate(context['top_news'], 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article['source_name']}")
            print(f"   Link: {article['link']}")
            print()
        
        print("\n--- Top Keywords ---")
        for i, keyword in enumerate(context['top_keywords'], 1):
            print(f"{i}. {keyword}")
        
        print("\n--- Best Quote ---")
        if context['best_quote']:
            print(f"\"{context['best_quote']['content']}\"")
            print(f"- {context['best_quote']['author']}")
        else:
            print("No relevant quote found.")
        
        print("\n--- Content Gaps ---")
        print(context['content_gaps'])
        
        print("\n--- Subtopic Keywords ---")
        for subtopic, keywords in context['subtopic_keywords'].items():
            print(f"\n{subtopic}:")
            for i, kw in enumerate(keywords[:5], 1):
                print(f"  {i}. {kw['word']}")
        
    finally:
        # Close aiohttp session
        await context_agent.close_session()

if __name__ == "__main__":
    asyncio.run(demo())