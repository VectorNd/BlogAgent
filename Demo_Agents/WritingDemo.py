# Example usage

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import logging
from dotenv import load_dotenv
import asyncio

from utils.path_utils import setup_project_root

# Set up project root path
setup_project_root()

from Agents.WritingAgent import WritingAgent

async def demo():
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment variables")
        return
    
    # Create WritingAgent
    writing_agent = WritingAgent(groq_api_key)
    
    # Mock topic analysis and context data for testing
    topic_analysis = {
        "topic": "How Python is used in Machine Learning",
        "assessment": "This topic explores the application of Python programming in machine learning, suitable for developers with some programming experience.",
        "subtopics": [
            "Popular Python Libraries for Machine Learning",
            "Data Preprocessing with Python",
            "Building ML Models in Python",
            "Evaluating Model Performance",
            "Deploying ML Models"
        ],
        "depth": "intermediate",
        "tone_guidance": "educational with practical examples",
        "requested_tone": "educational"
    }
    
    context = {
        "topic": "How Python is used in Machine Learning",
        "top_news": [
            {
                "title": "New TensorFlow Update Improves Python Integration",
                "description": "The latest TensorFlow update brings better Python integration with improved GPU utilization.",
                "link": "https://example.com/news1"
            }
        ],
        "top_keywords": [
            "Python machine learning",
            "TensorFlow",
            "PyTorch",
            "scikit-learn",
            "data preprocessing",
            "model deployment",
            "neural networks",
            "deep learning"
        ],
        "subtopic_keywords": {
            "Popular Python Libraries for Machine Learning": [
                {"word": "TensorFlow"},
                {"word": "PyTorch"},
                {"word": "scikit-learn"}
            ],
            "Data Preprocessing with Python": [
                {"word": "pandas"},
                {"word": "NumPy"},
                {"word": "data cleaning"}
            ]
        },
        "best_quote": {
            "content": "Python is the second best language for everything, and that's why it's the best language for machine learning.",
            "author": "Jake VanderPlas"
        },
        "all_quotes": [
            {
                "content": "Python is the second best language for everything, and that's why it's the best language for machine learning.",
                "author": "Jake VanderPlas"
            }
        ],
        "content_gaps": "Consider adding more information about recent advancements in AutoML."
    }
    
    try:
        # Generate a full blog post
        print("\n--- Generating full blog post ---\n")
        blog_data = await writing_agent.generate_blog_with_title(topic_analysis, context)
        
        # Save the blog to a file
        filename = "generated_blog.md"
        with open(filename, "w") as f:
            f.write(blog_data["content"])
        
        print("\n=== Blog Generation Complete ===")
        print(f"Title: {blog_data['title']}")
        print(f"Word Count: {blog_data['word_count']}")
        print(f"Reading Time: {blog_data['reading_time']} minutes")
        print(f"Saved to: {filename}")
        
        # Print a preview
        preview_lines = blog_data["content"].split("\n")[:15]
        print("\n=== Blog Preview ===\n")
        print("\n".join(preview_lines))
        print("...\n")
        
    except Exception as e:
        print(f"Error in demo: {str(e)}")

if __name__ == "__main__":
    asyncio.run(demo())