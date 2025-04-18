import asyncio
import os
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.path_utils import setup_project_root

# Set up project root path
setup_project_root()

from Agents.SEOAgent import SEOAgent

# Load environment variables
load_dotenv()

async def main():
    # Initialize Gemini client
    groq_client = ChatGroq(
        model="gemma2-9b-it",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7
    )
    
    # Initialize SEOAgent
    seo_agent = SEOAgent(groq_client)
    
    # Sample blog content
    topic = "Python Web Development"
    blog_content = """
    Python has become one of the most popular programming languages for web development. 
    With frameworks like Django and Flask, developers can create robust web applications quickly.
    
    Django, a high-level Python web framework, follows the "batteries-included" philosophy. 
    It provides an admin panel, database interfaces, and an ORM out of the box. 
    Django's security features help developers avoid common security mistakes.
    
    Flask, on the other hand, is a microframework that gives developers more flexibility. 
    It's perfect for smaller applications or when you need more control over components.
    
    Both frameworks have large communities and extensive documentation. 
    They support various databases and can be deployed on different platforms.
    """
    
    # Sample Datamuse keywords (simulating API response)
    datamuse_keywords = [
        {"word": "python", "score": 100},
        {"word": "web", "score": 90},
        {"word": "development", "score": 85},
        {"word": "framework", "score": 80},
        {"word": "django", "score": 95},
        {"word": "flask", "score": 90}
    ]
    
    try:
        # Run SEO optimization
        print("\n=== Starting SEO Optimization ===")
        seo_results = await seo_agent.optimize_content(topic, blog_content, datamuse_keywords)
        
        # Print results
        print("\n=== SEO Results ===")
        print(f"Title: {seo_results['title']}")
        print(f"Meta Description: {seo_results['meta_description']}")
        print(f"Keywords: {', '.join(seo_results['keywords'])}")
        print(f"Reading Time: {seo_results['reading_time']} minutes")
        print(f"Slug: {seo_results['slug']}")
        
        # Test individual methods
        print("\n=== Testing Individual Methods ===")
        
        # Test title generation
        title = await seo_agent.generate_title(topic, blog_content)
        print(f"\nGenerated Title: {title}")
        
        # Test meta description
        meta = await seo_agent.generate_meta_description(blog_content)
        print(f"\nGenerated Meta Description: {meta}")
        
        # Test keyword extraction
        keywords = await seo_agent.extract_keywords(topic, blog_content, datamuse_keywords)
        print(f"\nExtracted Keywords: {keywords}")
        
        # Test reading time calculation
        reading_time = await seo_agent.calculate_reading_time(blog_content)
        print(f"\nCalculated Reading Time: {reading_time} minutes")
        
        # Test slug generation
        slug = await seo_agent.suggest_slug(title)
        print(f"\nGenerated Slug: {slug}")
        
        # Test caching by running the same operations again
        print("\n=== Testing Cache ===")
        print("Running the same operations again to test caching...")
        
        # These should be faster due to caching
        cached_title = await seo_agent.generate_title(topic, blog_content)
        cached_meta = await seo_agent.generate_meta_description(blog_content)
        cached_keywords = await seo_agent.extract_keywords(topic, blog_content, datamuse_keywords)
        
        print("\nCached Results:")
        print(f"Title (cached): {cached_title}")
        print(f"Meta (cached): {cached_meta}")
        print(f"Keywords (cached): {cached_keywords}")
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")
    finally:
        # Clean up
        await seo_agent.close_session()

if __name__ == "__main__":
    asyncio.run(main())