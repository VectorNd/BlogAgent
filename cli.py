import argparse
import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from textstat import textstat
from BlogEngine import BlogEngine

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Blog Generation CLI - Generate SEO-optimized blog posts using AI"
    )
    
    # Required arguments
    parser.add_argument(
        "topic",
        help="Main topic for the blog post"
    )
    
    # Optional arguments
    parser.add_argument(
        "-t", "--tone",
        choices=["professional", "casual", "educational", "informative", "technical"],
        default="professional",
        help="Tone of the blog post (default: professional)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Directory to save generated files (default: ./output)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="Process multiple topics in batch mode"
    )
    
    return parser.parse_args()

async def generate_blog(topic: str, tone: str, output_dir: str, verbose: bool = False) -> Optional[dict]:
    try:
        if verbose:
            print(f"\nGenerating blog post for topic: {topic}")
            print(f"Tone: {tone}")
            print(f"Output directory: {output_dir}")
        
        # Initialize BlogEngine
        blog_engine = BlogEngine(output_dir=output_dir)
        
        # Generate blog
        result = await blog_engine.generate_blog(topic, tone)
        
        if verbose:
            print("\n=== Blog Generation Results ===")
            print(f"Title: {result['title']}")
            print(f"Reading Time: {result['reading_time']} minutes")
            print(f"Keywords: {', '.join(result['keywords'])}")
            
            # Calculate and display readability scores
            content = result['content']
            flesch_score = textstat.flesch_reading_ease(content)
            grade_level = textstat.flesch_kincaid_grade(content)
            
            print("\nReadability Scores:")
            print(f"Flesch Reading Ease: {flesch_score:.1f}")
            print(f"Flesch-Kincaid Grade Level: {grade_level:.1f}")
            
            print("\nGenerated Files:")
            print(f"- Markdown: {result['markdown_path']}")
            print(f"- Metadata: {result['metadata_path']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error generating blog for topic '{topic}':")
        print(f"Error: {str(e)}")
        return None

async def process_batch(topics: List[str], tone: str, output_dir: str, verbose: bool = False):
    results = []
    for topic in topics:
        print(f"\nProcessing topic: {topic}")
        result = await generate_blog(topic, tone, output_dir, verbose)
        if result:
            results.append(result)
    return results

async def main():
    # Load environment variables
    load_dotenv()
    
    # Verify required API keys
    required_keys = ["GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"Error: Missing required API keys: {', '.join(missing_keys)}")
        print("Please add them to your .env file")
        return
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.batch:
        # Read topics from stdin
        print("Enter topics (one per line, press Ctrl+D when done):")
        topics = []
        while True:
            try:
                topic = input()
                if topic.strip():
                    topics.append(topic.strip())
            except EOFError:
                break
        
        if not topics:
            print("No topics provided")
            return
        
        await process_batch(topics, args.tone, args.output_dir, args.verbose)
    else:
        await generate_blog(args.topic, args.tone, args.output_dir, args.verbose)

if __name__ == "__main__":
    asyncio.run(main()) 