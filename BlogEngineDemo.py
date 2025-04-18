import asyncio
import os
from dotenv import load_dotenv
from BlogEngine import BlogEngine

async def main():
    # Load environment variables
    load_dotenv()
    
    # Verify required environment variables
    required_vars = ["GROQ_API_KEY", "NEWSDATA_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please create a .env file with the following variables:")
        print("GROQ_API_KEY=your_groq_api_key")
        print("NEWSDATA_API_KEY=your_newsdata_api_key")
        return
    
    try:
        # Initialize BlogEngine
        print("\n=== Initializing BlogEngine ===")
        blog_engine = BlogEngine()
        
        # Test topics with different tones
        test_cases = [
            {
                "topic": "Python Web Development",
                "tone": "professional"
            }
            # {
            #     "topic": "Machine Learning Basics",
            #     "tone": "educational"
            # },
            # {
            #     "topic": "Cloud Computing Trends",
            #     "tone": "informative"
            # }
        ]
        
        for test_case in test_cases:
            print("\n=== Testing Blog Generation ===")
            print(f"Topic: {test_case['topic']}")
            print(f"Tone: {test_case['tone']}")
            
            try:
                # Generate blog
                result = await blog_engine.generate_blog(
                    topic=test_case['topic'],
                    tone=test_case['tone']
                )
                
                print("Yees")
                
                # Print results
                print("\n=== Blog Generation Results ===")
                print(result)
                
            except Exception as e:
                print(f"\n❌ Error generating blog for topic '{test_case['topic']}':")
                print(f"Error: {str(e)}")
                continue
            
            print("\n✓ Blog generated successfully")
        
        print("\n=== All tests completed ===")
        
    except Exception as e:
        print(f"\n❌ Error in demo: {str(e)}")
    finally:
        print("\nDemo completed")

if __name__ == "__main__":
    asyncio.run(main()) 