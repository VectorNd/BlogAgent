import asyncio
import os

from utils.path_utils import setup_project_root

# Set up project root path
setup_project_root()

from Agents.ExecutionAgent import ExecutionAgent

async def main():
    # Initialize ExecutionAgent with a test output directory
    execution_agent = ExecutionAgent(output_dir="./test_output")
    
    # Sample data for testing
    topic = "Test Blog Post"
    content = """
    # Test Blog Post
    
    This is a test blog post to verify the ExecutionAgent's functionality.
    
    ## Section 1
    This section contains some sample content to test the markdown export.
    
    ## Section 2
    More test content to ensure proper formatting and export.
    """
    
    metadata = {
        "title": "Test Blog Post: Verifying ExecutionAgent",
        "meta_description": "A test blog post to verify the functionality of the ExecutionAgent",
        "keywords": ["test", "execution", "agent", "verification"],
        "reading_time": 2,
        "slug": "test-blog-post-verification"
    }
    
    try:
        print("\n=== Testing ExecutionAgent ===")
        
        # Test individual methods
        print("\n1. Testing markdown export...")
        md_path = await execution_agent.export_markdown(content, metadata["slug"])
        print(f"✓ Markdown exported to: {md_path}")
        
        print("\n2. Testing metadata export...")
        json_path = await execution_agent.export_metadata(metadata, metadata["slug"])
        print(f"✓ Metadata exported to: {json_path}")
        
        print("\n3. Testing process summary...")
        await execution_agent.summarize_process(topic, metadata, md_path, json_path)
        
        print("\n4. Testing complete export process...")
        export_results = await execution_agent.execute_export(
            topic=topic,
            content=content,
            metadata=metadata
        )
        print("✓ Complete export successful:")
        print(f"  - Markdown: {export_results['markdown_path']}")
        print(f"  - Metadata: {export_results['metadata_path']}")
        
        print("\n=== All tests completed successfully ===")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
    finally:
        # Clean up test files
        try:
            if os.path.exists(md_path):
                os.remove(md_path)
            if os.path.exists(json_path):
                os.remove(json_path)
            if os.path.exists("./test_output"):
                os.rmdir("./test_output")
            print("\n✓ Test files cleaned up")
        except Exception as e:
            print(f"\n⚠️ Error during cleanup: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())