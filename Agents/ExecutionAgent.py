import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

class ExecutionAgent:
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Initialized ExecutionAgent with output directory: {output_dir}")
        
    async def export_markdown(self, content: str, filename: str) -> str:
        """Export blog content as markdown file."""
        self.logger.info(f"Exporting markdown content to {filename}.md")
        filepath = os.path.join(self.output_dir, f"{filename}.md")
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"Successfully exported markdown to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error exporting markdown: {str(e)}")
            raise
            
    async def export_metadata(self, metadata: Dict[str, Any], filename: str) -> str:
        """Export SEO metadata as JSON file."""
        self.logger.info(f"Exporting metadata to {filename}_meta.json")
        filepath = os.path.join(self.output_dir, f"{filename}_meta.json")
        
        try:
            # Add export timestamp to metadata
            metadata['export_timestamp'] = datetime.now().isoformat()
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Successfully exported metadata to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Error exporting metadata: {str(e)}")
            raise
            
    async def summarize_process(self, topic: str, metadata: Dict[str, Any], md_path: str, json_path: str) -> None:
        """Display CLI summary of completed process."""
        self.logger.info("Generating process summary")
        
        # # Create summary template
        # summary_template = ChatPromptTemplate.from_messages([
        #     ("system", "You are a helpful assistant that generates beautiful CLI summaries."),
        #     ("user", """Generate a beautiful CLI summary for a blog generation process with the following details:
        #     - Blog Title: {title}
        #     - Topic: {topic}
        #     - Reading Time: {reading_time} minutes
        #     - Keywords: {keywords}
        #     - Generated Files:
        #       - Markdown: {md_path}
        #       - Metadata: {json_path}
        #     """)
        # ])
        
        # Format the summary
        summary = f"""
        {'='*50}
        ✅ Blog Generated: {metadata['title']}
        📝 Topic: {topic}
        ⏱️ Reading Time: {metadata['reading_time']} minutes
        🔑 Keywords: {', '.join(metadata['keywords'])}
        🔗 Suggested URL: {metadata['slug']}
        📂 Files created:
        - {md_path}
        - {json_path}
        {'='*50}
        """
        
        print(summary)
        self.logger.info("Process summary displayed")
        
    async def execute_export(self, topic: str, content: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Execute the complete export process."""
        self.logger.info("Starting export process")
        
        try:
            # Generate filenames based on slug
            filename = metadata['slug']
            
            # Export content and metadata
            md_path = await self.export_markdown(content, filename)
            json_path = await self.export_metadata(metadata, filename)
            
            # Display summary
            await self.summarize_process(topic, metadata, md_path, json_path)
            
            return {
                "markdown_path": md_path,
                "metadata_path": json_path
            }
            
        except Exception as e:
            self.logger.error(f"Error during export process: {str(e)}")
            raise