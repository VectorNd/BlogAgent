import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import re

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
        
    def _generate_safe_filename(self, metadata: Dict[str, Any]) -> str:
        """Generate a safe filename from SEO metadata."""
        try:
            # First try to use the slug if it exists
            if 'slug' in metadata and metadata['slug']:
                filename = metadata['slug']
            else:
                # Fallback to title if no slug
                filename = metadata['title']
            
            # Convert to lowercase
            filename = filename.lower()
            
            # Replace spaces with hyphens
            filename = filename.replace(" ", "-")
            
            # Remove special characters but keep hyphens
            filename = re.sub(r'[^a-z0-9-]', '', filename)
            
            # Remove multiple consecutive hyphens
            filename = re.sub(r'-+', '-', filename)
            
            # Remove leading/trailing hyphens
            filename = filename.strip('-')
            
            # Ensure filename is not too long
            if len(filename) > 50:
                # Try to truncate at a word boundary
                truncated = filename[:47]
                last_hyphen = truncated.rfind('-')
                if last_hyphen > 0:
                    filename = truncated[:last_hyphen]
                else:
                    filename = truncated
                filename += "..."
            
            self.logger.info(f"Generated filename: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"Error generating filename: {str(e)}")
            # Fallback to a safe default
            return f"blog-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
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
        
        summary = f"""
        {'='*50}
        ✅ Blog Generated: {metadata['title']}
        📝 Topic: {topic}
        ⏱️ Reading Time: {metadata['reading_time']} minutes
        🔑 Keywords: {', '.join(metadata['keywords'])}
        🔗 URL Slug: {metadata.get('slug', 'N/A')}
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
            # Generate safe filename from metadata
            filename = self._generate_safe_filename(metadata)
            
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