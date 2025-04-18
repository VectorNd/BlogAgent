import logging
import os 
from dotenv import load_dotenv
from typing import Dict, Optional, Any
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.path_utils import setup_project_root

# Set up project root path
setup_project_root()

from Agents.TopicAgent import TopicAgent
from Agents.ContextAgent import ContextAgent
from Agents.WritingAgent import WritingAgent
from Agents.SEOAgent import SEOAgent
from Agents.ExecutionAgent import ExecutionAgent
from llm_evaluator import BlogEvaluator

load_dotenv()

class BlogEngine:
    """
    Main engine that orchestrates the blog generation process.
    
    This class coordinates all agents to generate a complete, SEO-optimized blog post.
    It handles the workflow from topic analysis to final export.
    """
    
    def __init__(self):
        """
        Initialize the BlogEngine with necessary API keys.
        
        Args:
            api_keys: Dictionary containing API keys for various services
                     Required keys: 'gemini', 'newsdata'
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        try:
            # Initialize Gemini client
            # self.gemini_client = ChatGoogleGenerativeAI(
            #         model="gemini-2.0-flash-001",
            #         google_api_key=os.getenv("GEMINI_API_KEY"),
            #         temperature=0.7
            # )

            # self.gemini_client = ChatGroq(
            #     model="gemma2-9b-it",
            #     api_key=os.getenv("GROQ_API_KEY"),
            #     temperature=0.7
            # )
            
            # Initialize agents
            self.topic_agent = TopicAgent(os.getenv("GROQ_API_KEY"))
            self.context_agent = ContextAgent(os.getenv("NEWSDATA_API_KEY"), os.getenv("GROQ_API_KEY"))
            self.writing_agent = WritingAgent(os.getenv("GROQ_API_KEY"))
            self.seo_agent = SEOAgent(os.getenv("GROQ_API_KEY"))
            self.execution_agent = ExecutionAgent()

            # Initialize blog evaluator
            self.blog_evaluator = BlogEvaluator()
            
            self.logger.info("BlogEngine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing BlogEngine: {str(e)}")
            raise
    
    async def generate_blog(self, topic: str, tone: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a complete blog post with SEO optimization.
        
        Args:
            topic: The main topic for the blog post
            tone: Optional tone specification (e.g., 'professional', 'casual')
            
        Returns:
            Dictionary containing:
            - content: The generated blog content
            - metadata: SEO metadata
            - files: Paths to exported files
        """
        try:
            self.logger.info(f"Starting blog generation for topic: {topic}")
            
            # Step 1: Analyze topic
            self.logger.info("Analyzing topic...")
            topic_analysis = await self.topic_agent.analyze_topic(topic, tone)
            
            # Step 2: Conduct research
            self.logger.info("Gathering context...")
            research_context = await self.context_agent.gather_context(
                topic, topic_analysis["subtopics"]
            )
            
            # Step 3: Generate content
            self.logger.info("Generating blog content...")
            blog_content = await self.writing_agent.generate_full_blog(
                topic_analysis, research_context
            )

            # Step 4: Evaluate blog content
            self.logger.info("Evaluating blog content...")
            evaluation_scores = self.blog_evaluator.evaluate_blog(blog_content)
            
            # Step 5: Optimize for SEO
            self.logger.info("Optimizing for SEO...")
            seo_metadata = await self.seo_agent.optimize_content(
                topic, blog_content, research_context["keywords"]
            )
            
            # Step 6: Export and summarize
            self.logger.info("Exporting results...")
            filename_base = seo_metadata["slug"]
            md_path = await self.execution_agent.export_markdown(blog_content, filename_base)
            json_path = await self.execution_agent.export_metadata(seo_metadata, filename_base)


            # eval_path = await self.execution_agent.export_evaluation(evaluation_scores, filename_base)
            
            # Step 7: Summarize process
            await self.execution_agent.summarize_process(topic, seo_metadata, md_path, json_path)
            
            self.logger.info("Blog generation completed successfully")
            
            return {
                "content": blog_content,
                "metadata": seo_metadata,
                "evaluation": evaluation_scores,
                "files": {
                    "markdown": md_path,
                    "json": json_path
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating blog: {str(e)}")
            raise
        finally:
            # Clean up any resources
            await self._cleanup()
    
    async def _cleanup(self):
        """Clean up resources used by agents."""
        try:
            if hasattr(self, 'context_agent'):
                await self.context_agent.close_session()
            if hasattr(self, 'seo_agent'):
                await self.seo_agent.close_session()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")