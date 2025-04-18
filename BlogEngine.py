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
from Agents.AgentCoordinator import AgentCoordinator
from llm_evaluator import BlogEvaluator

load_dotenv()

class BlogEngine:
    """
    Main engine that orchestrates the blog generation process.
    
    This class coordinates all agents to generate a complete, SEO-optimized blog post.
    It handles the workflow from topic analysis to final export.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the BlogEngine with necessary API keys.
        
        Args:
            output_dir: Directory to save generated files
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
            self.execution_agent = ExecutionAgent(output_dir=output_dir)

            # Initialize blog evaluator
            self.blog_evaluator = BlogEvaluator()
            
            # Initialize agent coordinator
            self.coordinator = AgentCoordinator(
                self.topic_agent,
                self.context_agent,
                self.writing_agent,
                self.seo_agent,
                self.execution_agent
            )
            
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
            - quality_metrics: Content quality scores
            - research_context: Research data and insights
            - files: Paths to exported files
        """
        try:
            self.logger.info(f"Starting blog generation for topic: {topic}")
            
            # Use coordinator to generate blog
            result = await self.coordinator.generate_blog(topic, tone)
            
            # Additional evaluation if needed
            evaluation_scores = self.blog_evaluator.evaluate_blog(result["content"])
            result["evaluation"] = evaluation_scores
            
            self.logger.info("Blog generation completed successfully")
            
            return result
            
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