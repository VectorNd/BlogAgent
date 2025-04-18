import logging
from typing import Dict, Optional
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from utils.quality_metrics import QualityMetrics

class AgentCoordinator:
    """
    Coordinates the interaction between different agents in the blog generation process.
    Manages the workflow, implements quality checks, and handles error recovery.
    """
    
    def __init__(self, topic_agent, context_agent, writing_agent, seo_agent, execution_agent):
        """
        Initialize the coordinator with all required agents.
        
        Args:
            topic_agent: Instance of TopicAgent
            context_agent: Instance of ContextAgent
            writing_agent: Instance of WritingAgent
            seo_agent: Instance of SEOAgent
            execution_agent: Instance of ExecutionAgent
        """
        self.topic_agent = topic_agent
        self.context_agent = context_agent
        self.writing_agent = writing_agent
        self.seo_agent = seo_agent
        self.execution_agent = execution_agent
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Initialize progress tracking
        self.progress = {
            "topic_analysis": False,
            "context_research": False,
            "content_generation": False,
            "seo_optimization": False,
            "export": False
        }
        
        # Initialize quality metrics
        self.quality_metrics = {
            "readability": 0.0,
            "keyword_density": {},
            "structure": 0.0,
            "length": 0.0
        }
        
        # Initialize retry counts
        self.retry_counts = {
            "topic_analysis": 0,
            "context_research": 0,
            "content_generation": 0,
            "seo_optimization": 0,
            "export": 0
        }
        
        # Maximum retry attempts
        self.max_retries = 3
    
    # async def _validate_topic_analysis(self, topic_analysis: Dict) -> bool:
    #     """Validate the output of topic analysis."""
    #     required_fields = ["assessment", "subtopics", "depth", "tone_guidance"]
    #     if not all(field in topic_analysis for field in required_fields):
    #         self.logger.error("Topic analysis missing required fields")
    #         return False
            
    #     if not isinstance(topic_analysis["subtopics"], list) or len(topic_analysis["subtopics"]) < 3:
    #         self.logger.error("Insufficient subtopics generated")
    #         return False
            
    #     return True
    
    # async def _validate_context_research(self, research_context: Dict) -> bool:
    #     """Validate the research context."""
    #     required_fields = ["keywords", "sources", "insights"]
    #     if not all(field in research_context for field in required_fields):
    #         self.logger.error("Research context missing required fields")
    #         return False
            
    #     if not research_context["keywords"] or not research_context["sources"]:
    #         self.logger.error("Insufficient research data")
    #         return False
            
    #     return True
    
    # async def _validate_content(self, content: str, topic_analysis: Dict, research_context: Dict) -> bool:
    #     """Validate the generated content."""
    #     # Check content length
    #     word_count = len(content.split())
    #     if word_count < 500:
    #         self.logger.error("Content too short")
    #         return False
            
    #     # Check for subtopic coverage
    #     for subtopic in topic_analysis["subtopics"]:
    #         if subtopic.lower() not in content.lower():
    #             self.logger.warning(f"Subtopic '{subtopic}' not covered in content")
    #             return False
                
    #     # Check for keyword coverage
    #     for keyword in research_context["keywords"]:
    #         if keyword.lower() not in content.lower():
    #             self.logger.warning(f"Keyword '{keyword}' not used in content")
                
    #     return True
    
    # async def _validate_seo_optimization(self, seo_metadata: Dict) -> bool:
    #     """Validate SEO optimization results."""
    #     required_fields = ["title", "meta_description", "keywords", "reading_time", "slug"]
    #     if not all(field in seo_metadata for field in required_fields):
    #         self.logger.error("SEO metadata missing required fields")
    #         return False
            
    #     if len(seo_metadata["keywords"]) < 3:
    #         self.logger.error("Insufficient keywords generated")
    #         return False
            
    #     return True
    
    async def generate_blog(self, topic: str, tone: Optional[str] = None) -> Dict:
        """
        Coordinate the blog generation process with quality checks and error recovery.
        
        Args:
            topic: The main topic for the blog post
            tone: Optional tone specification
            
        Returns:
            Dictionary containing the generated blog and metadata
        """
        self.logger.info(f"Starting blog generation for topic: {topic}")
        
        try:
            # Step 1: Topic Analysis
            self.logger.info("Analyzing topic...")
            topic_analysis = await self.topic_agent.analyze_topic(topic, tone)
            
            # if not await self._validate_topic_analysis(topic_analysis):
            #     raise ValueError("Topic analysis validation failed")

            print(topic_analysis)
                
            self.progress["topic_analysis"] = True
            
            # Step 2: Context Research
            self.logger.info("Conducting research...")
            research_context = await self.context_agent.gather_context(
                topic, topic_analysis["subtopics"]
            )
            
            print(research_context)
            # if not await self._validate_context_research(research_context):
            #     raise ValueError("Context research validation failed")
                
            self.progress["context_research"] = True
            
            # Step 3: Content Generation
            self.logger.info("Generating content...")
            blog_content = await self.writing_agent.generate_full_blog(
                topic_analysis, research_context
            )
            
            # if not await self._validate_content(blog_content, topic_analysis, research_context):
            #     raise ValueError("Content validation failed")

            print(blog_content)
                
            self.progress["content_generation"] = True
            
            # Step 4: SEO Optimization
            self.logger.info("Optimizing for SEO...")
            seo_metadata = await self.seo_agent.optimize_content(
                topic=topic,
                context=research_context,
                tone=tone,
                content=blog_content
            )
            
            # if not await self._validate_seo_optimization(seo_metadata):
            #     raise ValueError("SEO optimization validation failed")

            print(seo_metadata)
                
            self.progress["seo_optimization"] = True
            
            # Step 5: Export
            self.logger.info("Exporting results...")
            export_results = await self.execution_agent.execute_export(
                topic=topic,
                content=blog_content,
                metadata=seo_metadata
            )

            print(export_results)
            
            self.progress["export"] = True
            
            # Calculate quality metrics
            self.quality_metrics = QualityMetrics.calculate_overall_quality(
                blog_content,
                seo_metadata["keywords"],
                topic_analysis["subtopics"]
            )

            print(self.quality_metrics)
            
            self.logger.info("Blog generation completed successfully")
            
            return {
                "content": blog_content,
                "metadata": seo_metadata,
                "topic_analysis": topic_analysis,
                "research_context": research_context,
                "quality_metrics": self.quality_metrics,
                "files": export_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in blog generation: {str(e)}")
    
    # async def _handle_error(self, error: Exception):
    #     """Handle errors and implement recovery strategies."""
    #     self.logger.error(f"Error occurred: {str(error)}")
        
    #     # Implement recovery based on progress
    #     if not self.progress["topic_analysis"]:
    #         if self.retry_counts["topic_analysis"] < self.max_retries:
    #             self.logger.info("Retrying topic analysis...")
    #             self.retry_counts["topic_analysis"] += 1
    #             # Add retry logic here
    #         else:
    #             self.logger.error("Max retries reached for topic analysis")
                
    #     elif not self.progress["context_research"]:
    #         if self.retry_counts["context_research"] < self.max_retries:
    #             self.logger.info("Retrying context research...")
    #             self.retry_counts["context_research"] += 1
    #             # Add retry logic here
    #         else:
    #             self.logger.error("Max retries reached for context research")
                
    #     elif not self.progress["content_generation"]:
    #         if self.retry_counts["content_generation"] < self.max_retries:
    #             self.logger.info("Retrying content generation...")
    #             self.retry_counts["content_generation"] += 1
    #             # Add retry logic here
    #         else:
    #             self.logger.error("Max retries reached for content generation")
                
    #     elif not self.progress["seo_optimization"]:
    #         if self.retry_counts["seo_optimization"] < self.max_retries:
    #             self.logger.info("Retrying SEO optimization...")
    #             self.retry_counts["seo_optimization"] += 1
    #             # Add retry logic here
    #         else:
    #             self.logger.error("Max retries reached for SEO optimization")
                
    #     elif not self.progress["export"]:
    #         if self.retry_counts["export"] < self.max_retries:
    #             self.logger.info("Retrying export...")
    #             self.retry_counts["export"] += 1
    #             # Add retry logic here
    #         else:
    #             self.logger.error("Max retries reached for export") 