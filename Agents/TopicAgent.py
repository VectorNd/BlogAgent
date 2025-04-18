import asyncio
import functools
import logging
from typing import Dict, Optional, List

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class TopicAnalysis(BaseModel):
    """Schema for topic analysis output"""
    assessment: str = Field(description="Brief assessment of the topic's scope and target audience")
    subtopics: list[str] = Field(description="List of 4-6 logical subtopics for the blog post")
    depth: str = Field(description="Recommended knowledge depth (beginner/intermediate/advanced)")
    tone_guidance: str = Field(description="Specific tone recommendations for the content")

class TopicAgent:
    """
    Agent responsible for analyzing input topics and breaking them into subtopics.
    
    This agent uses Google's Gemini API through LangChain to understand the given topic,
    break it into logical subtopics, and determine appropriate content tone.
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize the TopicAgent with Groq API key.
        
        Args:
            groq_api_key: API key for Groq
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize LangChain with Gemini
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     google_api_key=gemini_api_key,
        #     temperature=0.7
        # )

        self.llm = ChatGroq(
            model="gemma2-9b-it",
            api_key=groq_api_key,
            temperature=0.7
        )
        
        # Set up the output parser
        self.parser = JsonOutputParser(pydantic_object=TopicAnalysis)
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional blog content strategist. Analyze the given topic and provide:
            1. A brief assessment of the topic's scope and target audience
            2. 4-6 logical subtopics that would make good H2 headings
            3. Recommended knowledge depth (beginner/intermediate/advanced)
            4. Specific tone guidance for the content
            
            Format your response as JSON following the specified schema."""),
            ("human", "Topic: {topic}\nTone: {tone}")
        ])
        
        # Cache for topic analysis results
        self.topic_cache = {}
        
    @functools.lru_cache(maxsize=50)
    async def analyze_topic(self, topic: str, tone: Optional[str] = None) -> Dict:
        """
        Analyze a topic to break it down into subtopics and determine appropriate structure.
        
        Args:
            topic: The main blog topic to analyze
            tone: Optional tone specification (educational, formal, creative, etc.)
            
        Returns:
            Dictionary containing analyzed topic structure with subtopics and tone guidance
        """
        self.logger.info(f"Analyzing topic: '{topic}' with tone: {tone if tone else 'not specified'}")
        
        # Check cache first
        cache_key = f"{topic}_{tone}"
        if cache_key in self.topic_cache:
            self.logger.info(f"Using cached analysis for '{topic}'")
            return self.topic_cache[cache_key]
        
        try:
            # Create the chain
            chain = self.prompt | self.llm | self.parser
            
            # Generate content with retry logic
            response = await self._generate_with_retry(chain, topic, tone)
            
            # Add additional metadata
            topic_analysis = {
                "topic": topic,
                "requested_tone": tone,
                **response  # response is already a dict, no need for .dict()
            }
            
            # Cache the result
            self.topic_cache[cache_key] = topic_analysis
            
            self.logger.info(f"Successfully analyzed topic with {len(topic_analysis['subtopics'])} subtopics")
            return topic_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing topic '{topic}': {str(e)}")
            raise
    
    async def _generate_with_retry(self, chain, topic: str, tone: Optional[str], max_retries: int = 3) -> TopicAnalysis:
        """
        Generate content with retry logic.
        
        Args:
            chain: The LangChain chain to use
            topic: The topic to analyze
            tone: Optional tone specification
            max_retries: Maximum number of retry attempts
            
        Returns:
            Parsed topic analysis
        """
        retries = 0
        while retries < max_retries:
            try:
                # Run the chain
                response = await asyncio.to_thread(
                    chain.invoke,
                    {"topic": topic, "tone": tone if tone else "not specified"}
                )
                return response
            except Exception as e:
                self.logger.warning(f"Error in generation (attempt {retries+1}): {str(e)}")
                await asyncio.sleep(2 ** retries)  # Exponential backoff
                retries += 1
                
        raise Exception(f"Failed to generate topic analysis after {max_retries} attempts")
    
    async def suggest_blog_structure(self, topic_analysis: Dict) -> Dict:
        """
        Suggest a comprehensive blog structure based on topic analysis.
        
        Args:
            topic_analysis: The output from analyze_topic()
            
        Returns:
            Dictionary with suggested blog structure including word counts
        """
        topic = topic_analysis["topic"]
        subtopics = topic_analysis["subtopics"]
        depth = topic_analysis["knowledge_depth"]
        
        # Adjust word count based on depth
        word_counts = {
            "beginner": {"intro": 120, "section": 200, "conclusion": 100},
            "intermediate": {"intro": 150, "section": 250, "conclusion": 120},
            "advanced": {"intro": 180, "section": 300, "conclusion": 150}
        }
        
        counts = word_counts.get(depth, word_counts["intermediate"])
        
        # Calculate total approximate word count
        total_words = counts["intro"] + (len(subtopics) * counts["section"]) + counts["conclusion"]
        
        structure = {
            "title": f"Comprehensive Guide to {topic}",
            "intro_word_count": counts["intro"],
            "sections": [{"heading": subtopic, "word_count": counts["section"]} for subtopic in subtopics],
            "conclusion_word_count": counts["conclusion"],
            "estimated_total_words": total_words,
            "estimated_reading_time": f"{round(total_words / 200, 1)} minutes"  # Assuming 200 words per minute
        }
        
        return structure

    async def suggest_related_topics(self, topic: str) -> List[str]:
        """Suggest related topics for the given main topic."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a topic suggester. Suggest related topics that would complement the main topic."),
            ("human", f"Main Topic: {topic}")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({})
        return result.split("\n")
    
    async def validate_topic(self, topic: str) -> Dict:
        """Validate if the topic is suitable for a blog post."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a topic validator. Assess if the topic is suitable for a blog post."),
            ("human", f"Topic: {topic}")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({})
        return {"topic": topic, "is_valid": "yes" in result.lower(), "feedback": result}