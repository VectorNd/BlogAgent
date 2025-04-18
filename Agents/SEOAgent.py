from typing import List, Dict
import asyncio
import aiohttp
import logging
import functools
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

class SEOAgent:
    def __init__(self, gemini_client: ChatGoogleGenerativeAI):
        self.gemini_client = gemini_client
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Initialize keyword cache
        self.keyword_cache = {}
        self.cache_expiry = 3600  # Cache expiry time in seconds (1 hour)
        
    async def initialize_session(self):
        if not self.session:
            self.logger.info("Initializing aiohttp session")
            self.session = aiohttp.ClientSession()
            
    async def close_session(self):
        if self.session:
            self.logger.info("Closing aiohttp session")
            await self.session.close()
            self.session = None
            
    def _is_cache_valid(self, cache_timestamp: float) -> bool:
        """Check if cache entry is still valid."""
        current_time = datetime.now().timestamp()
        return (current_time - cache_timestamp) < self.cache_expiry
        
    @functools.lru_cache(maxsize=100)
    async def generate_title(self, topic: str, blog_content: str) -> str:
        """Generate SEO-optimized title using Gemini."""
        self.logger.info(f"Generating title for topic: {topic}")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SEO specialist. Generate a compelling, SEO-optimized title for the given blog content. The title should be engaging, include relevant keywords, and be under 60 characters."),
            ("user", f"Topic: {topic}\nContent: {blog_content}")
        ])
        
        try:
            chain = prompt | self.gemini_client | StrOutputParser()
            title = await chain.ainvoke({})
            self.logger.info(f"Generated title: {title.strip()}")
            return title.strip()
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            raise
        
    @functools.lru_cache(maxsize=100)
    async def generate_meta_description(self, blog_content: str) -> str:
        """Generate meta description (max 160 chars) using Gemini."""
        self.logger.info("Generating meta description")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SEO specialist. Generate a compelling meta description for the given blog content. The description should be under 160 characters, include relevant keywords, and encourage clicks."),
            ("user", blog_content)
        ])
        
        try:
            chain = prompt | self.gemini_client | StrOutputParser()
            description = await chain.ainvoke({})
            self.logger.info(f"Generated meta description: {description.strip()[:160]}")
            return description.strip()[:160]
        except Exception as e:
            self.logger.error(f"Error generating meta description: {str(e)}")
            raise
        
    async def extract_keywords(self, topic: str, blog_content: str, datamuse_keywords: List[Dict]) -> List[str]:
        """Generate relevant tags/keywords using both Gemini and Datamuse."""
        self.logger.info(f"Extracting keywords for topic: {topic}")
        
        # Check cache first
        cache_key = f"{topic}_{hash(blog_content)}"
        if cache_key in self.keyword_cache:
            cache_entry = self.keyword_cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.logger.info("Using cached keywords")
                return cache_entry['keywords']
        
        try:
            # Get keywords from Gemini
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert SEO specialist. Extract the most relevant keywords and tags from the given content. Return them as a comma-separated list."),
                ("user", f"Topic: {topic}\nContent: {blog_content}")
            ])
            
            chain = prompt | self.gemini_client | StrOutputParser()
            gemini_keywords = await chain.ainvoke({})
            gemini_keywords = [k.strip() for k in gemini_keywords.split(",")]
            self.logger.info(f"Gemini keywords: {gemini_keywords}")
            
            # Combine with Datamuse keywords
            datamuse_words = [kw["word"] for kw in datamuse_keywords]
            self.logger.info(f"Datamuse keywords: {datamuse_words}")
            
            # Remove duplicates and limit to 10 keywords
            all_keywords = list(set(gemini_keywords + datamuse_words))
            final_keywords = all_keywords[:10]
            
            # Cache the results
            self.keyword_cache[cache_key] = {
                'keywords': final_keywords,
                'timestamp': datetime.now().timestamp()
            }
            
            self.logger.info(f"Final keywords: {final_keywords}")
            return final_keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {str(e)}")
            raise
        
    @functools.lru_cache(maxsize=1000)
    async def calculate_reading_time(self, content: str) -> int:
        """Estimate reading time based on word count."""
        self.logger.info("Calculating reading time")
        words = content.split()
        # Assuming average reading speed of 200 words per minute
        reading_time = len(words) // 200
        result = max(1, reading_time)  # Minimum 1 minute
        self.logger.info(f"Estimated reading time: {result} minutes")
        return result
        
    @functools.lru_cache(maxsize=100)
    async def suggest_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title."""
        self.logger.info(f"Generating slug for title: {title}")
        # Convert to lowercase
        slug = title.lower()
        
        # Replace spaces with hyphens
        slug = slug.replace(" ", "-")
        
        # Remove special characters
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        
        # Remove multiple consecutive hyphens
        while "--" in slug:
            slug = slug.replace("--", "-")
            
        # Remove leading/trailing hyphens
        slug = slug.strip("-")
        
        self.logger.info(f"Generated slug: {slug}")
        return slug
        
    async def optimize_content(self, topic: str, blog_content: str, datamuse_keywords: List[Dict]) -> Dict:
        """Run all SEO tasks concurrently."""
        self.logger.info(f"Starting SEO optimization for topic: {topic}")
        # Initialize session if needed
        await self.initialize_session()
        
        try:
            # Run all SEO tasks concurrently
            tasks = [
                self.generate_title(topic, blog_content),
                self.generate_meta_description(blog_content),
                self.extract_keywords(topic, blog_content, datamuse_keywords),
                self.calculate_reading_time(blog_content),
            ]
            
            self.logger.info("Executing SEO tasks concurrently")
            title, meta, keywords, reading_time = await asyncio.gather(*tasks)
            slug = await self.suggest_slug(title)
            
            result = {
                "title": title,
                "meta_description": meta,
                "keywords": keywords,
                "reading_time": reading_time,
                "slug": slug
            }
            
            self.logger.info("SEO optimization completed successfully")
            self.logger.debug(f"SEO results: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during SEO optimization: {str(e)}")
            raise            
        finally:
            # Clean up session
            await self.close_session()
