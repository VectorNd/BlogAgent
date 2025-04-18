from typing import List, Dict, Optional
import asyncio
import aiohttp
import logging
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

class SEOAgent:
    def __init__(self, groq_api_key: str):
        self.groq_client = ChatGroq(
            model_name="gemma2-9b-it",
            temperature=0.7,
            max_tokens=2048,
            max_retries=2,
            api_key=groq_api_key,
            model_kwargs={"top_p": 0.95}
        )
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already present
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        # Initialize caches
        self.title_cache = {}
        self.meta_cache = {}
        self.keyword_cache = {}
        self.reading_time_cache = {}
        self.slug_cache = {}
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
        
    def _generate_cache_key(self, topic: str, context: Dict, tone: str) -> str:
        """Generate a stable cache key from the input parameters."""
        # Create a stable string representation of the context
        context_str = json.dumps(context, sort_keys=True)
        return f"{topic}_{tone}_{context_str}"
        
    async def generate_title(self, topic: str, context: Dict, tone: str) -> str:
        """Generate SEO-optimized title using Groq."""
        self.logger.info(f"Generating title for topic: {topic}")
        
        # Check cache first
        cache_key = self._generate_cache_key(topic, context, tone)
        if cache_key in self.title_cache:
            cache_entry = self.title_cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.logger.info("Using cached title")
                return cache_entry['title']
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SEO specialist. Generate a compelling, SEO-optimized title for the given topic. The title should be engaging, include relevant keywords, and be under 60 characters."),
            ("user", "Topic: {topic}\nTone: {tone}\nContext: {context}")
        ])
        
        try:
            chain = prompt | self.groq_client | StrOutputParser()
            title = await chain.ainvoke({
                "topic": topic,
                "tone": tone,
                "context": json.dumps(context)
            })
            title = title.strip()
            
            # Cache the result
            self.title_cache[cache_key] = {
                'title': title,
                'timestamp': datetime.now().timestamp()
            }
            
            self.logger.info(f"Generated title: {title}")
            return title
        except Exception as e:
            self.logger.error(f"Error generating title: {str(e)}")
            raise
        
    async def generate_meta_description(self, topic: str, context: Dict, tone: str) -> str:
        """Generate meta description (max 160 chars) using Groq."""
        self.logger.info("Generating meta description")
        
        # Check cache first
        cache_key = self._generate_cache_key(topic, context, tone)
        if cache_key in self.meta_cache:
            cache_entry = self.meta_cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.logger.info("Using cached meta description")
                return cache_entry['meta']
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert SEO specialist. Generate a compelling meta description for the given topic. The description should be under 160 characters, include relevant keywords, and encourage clicks."),
            ("user", "Topic: {topic}\nTone: {tone}\nContext: {context}")
        ])
        
        try:
            chain = prompt | self.groq_client | StrOutputParser()
            description = await chain.ainvoke({
                "topic": topic,
                "tone": tone,
                "context": json.dumps(context)
            })
            description = description.strip()[:160]
            
            # Cache the result
            self.meta_cache[cache_key] = {
                'meta': description,
                'timestamp': datetime.now().timestamp()
            }
            
            self.logger.info(f"Generated meta description: {description}")
            return description
        except Exception as e:
            self.logger.error(f"Error generating meta description: {str(e)}")
            raise
        
    async def extract_keywords(self, topic: str, context: Dict, tone: str) -> List[str]:
        """Generate relevant tags/keywords using Groq."""
        self.logger.info(f"Extracting keywords for topic: {topic}")
        
        # Check cache first
        cache_key = self._generate_cache_key(topic, context, tone)
        if cache_key in self.keyword_cache:
            cache_entry = self.keyword_cache[cache_key]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.logger.info("Using cached keywords")
                return cache_entry['keywords']
        
        try:
            # Get keywords from Groq
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert SEO specialist. Extract the most relevant keywords and tags for the given topic. Return them as a comma-separated list."),
                ("user", "Topic: {topic}\nTone: {tone}\nContext: {context}")
            ])
            
            chain = prompt | self.groq_client | StrOutputParser()
            keywords = await chain.ainvoke({
                "topic": topic,
                "tone": tone,
                "context": json.dumps(context)
            })
            keywords = [k.strip() for k in keywords.split(",")]
            
            # Limit to 10 keywords
            final_keywords = keywords[:10]
            
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
        
    async def calculate_reading_time(self, content: str) -> int:
        """Estimate reading time based on word count."""
        self.logger.info("Calculating reading time")
        
        # Check cache first
        if content in self.reading_time_cache:
            cache_entry = self.reading_time_cache[content]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.logger.info("Using cached reading time")
                return cache_entry['time']
        
        words = content.split()
        # Assuming average reading speed of 200 words per minute
        reading_time = len(words) // 200
        result = max(1, reading_time)  # Minimum 1 minute
        
        # Cache the result
        self.reading_time_cache[content] = {
            'time': result,
            'timestamp': datetime.now().timestamp()
        }
        
        self.logger.info(f"Estimated reading time: {result} minutes")
        return result
        
    async def suggest_slug(self, title: str) -> str:
        """Generate URL-friendly slug from title."""
        self.logger.info(f"Generating slug for title: {title}")
        
        # Check cache first
        if title in self.slug_cache:
            cache_entry = self.slug_cache[title]
            if self._is_cache_valid(cache_entry['timestamp']):
                self.logger.info("Using cached slug")
                return cache_entry['slug']
        
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
        
        # Cache the result
        self.slug_cache[title] = {
            'slug': slug,
            'timestamp': datetime.now().timestamp()
        }
        
        self.logger.info(f"Generated slug: {slug}")
        return slug
        
    async def optimize_content(self, topic: str, context: Dict, tone: str, content: Optional[str] = None) -> Dict:
        """Run all SEO tasks concurrently."""
        self.logger.info(f"Starting SEO optimization for topic: {topic}")
        # Initialize session if needed
        await self.initialize_session()
        
        try:
            # Run all SEO tasks concurrently
            tasks = [
                self.generate_title(topic, context, tone),
                self.generate_meta_description(topic, context, tone),
                self.extract_keywords(topic, context, tone),
                self.calculate_reading_time(content) if content else asyncio.sleep(0),
            ]
            
            self.logger.info("Executing SEO tasks concurrently")
            title, meta, keywords, reading_time = await asyncio.gather(*tasks)
            slug = await self.suggest_slug(title)
            
            result = {
                "title": title,
                "meta_description": meta,
                "keywords": keywords,
                "reading_time": reading_time if content else None,
                "slug": slug
            }
            
            self.logger.info("SEO optimization completed successfully")
            self.logger.debug(f"SEO results: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during SEO optimization: {str(e)}")
            # Return a default result instead of raising the error
            return {
                "title": f"{topic} - Blog Post",
                "meta_description": f"Learn about {topic} in this informative blog post.",
                "keywords": [topic.lower()],
                "reading_time": None,
                "slug": topic.lower().replace(" ", "-")
            }
        finally:
            # Clean up session
            await self.close_session()
