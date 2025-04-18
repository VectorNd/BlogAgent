import asyncio
import functools
import json
import logging
import time
from typing import Dict, List

import aiohttp
from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

class ContextAgent:
    """
    Agent responsible for gathering contextual information for blog topics.
    
    Uses various APIs to collect news, semantic keywords, and quotes related to a topic.
    """
    
    def __init__(self, newsdata_api_key: str, groq_api_key: str):
        """
        Initialize the ContextAgent with necessary API keys.
        
        Args:
            newsdata_api_key: API key for NewsData.io
            groq_api_key: API key for Groq (used for context processing)
        """
        self.logger = logging.getLogger(__name__)
        self.newsdata_api_key = newsdata_api_key
        
        # Initialize LangChain LLM
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     google_api_key=gemini_api_key,
        #     temperature=0.2,
        #     max_output_tokens=2048,
        #     verbose=False
        # )

        self.llm = ChatGroq(
            model="gemma2-9b-it",
            api_key=groq_api_key,
            temperature=0.2,
            max_tokens=2048,
            verbose=False
        )
        
        # Initialize session for async HTTP requests
        self.session = None
        
        # Caches for API responses
        self.news_cache = {}
        self.keywords_cache = {}
        self.quotes_cache = {}
    
    async def initialize_session(self):
        """Initialize aiohttp session if not already created."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session if it exists."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    @functools.lru_cache(maxsize=100)
    async def fetch_relevant_news(self, topic: str, max_results: int = 5) -> List[Dict]:
        """
        Fetch relevant news articles from NewsData.io API.
        
        Args:
            topic: The topic to search for
            max_results: Maximum number of news articles to return
            
        Returns:
            List of news article dictionaries
        """
        self.logger.info(f"Fetching news for topic: '{topic}'")
        
        # Check cache first
        cache_key = f"news_{topic}_{max_results}"
        if cache_key in self.news_cache:
            self.logger.info(f"Using cached news for '{topic}'")
            return self.news_cache[cache_key]
        
        # Ensure session is initialized
        await self.initialize_session()
        
        # Prepare request URL with parameters
        url = "https://newsdata.io/api/1/news"
        params = {
            "apikey": self.newsdata_api_key,
            "q": topic,
            "language": "en",
            "size": max_results
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"NewsData API error ({response.status}): {error_text}")
                    return []
                
                data = await response.json()
                
                if data.get("status") != "success":
                    self.logger.error(f"NewsData API returned error: {data.get('results', {})}")
                    return []
                
                # Extract and simplify relevant information
                articles = []
                for article in data.get("results", [])[:max_results]:
                    articles.append({
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "content": article.get("content", ""),
                        "link": article.get("link", ""),
                        "source_name": article.get("source_name", ""),
                        "pubDate": article.get("pubDate", "")
                    })
                
                # Cache the results
                self.news_cache[cache_key] = articles
                
                self.logger.info(f"Found {len(articles)} news articles for '{topic}'")
                return articles
                
        except Exception as e:
            self.logger.error(f"Error fetching news for '{topic}': {str(e)}")
            return []
    
    @functools.lru_cache(maxsize=200)
    async def get_semantic_keywords(self, seed_keyword: str, relation_type: str = "ml", max_results: int = 15) -> List[Dict]:
        """
        Generate semantically related keywords using LLM.
        
        Args:
            seed_keyword: The base keyword to find variations for
            relation_type: Type of relation ('ml'=means like, 'rel_trg'=triggers, etc.)
            max_results: Maximum number of keywords to return
            
        Returns:
            List of related keywords with scores
        """
        self.logger.info(f"Generating semantic keywords for: '{seed_keyword}'")
        
        # Check cache first
        cache_key = f"keywords_{seed_keyword}_{relation_type}_{max_results}"
        if cache_key in self.keywords_cache:
            self.logger.info(f"Using cached keywords for '{seed_keyword}'")
            return self.keywords_cache[cache_key]
        
        try:
            # Map relation types to natural language descriptions
            relation_description = {
                "ml": "words that mean similar things to",
                "rel_trg": "words that are triggered by or associated with",
                "rel_syn": "synonyms of",
                "rel_ant": "antonyms of",
                "rel_spc": "more specific examples of",
                "rel_gen": "more general categories that include"
            }.get(relation_type, "related to")
            
            # Create prompt for keyword generation
            prompt = ChatPromptTemplate.from_template(
                """
                Generate {max_results} keywords that are {relation_description} "{seed_keyword}".
                Each keyword should be relevant and semantically connected to the seed word.
                For each keyword, also provide:
                1. A relevance score from 0 to 100
                2. A list of tags describing the relationship
                
                Format the response as a JSON array with the following structure:
                [
                    {{
                        "word": "the keyword",
                        "score": relevance_score,
                        "tags": ["relevant", "tags"]
                    }},
                    ...
                ]
                
                Make sure the keywords are diverse and cover different aspects of the topic.
                """
            )
            
            # Run the chain
            chain = prompt | self.llm
            response = await asyncio.to_thread(
                chain.invoke,
                {
                    "seed_keyword": seed_keyword,
                    "relation_description": relation_description,
                    "max_results": max_results
                }
            )
            
            # Parse JSON from the response
            response_text = response.content
            
            # Extract JSON if embedded in text
            import re
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            try:
                keywords = json.loads(response_text)
                # Ensure we don't return more than max_results
                keywords = keywords[:max_results]
                
                # Cache the results
                self.keywords_cache[cache_key] = keywords
                
                self.logger.info(f"Generated {len(keywords)} semantic keywords for '{seed_keyword}'")
                return keywords
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON from LLM response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error generating semantic keywords for '{seed_keyword}': {str(e)}")
            return []
    
    @functools.lru_cache(maxsize=100)
    async def fetch_relevant_quotes(self, keyword: str, max_results: int = 3) -> List[Dict]:
        """
        Generate relevant quotes using LLM for the given keyword.
        
        Args:
            keyword: The keyword/topic to generate quotes for
            max_results: Maximum number of quotes to return
            
        Returns:
            List of quote dictionaries
        """
        self.logger.info(f"Generating quotes for keyword: '{keyword}'")
        
        # Check cache first
        cache_key = f"quotes_{keyword}_{max_results}"
        if cache_key in self.quotes_cache:
            self.logger.info(f"Using cached quotes for '{keyword}'")
            return self.quotes_cache[cache_key]
        
        try:
            # Create prompt for quote generation
            prompt = ChatPromptTemplate.from_template(
                """
                Generate {max_results} relevant and inspiring quotes about "{keyword}".
                Each quote should be unique, meaningful, and related to the topic.
                Include a relevant author for each quote.
                Format the response as a JSON array with the following structure:
                [
                    {{
                        "content": "the quote text",
                        "author": "author name",
                        "tags": ["relevant", "tags"]
                    }},
                    ...
                ]
                """
            )
            
            # Run the chain
            chain = prompt | self.llm
            response = await asyncio.to_thread(
                chain.invoke,
                {"keyword": keyword, "max_results": max_results}
            )
            
            # Parse JSON from the response
            response_text = response.content
            
            # Extract JSON if embedded in text
            import re
            json_match = re.search(r'(\[.*\])', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            
            try:
                quotes = json.loads(response_text)
                # Ensure we don't return more than max_results
                quotes = quotes[:max_results]
                
                # Cache the results
                self.quotes_cache[cache_key] = quotes
                
                self.logger.info(f"Generated {len(quotes)} quotes for '{keyword}'")
                return quotes
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON from LLM response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error generating quotes for '{keyword}': {str(e)}")
            return []
    
    async def gather_context(self, topic: str, subtopics: List[str]) -> Dict:
        """
        Gather comprehensive context for a blog topic using multiple APIs.
        
        Args:
            topic: The main blog topic
            subtopics: List of subtopics from topic analysis
            
        Returns:
            Dictionary containing all contextual information
        """
        self.logger.info(f"Gathering context for topic: '{topic}' with {len(subtopics)} subtopics")
        
        # Create tasks for parallel execution
        tasks = [
            self.fetch_relevant_news(topic, max_results=5),
        ]
        
        # Add tasks for each subtopic's semantic keywords
        keyword_tasks = [self.get_semantic_keywords(subtopic) for subtopic in subtopics]
        tasks.extend(keyword_tasks)
        
        # Add task for main topic semantic keywords
        tasks.append(self.get_semantic_keywords(topic, max_results=20))
        
        # Add task for quotes
        tasks.append(self.fetch_relevant_quotes(topic))
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Extract results
        news_articles = results[0]
        subtopic_keywords = results[1:len(subtopics)+1]
        main_keywords = results[len(subtopics)+1]
        quotes = results[len(subtopics)+2]
        
        # Process keywords to remove duplicates and organize by subtopic
        all_keywords = set()
        keyword_mapping = {}
        
        # Process main topic keywords
        main_topic_keywords = []
        for keyword in main_keywords:
            if keyword["word"] not in all_keywords:
                all_keywords.add(keyword["word"])
                main_topic_keywords.append(keyword)
        
        # Process subtopic keywords
        for i, subtopic in enumerate(subtopics):
            subtopic_kw = []
            for keyword in subtopic_keywords[i]:
                if keyword["word"] not in all_keywords:
                    all_keywords.add(keyword["word"])
                    subtopic_kw.append(keyword)
            keyword_mapping[subtopic] = subtopic_kw
        
        # Organize results
        context = {
            "topic": topic,
            "news_articles": news_articles,
            "main_keywords": main_topic_keywords,
            "subtopic_keywords": keyword_mapping,
            "quotes": quotes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return context
    
    async def analyze_context_relevance(self, topic: str, gathered_context: Dict) -> Dict:
        """
        Analyze the gathered context to determine most relevant elements for the blog.
        
        Args:
            topic: The main blog topic
            gathered_context: The output from gather_context()
            
        Returns:
            Dictionary containing analysis of context relevance
        """
        self.logger.info(f"Analyzing context relevance for topic: '{topic}'")
        
        # Create LangChain prompt for context analysis
        prompt = ChatPromptTemplate.from_template(
            """
            Analyze the following data gathered for a blog post on "{topic}".
            
            I need you to identify:
            1. The 3 most relevant news articles from this list:
            {news_list}
            
            2. The 10 most relevant SEO keywords from this list:
            {keyword_list}
            
            3. The most impactful quote to include:
            {quote_list}
            
            4. Identify any content gaps or additional research needed based on what you see.
            
            Return your analysis as a JSON object with these keys:
            - top_news (array of indices)
            - top_keywords (array of strings)
            - best_quote (index number)
            - content_gaps (string describing any missing information)
            """
        )
        
        # Format input data for the prompt
        news_list = "\n".join([f"{i}. {article['title']} - {article['description'][:100]}..." 
                              for i, article in enumerate(gathered_context['news_articles'])])
        
        keyword_list = "\n".join([f"{i}. {kw['word']} (score: {kw['score']})" 
                                 for i, kw in enumerate(gathered_context['main_keywords'][:20])])
        
        quote_list = "\n".join([f"{i}. \"{quote['content']}\" - {quote['author']}" 
                               for i, quote in enumerate(gathered_context['quotes'])])
        
        # If any lists are empty, provide placeholders
        if not news_list:
            news_list = "No news articles found."
        if not keyword_list:
            keyword_list = "No keywords found."
        if not quote_list:
            quote_list = "No quotes found."
        
        # Run the chain
        try:
            chain = prompt | self.llm
            response = await asyncio.to_thread(
                chain.invoke,
                {"topic": topic, "news_list": news_list, "keyword_list": keyword_list, "quote_list": quote_list}
            )
            
            # Parse JSON from the response
            response_text = response.content
            
            # Extract JSON if embedded in text
            import re
            json_match = re.search(r'({.*})', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
                
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                self.logger.error("Failed to parse JSON from LLM response")
                # Provide default analysis
                analysis = {
                    "top_news": list(range(min(3, len(gathered_context['news_articles'])))),
                    "top_keywords": [kw["word"] for kw in gathered_context['main_keywords'][:10]],
                    "best_quote": 0 if gathered_context['quotes'] else None,
                    "content_gaps": "Consider gathering more specific information related to recent developments."
                }
                
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing context relevance: {str(e)}")
            # Provide default analysis on error
            return {
                "top_news": list(range(min(3, len(gathered_context['news_articles'])))),
                "top_keywords": [kw["word"] for kw in gathered_context['main_keywords'][:10]],
                "best_quote": 0 if gathered_context['quotes'] else None,
                "content_gaps": "Error in analysis. Consider manual review of gathered context."
            }
    
    async def prepare_enriched_context(self, topic: str, subtopics: List[str]) -> Dict:
        """
        Gather and analyze context to prepare enriched, blog-ready contextual information.
        
        Args:
            topic: The main blog topic
            subtopics: List of subtopics from topic analysis
            
        Returns:
            Dictionary with fully processed and analyzed context
        """
        try:
            # First gather raw context
            raw_context = await self.gather_context(topic, subtopics)
            
            # Then analyze for relevance
            relevance = await self.analyze_context_relevance(topic, raw_context)
            
            # Filter and organize the most relevant information
            enriched_context = {
                "topic": topic,
                "top_news": [raw_context["news_articles"][i] for i in relevance["top_news"] 
                            if i < len(raw_context["news_articles"])],
                "top_keywords": relevance["top_keywords"],
                "subtopic_keywords": raw_context["subtopic_keywords"],
                "best_quote": raw_context["quotes"][relevance["best_quote"]] if relevance["best_quote"] is not None 
                              and relevance["best_quote"] < len(raw_context["quotes"]) else None,
                "all_quotes": raw_context["quotes"],
                "content_gaps": relevance["content_gaps"],
                "timestamp": raw_context["timestamp"]
            }
            
            return enriched_context
            
        except Exception as e:
            self.logger.error(f"Error preparing enriched context: {str(e)}")
            # Return minimally structured context on error
            return {
                "topic": topic,
                "top_news": [],
                "top_keywords": [],
                "subtopic_keywords": {},
                "best_quote": None,
                "all_quotes": [],
                "content_gaps": f"Error gathering context: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }