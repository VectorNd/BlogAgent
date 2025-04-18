import asyncio
import logging
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks import CallbackManager

class ContentGenerationHandler(BaseCallbackHandler):
    """Callback handler for tracking content generation progress."""
    
    def __init__(self):
        self.section_count = 0
        self.total_sections = 0
    
    def set_total_sections(self, total):
        self.total_sections = total
        self.section_count = 0
    
    def on_llm_start(self, *args, **kwargs):
        self.section_count += 1
        percentage = (self.section_count / self.total_sections) * 100 if self.total_sections > 0 else 0
        print(f"Generating content... {percentage:.1f}% complete ({self.section_count}/{self.total_sections})")

class WritingAgent:
    """
    Agent responsible for generating well-structured blog content.
    
    Uses Google's Gemini API via LangChain to generate outlines,
    introductions, section content, and conclusions for blog posts.
    """
    
    def __init__(self, groq_api_key: str):
        """
        Initialize the WritingAgent with Groq API key.
        
        Args:
            groq_api_key: API key for Groq
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup LangChain cache
        set_llm_cache(InMemoryCache())
        
        # Create callback handler for tracking progress
        self.handler = ContentGenerationHandler()
        
        # Initialize Gemini model through LangChain
        self._setup_llm_models(groq_api_key)
        
        # Create prompt templates
        self._setup_prompt_templates()
    
    def _setup_llm_models(self, groq_api_key: str):
        """Set up different LLM configurations for various writing tasks."""
        # Base LLM with default settings
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     google_api_key=gemini_api_key,
        #     temperature=0.7,  # More creative for general content
        #     max_output_tokens=2048,
        #     top_p=0.95,
        #     top_k=40,
        #     callback_manager=CallbackManager([self.handler]),
        #     verbose=False
        # )


        self.llm = ChatGroq(
            model_name="gemma2-9b-it",
            temperature=0.7,
            max_tokens=2048,
            max_retries=2,
            api_key=groq_api_key,
            model_kwargs={"top_p": 0.95},
            # top_k=40,
            verbose=False,
            callback_manager=CallbackManager([self.handler])
        )
        
        # LLM for outlines (more structured, less creative)
        # self.outline_llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     google_api_key=gemini_api_key,
        #     temperature=0.4,  # Lower temperature for more structured output
        #     max_output_tokens=2048,
        #     top_p=0.95,
        #     top_k=40,
        #     verbose=False
        # )

        self.outline_llm = ChatGroq(
            model_name="gemma2-9b-it",
            temperature=0.4,
            max_tokens=2048,
            max_retries=2,
            api_key=groq_api_key,
            verbose=False,
            model_kwargs={"top_p": 0.95},
            # top_k=40
        )
        
        # self.intro_llm = ChatGoogleGenerativeAI(
        #     model="gemini-2.0-flash",
        #     google_api_key=gemini_api_key,
        #     temperature=0.8,  # Higher temperature for creative intros
        #     max_output_tokens=2048,
        #     top_p=0.95,
        #     top_k=40,
        #     verbose=False
        # )

        self.intro_llm = ChatGroq(
            model_name="gemma2-9b-it",
            temperature=0.8,
            max_tokens=2048,
            max_retries=2,
            api_key=groq_api_key,
            verbose=False,
            model_kwargs={"top_p": 0.95},
            # top_k=40
        )
    
    def _setup_prompt_templates(self):
        """Set up prompt templates for different content generation tasks."""
        # Template for generating blog outline
        self.outline_template = ChatPromptTemplate.from_template(
            """
            You are a professional blog content strategist. Create a detailed outline for a blog post on:
            
            Topic: {topic}
            
            Use the following subtopics as a starting point, but feel free to refine them or add 
            additional relevant subtopics if necessary:
            
            {subtopics_list}
            
            Additional Context:
            - Target audience knowledge level: {depth}
            - Tone: {tone}
            
            Provide a well-structured outline with H2-level headings only (no H1 or H3).
            Return only the list of H2 headings, formatted as a YAML array.
            
            Example format:
            ```yaml
            headings:
            - "First H2 Heading"
            - "Second H2 Heading"
            - "Third H2 Heading"
            ```
            """
        )
        
        # Template for writing introduction
        self.intro_template = ChatPromptTemplate.from_template(
            """
            Write an engaging introduction (approximately 100-150 words) for a blog post on:
            
            Topic: {topic}
            
            This blog will cover the following main points:
            {outline}
            
            The introduction should:
            - Hook the reader with a compelling opening
            - Clearly state what the blog post will cover
            - Establish the tone of {tone}
            - Address the reader at a {depth} knowledge level
            - Include at least one relevant keyword from this list: {keywords}
            
            Context (include if relevant):
            {context}
            
            Format the introduction with proper Markdown. Do not include any headings.
            Write approximately 100-150 words.
            """
        )
        
        # Template for writing individual sections
        self.section_template = ChatPromptTemplate.from_template(
            """
            Write content for a section of a blog post with the following H2 heading:
            
            "{heading}"
            
            This is part of a larger blog post on:
            Topic: {topic}
            
            Content specifications:
            - Length: Approximately 200-300 words
            - Knowledge level: {depth}
            - Tone: {tone}
            - Include at least two of these keywords naturally, if relevant: {keywords}
            
            Relevant context that may help with writing this section:
            {context}
            
            Format the content with proper Markdown. Include the H2 heading.
            Use bullet points, numbered lists or bold text where appropriate to improve readability.
            Write engaging, informative content that provides genuine value to the reader.
            Do not worry about transitioning to the next section.
            """
        )
        
        # Template for writing conclusion
        self.conclusion_template = ChatPromptTemplate.from_template(
            """
            Write a strong conclusion (approximately 100-150 words) for a blog post on:
            
            Topic: {topic}
            
            The blog post covered these main points:
            {outline}
            
            Your conclusion should:
            - Summarize the key takeaways
            - Include a compelling call-to-action that encourages the reader to:
              {call_to_action}
            - Maintain the {tone} tone
            - End on a positive and encouraging note
            
            Format the conclusion with proper Markdown. Start with an H2 heading "Conclusion".
            Write approximately 100-150 words.
            """
        )
    
    async def create_outline(self, topic: str, subtopics: List[str], depth: str, tone: str) -> List[str]:
        """
        Generate a blog outline with H2-level headings.
        
        Args:
            topic: The main blog topic
            subtopics: List of suggested subtopics
            depth: Target audience knowledge level (beginner/intermediate/advanced)
            tone: Content tone (educational, formal, conversational, etc.)
            
        Returns:
            List of H2 headings for the blog outline
        """
        self.logger.info(f"Creating outline for topic: '{topic}'")
        
        # Format subtopics as a bullet list
        subtopics_list = "\n".join([f"- {subtopic}" for subtopic in subtopics])
        
        # Create the chain
        outline_chain = LLMChain(
            llm=self.outline_llm,
            prompt=self.outline_template,
            verbose=False
        )
        
        try:
            # Run the chain
            response = await asyncio.to_thread(
                outline_chain.invoke,
                {
                    "topic": topic,
                    "subtopics_list": subtopics_list,
                    "depth": depth,
                    "tone": tone
                }
            )
            
            # Parse the YAML response
            import yaml
            import re
            
            result_text = response["text"]
            
            # Try to extract YAML content
            yaml_match = re.search(r'```(?:yaml)?\s*(.*?)```', result_text, re.DOTALL)
            if yaml_match:
                yaml_content = yaml_match.group(1)
                try:
                    outline_data = yaml.safe_load(yaml_content)
                    if isinstance(outline_data, dict) and "headings" in outline_data:
                        headings = outline_data["headings"]
                        if headings and isinstance(headings, list):
                            self.logger.info(f"Successfully created outline with {len(headings)} headings")
                            return headings
                except Exception as e:
                    self.logger.warning(f"Failed to parse YAML: {str(e)}")
            
            # Fallback: extract headings using regex
            self.logger.info("Using fallback method to extract headings")
            heading_matches = re.findall(r'- ["\']?(.*?)["\']?(?:\n|$)', result_text)
            
            if heading_matches:
                headings = [match.strip() for match in heading_matches if match.strip()]
                self.logger.info(f"Extracted {len(headings)} headings using regex")
                return headings
            
            # If all parsing fails, return the subtopics as headings
            self.logger.warning("Failed to parse outline response, using subtopics as headings")
            return subtopics
            
        except Exception as e:
            self.logger.error(f"Error creating outline: {str(e)}")
            # Return subtopics as fallback
            return subtopics
    
    async def write_introduction(self, topic: str, outline: List[str], depth: str, tone: str, 
                                context: Dict, keywords: List[str]) -> str:
        """
        Write an engaging introduction for the blog post.
        
        Args:
            topic: The main blog topic
            outline: List of H2 headings from the outline
            depth: Target audience knowledge level
            tone: Content tone
            context: Contextual information for the blog
            keywords: List of relevant keywords
            
        Returns:
            Markdown formatted introduction (100-150 words)
        """
        self.logger.info(f"Writing introduction for topic: '{topic}'")
        
        # Format the outline as a bullet list
        outline_text = "\n".join([f"- {heading}" for heading in outline])
        
        # Select relevant context to include
        context_text = ""
        if context.get("top_news") and len(context["top_news"]) > 0:
            news = context["top_news"][0]
            context_text += f"Recent news: {news.get('title')} - {news.get('description', '')[:100]}...\n\n"
        
        if context.get("best_quote"):
            quote = context["best_quote"]
            context_text += f"You might consider using this quote: \"{quote.get('content')}\" - {quote.get('author')}\n\n"
        
        # Format keywords for the prompt
        keywords_text = ", ".join(keywords[:5])  # Limit to 5 keywords
        
        # Create the chain
        intro_chain = LLMChain(
            llm=self.intro_llm,
            prompt=self.intro_template,
            verbose=False
        )
        
        try:
            # Run the chain
            response = await asyncio.to_thread(
                intro_chain.invoke,
                {
                    "topic": topic,
                    "outline": outline_text,
                    "depth": depth,
                    "tone": tone,
                    "context": context_text,
                    "keywords": keywords_text
                }
            )
            
            introduction = response["text"].strip()
            
            # Ensure the introduction doesn't contain headings
            import re
            introduction = re.sub(r'^\s*#.*$', '', introduction, flags=re.MULTILINE).strip()
            
            self.logger.info(f"Successfully wrote introduction ({len(introduction.split())} words)")
            return introduction
            
        except Exception as e:
            self.logger.error(f"Error writing introduction: {str(e)}")
            # Generate a simple fallback introduction
            return f"""
            Welcome to our comprehensive guide on {topic}. In this blog post, we'll explore the key aspects of this fascinating subject, including {', '.join(outline[:3])} and more. Whether you're new to this topic or looking to deepen your understanding, this article will provide valuable insights to help you navigate {topic} with confidence.
            """
    
    async def write_section(self, heading: str, topic: str, depth: str, tone: str, 
                           context: Dict, keywords: List[str]) -> str:
        """
        Write content for a single blog section.
        
        Args:
            heading: The H2 heading for this section
            topic: The main blog topic
            depth: Target audience knowledge level
            tone: Content tone
            context: Contextual information for the blog
            keywords: List of relevant keywords
            
        Returns:
            Markdown formatted section content (200-300 words)
        """
        self.logger.info(f"Writing section: '{heading}'")
        
        # Prepare context specific to this section
        section_context = ""
        
        # Add related keywords if available
        related_keywords = []
        for subtopic, keywords_list in context.get("subtopic_keywords", {}).items():
            if any(kw.lower() in heading.lower() for kw in subtopic.split()):
                related_keywords = [kw["word"] for kw in keywords_list[:5]]
                break
        
        if related_keywords:
            section_context += f"Related keywords for this section: {', '.join(related_keywords)}\n\n"
        
        # Add relevant news if available
        for news in context.get("top_news", []):
            if any(kw.lower() in news.get("title", "").lower() or kw.lower() in news.get("description", "").lower() 
                  for kw in heading.split()):
                section_context += f"Relevant news: {news.get('title')} - {news.get('description', '')[:100]}...\n\n"
                break
        
        # Add a quote if relevant to this section
        for quote in context.get("all_quotes", []):
            if any(kw.lower() in quote.get("content", "").lower() for kw in heading.split()):
                section_context += f"Relevant quote: \"{quote.get('content')}\" - {quote.get('author')}\n\n"
                break
        
        # Format keywords for the prompt (combine general keywords with section-specific ones)
        all_keywords = keywords + related_keywords
        unique_keywords = list(set(all_keywords))[:8]  # Limit to 8 unique keywords
        keywords_text = ", ".join(unique_keywords)
        
        # Create the chain
        section_chain = LLMChain(
            llm=self.llm,
            prompt=self.section_template,
            verbose=False
        )
        
        try:
            # Run the chain
            response = await asyncio.to_thread(
                section_chain.invoke,
                {
                    "heading": heading,
                    "topic": topic,
                    "depth": depth,
                    "tone": tone,
                    "context": section_context,
                    "keywords": keywords_text
                }
            )
            
            section_content = response["text"].strip()
            
            # Ensure the section has the correct heading format
            if not section_content.startswith("## "):
                section_content = f"## {heading}\n\n{section_content}"
            
            self.logger.info(f"Successfully wrote section '{heading}' ({len(section_content.split())} words)")
            return section_content
            
        except Exception as e:
            self.logger.error(f"Error writing section '{heading}': {str(e)}")
            # Generate a simple fallback section
            return f"""
            ## {heading}
            
            This section explores the important aspects of {heading} as it relates to {topic}. Understanding these concepts is crucial for anyone looking to master this subject. The key points to remember include the fundamentals of how this works, why it matters in the broader context, and practical applications you can implement right away.
            
            * The first important element is understanding the basic principles
            * Secondly, consider how these concepts apply in real-world scenarios
            * Finally, think about how you can leverage this knowledge in your own projects
            
            As you continue to explore {topic}, keep these ideas in mind to enhance your understanding and implementation.
            """
    
    async def write_conclusion(self, topic: str, outline: List[str], depth: str, tone: str, call_to_action: str = None) -> str:
        """
        Write a strong conclusion with a call-to-action.
        
        Args:
            topic: The main blog topic
            outline: List of H2 headings from the outline
            depth: Target audience knowledge level
            tone: Content tone
            call_to_action: Optional specific call-to-action direction
            
        Returns:
            Markdown formatted conclusion (100-150 words)
        """
        self.logger.info(f"Writing conclusion for topic: '{topic}'")
        
        # Format the outline as a bullet list
        outline_text = "\n".join([f"- {heading}" for heading in outline])
        
        # Default call-to-action if none provided
        if not call_to_action:
            if depth == "beginner":
                call_to_action = "continue learning about the topic and apply the basic concepts they've learned"
            elif depth == "intermediate":
                call_to_action = "implement these strategies in their own projects and dive deeper into specific areas of interest"
            else:  # advanced
                call_to_action = "contribute to the field, share their expertise, or explore cutting-edge applications of these concepts"
        
        # Create the chain
        conclusion_chain = LLMChain(
            llm=self.llm,
            prompt=self.conclusion_template,
            verbose=False
        )
        
        try:
            # Run the chain
            response = await asyncio.to_thread(
                conclusion_chain.invoke,
                {
                    "topic": topic,
                    "outline": outline_text,
                    "tone": tone,
                    "call_to_action": call_to_action
                }
            )
            
            conclusion = response["text"].strip()
            
            # Ensure the conclusion has the correct heading
            if not conclusion.lower().startswith("## conclusion"):
                conclusion = f"## Conclusion\n\n{conclusion}"
            
            self.logger.info(f"Successfully wrote conclusion ({len(conclusion.split())} words)")
            return conclusion
            
        except Exception as e:
            self.logger.error(f"Error writing conclusion: {str(e)}")
            # Generate a simple fallback conclusion
            return f"""
            ## Conclusion
            
            In this guide, we've explored the key aspects of {topic}, including {', '.join(outline[:3])}. We hope these insights will help you better understand and apply these concepts in your own context. Remember that mastering {topic} is a journey, and continuous learning is essential. We encourage you to put these ideas into practice and explore further resources to deepen your knowledge. Thank you for reading, and we wish you success in your endeavors with {topic}!
            """
    
    async def generate_full_blog(self, topic_analysis: Dict, context: Dict) -> str:
        """
        Generate a complete blog post by coordinating all writing components.
        
        Args:
            topic_analysis: Output from TopicAgent's analyze_topic method
            context: Output from ContextAgent's prepare_enriched_context method
            
        Returns:
            Complete markdown formatted blog post
        """
        topic = topic_analysis["topic"]
        depth = topic_analysis["knowledge_depth"]
        tone = topic_analysis["requested_tone"]
        subtopics = topic_analysis["subtopics"]
        
        self.logger.info(f"Generating full blog for topic: '{topic}'")
        
        # Step 1: Create the outline
        outline = await self.create_outline(topic, subtopics, depth, tone)
        
        # Calculate total sections for progress tracking
        total_sections = len(outline) + 2  # Sections + intro + conclusion
        self.handler.set_total_sections(total_sections)
        
        # Step 2: Write the introduction
        introduction = await self.write_introduction(
            topic, 
            outline, 
            depth, 
            tone, 
            context, 
            context.get("top_keywords", [])
        )
        
        # Step 3: Write each section concurrently
        section_tasks = [
            self.write_section(
                heading, 
                topic, 
                depth, 
                tone, 
                context, 
                context.get("top_keywords", [])
            ) 
            for heading in outline
        ]
        
        sections = await asyncio.gather(*section_tasks)
        
        # Step 4: Write the conclusion
        conclusion = await self.write_conclusion(topic, outline, depth, tone)
        
        # Step 5: Assemble the full blog post
        blog_title = f"# {topic}\n\n"
        
        full_blog = blog_title + introduction + "\n\n"
        
        for section in sections:
            full_blog += section + "\n\n"
        
        full_blog += conclusion
        
        self.logger.info(f"Successfully generated full blog ({len(full_blog.split())} words)")
        return full_blog
    
    async def generate_blog_with_title(self, topic_analysis: Dict, context: Dict, title: str = None) -> Dict:
        """
        Generate a complete blog post with metadata.
        
        Args:
            topic_analysis: Output from TopicAgent's analyze_topic method
            context: Output from ContextAgent's prepare_enriched_context method
            title: Optional custom title for the blog
            
        Returns:
            Dictionary with blog content and metadata
        """
        # Generate the blog content
        blog_content = await self.generate_full_blog(topic_analysis, context)
        
        # Use provided title or generate from topic
        if not title:
            title = f"Complete Guide to {topic_analysis['topic']}"
        
        # Calculate reading time
        word_count = len(blog_content.split())
        reading_time = round(word_count / 200, 1)  # Assuming 200 words per minute
        
        # Return blog with metadata
        return {
            "title": title,
            "content": blog_content,
            "word_count": word_count,
            "reading_time": reading_time,
            "topic": topic_analysis["topic"],
            "depth": topic_analysis["knowledge_depth"],
            "tone": topic_analysis["requested_tone"]
        }