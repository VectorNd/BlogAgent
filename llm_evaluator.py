from typing import Dict, Any
import textstat
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import json
import numpy as np
from datetime import datetime
import os
load_dotenv()

class BlogEvaluator:
    def __init__(self):
        self.metrics = {
            'readability': self._evaluate_readability,
            'seo': self._evaluate_seo,
            'structure': self._evaluate_structure,
            'grammar': self._evaluate_grammar,
            'engagement': self._evaluate_engagement,
            'coverage': self._evaluate_coverage,
            'originality': self._evaluate_originality
        }
        
    def _evaluate_readability(self, text: str) -> float:
        """Evaluate readability using various text statistics."""
        scores = {
            'flesch_reading_ease': textstat.flesch_reading_ease(text),
            'smog_index': textstat.smog_index(text),
            'coleman_liau_index': textstat.coleman_liau_index(text),
            'automated_readability_index': textstat.automated_readability_index(text)
        }
        # Normalize scores to 0-1 range
        normalized_scores = {
            'flesch': min(max((scores['flesch_reading_ease'] - 0) / (100 - 0), 0), 1),
            'smog': min(max((20 - scores['smog_index']) / (20 - 0), 0), 1),
            'coleman': min(max((20 - scores['coleman_liau_index']) / (20 - 0), 0), 1),
            'automated': min(max((20 - scores['automated_readability_index']) / (20 - 0), 0), 1)
        }
        return np.mean(list(normalized_scores.values()))

    def _evaluate_seo(self, text: str) -> float:
        """Evaluate SEO optimization."""
        # Basic SEO metrics
        word_count = len(text.split())
        paragraph_count = len(text.split('\n\n'))
        avg_paragraph_length = word_count / max(paragraph_count, 1)
        
        # Check for common SEO elements
        has_headers = any(header in text.lower() for header in ['h1:', 'h2:', 'h3:'])
        has_meta = 'meta description' in text.lower()
        
        # Score calculation
        score = 0
        if 300 <= word_count <= 2000:
            score += 0.3
        if 3 <= paragraph_count <= 10:
            score += 0.2
        if 50 <= avg_paragraph_length <= 200:
            score += 0.2
        if has_headers:
            score += 0.15
        if has_meta:
            score += 0.15
            
        return score

    def _evaluate_structure(self, text: str) -> float:
        """Evaluate content structure and coherence."""
        # Basic structure checks
        has_intro = 'introduction' in text.lower()[:200]
        has_conclusion = 'conclusion' in text.lower()[-200:]
        has_transitions = any(word in text.lower() for word in ['however', 'therefore', 'moreover', 'furthermore'])
        
        score = 0
        if has_intro:
            score += 0.3
        if has_conclusion:
            score += 0.3
        if has_transitions:
            score += 0.4
            
        return score

    def _evaluate_grammar(self, text: str) -> float:
        """Evaluate grammar and syntax."""
        # Using textstat's sentence count and syllable count as proxies
        sentence_count = textstat.sentence_count(text)
        syllable_count = textstat.syllable_count(text)
        
        # Basic grammar checks
        has_punctuation = any(p in text for p in ['.', '!', '?'])
        has_capitalization = any(c.isupper() for c in text)
        
        score = 0
        if sentence_count > 0:
            score += 0.3
        if syllable_count > 0:
            score += 0.3
        if has_punctuation:
            score += 0.2
        if has_capitalization:
            score += 0.2
            
        return score

    def _evaluate_engagement(self, text: str) -> float:
        """Evaluate engagement potential."""
        # Check for engagement elements
        has_questions = any(text.strip().endswith('?') for text in text.split('\n'))
        has_lists = any(text.strip().startswith(('-', '*', '1.', '2.', '3.')) for text in text.split('\n'))
        has_quotes = '"' in text or "'" in text
        
        score = 0
        if has_questions:
            score += 0.4
        if has_lists:
            score += 0.3
        if has_quotes:
            score += 0.3
            
        return score

    def _evaluate_coverage(self, text: str) -> float:
        """Evaluate topic coverage."""
        # Using word count and paragraph count as proxies
        word_count = len(text.split())
        paragraph_count = len(text.split('\n\n'))
        
        score = 0
        if word_count >= 500:
            score += 0.5
        if paragraph_count >= 3:
            score += 0.5
            
        return score

    def _evaluate_originality(self, text: str) -> float:
        """Evaluate content originality."""
        # Using text length and vocabulary diversity as proxies
        words = text.split()
        unique_words = set(words)
        vocabulary_ratio = len(unique_words) / len(words) if words else 0
        
        score = 0
        if len(words) >= 300:
            score += 0.5
        if vocabulary_ratio >= 0.5:
            score += 0.5
            
        return score

    def evaluate_blog(self, text: str) -> Dict[str, float]:
        """Evaluate a blog post on all metrics."""
        results = {}
        # print(text)
        # print(self.metrics.items())
        for metric_name, metric_func in self.metrics.items():
            results[metric_name] = metric_func(text)
        return results

class LLMEvaluator:
    def __init__(self):
        self.blog_evaluator = BlogEvaluator()
        self.models = {
            # 'mixtral-8x7b-32768': ChatGroq(
            #     model_name="mixtral-8x7b-32768",
            #     temperature=0.7,
            #     max_tokens=2048,
            #     max_retries=2,
            #     api_key=os.getenv("GROQ_API_KEY")
            # ),
            # 'llama3-70b-8192': ChatGroq(
            #     model_name="llama3-70b-8192",
            #     temperature=0.7,
            #     max_tokens=2048,
            #     max_retries=2,
            #     api_key=os.getenv("GROQ_API_KEY")
            # ),
            # 'llama-3.3-70b-versatile': ChatGroq(
            #     model_name="llama-3.3-70b-versatile",
            #     temperature=0.7,
            #     max_tokens=2048,
            #     max_retries=2,
            #     api_key=os.getenv("GROQ_API_KEY")
            # ),
            # 'llama3-8b-8192': ChatGroq(
            #     model_name="llama3-8b-8192",
            #     temperature=0.5,
            #     max_tokens=2048,
            #     max_retries=2,
            #     api_key=os.getenv("GROQ_API_KEY")
            # ),
            # 'deepseek-r1-distill-llama-70b': ChatGroq(
            #     model_name="deepseek-r1-distill-llama-70b",
            #     temperature=0.5,
            #     max_tokens=2048,
            #     max_retries=2,
            #     api_key=os.getenv("GROQ_API_KEY")
            # ),
            # 
            # 'gemma2-9b-it': ChatGroq(
            #     model_name="gemma2-9b-it",
            #     temperature=0.5,
            #     max_tokens=2048,
            #     max_retries=2,
            #     api_key=os.getenv("GROQ_API_KEY")
            # ),
            'gemini-flash': ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.5,
                max_output_tokens=2048,
                verbose=False
            ),
            # 'gemini-pro': ChatGoogleGenerativeAI(
            #     model="gemini-1.5-pro",
            #     google_api_key=os.getenv("GEMINI_API_KEY"),
            #     temperature=0.5,
            #     max_output_tokens=2048,
            #     verbose=False
            # ) 
        }
        
    async def generate_blog(self, model, topic: str) -> str:
        """Generate a blog post using the specified model."""
        prompt = f"""Write a comprehensive blog post about {topic}. 
        The blog should be well-structured, engaging, and optimized for SEO. 
        Include an introduction, main points, and a conclusion."""
        
        response = await model.agenerate([[HumanMessage(content=prompt)]])
        return response.generations[0][0].text

    async def evaluate_models(self, topic: str, num_runs: int = 3) -> Dict[str, Any]:
        """Evaluate all models on the given topic."""
        results = {}
        
        for model_name, model in self.models.items():
            model_results = []
            for _ in range(num_runs):
                blog = await self.generate_blog(model, topic)
                evaluation = self.blog_evaluator.evaluate_blog(blog)
                model_results.append(evaluation)
            
            # Calculate average scores
            avg_scores = {}
            for metric in self.blog_evaluator.metrics.keys():
                scores = [run[metric] for run in model_results]
                avg_scores[metric] = np.mean(scores)
            
            results[model_name] = {
                'average_scores': avg_scores,
                'runs': model_results
            }
        
        return results

    def save_results(self, results: Dict[str, Any], topic: str):
        """Save evaluation results to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_evaluation_{topic.replace(' ', '_')}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filename

async def main():
    evaluator = LLMEvaluator()
    topic = "The Future of Artificial Intelligence in Healthcare"
    
    print(f"Starting evaluation for topic: {topic}")
    results = await evaluator.evaluate_models(topic)
    
    filename = evaluator.save_results(results, topic)
    print(f"Evaluation results saved to {filename}")
    
    # Print summary of results
    print("\nEvaluation Summary:")
    for model, data in results.items():
        print(f"\n{model}:")
        for metric, score in data['average_scores'].items():
            print(f"  {metric}: {score:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 