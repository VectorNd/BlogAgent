import re
from typing import List, Dict
import textstat

class QualityMetrics:
    """Utility class for calculating various content quality metrics."""
    
    @staticmethod
    def calculate_readability(content: str) -> float:
        """
        Calculate content readability score using multiple metrics.
        
        Args:
            content: The content to analyze
            
        Returns:
            Normalized readability score between 0 and 1
        """
        # Calculate various readability scores
        flesch_score = textstat.flesch_reading_ease(content)
        grade_level = textstat.flesch_kincaid_grade(content)
        smog_index = textstat.smog_index(content)
        
        # Normalize scores to 0-1 range
        # Flesch score: 0-100 (higher is better)
        # Grade level: 1-12 (lower is better)
        # SMOG index: 1-20 (lower is better)
        
        normalized_flesch = flesch_score / 100
        normalized_grade = 1 - (min(grade_level, 12) / 12)
        normalized_smog = 1 - (min(smog_index, 20) / 20)
        
        # Weighted average of normalized scores
        readability_score = (
            0.5 * normalized_flesch +
            0.3 * normalized_grade +
            0.2 * normalized_smog
        )
        
        return max(0, min(1, readability_score))
    
    @staticmethod
    def calculate_keyword_density(content: str, keywords: List[str]) -> Dict[str, float]:
        """
        Calculate keyword density for each keyword.
        
        Args:
            content: The content to analyze
            keywords: List of keywords to check
            
        Returns:
            Dictionary of keyword densities
        """
        # Convert content to lowercase and split into words
        words = re.findall(r'\w+', content.lower())
        total_words = len(words)
        
        # Calculate density for each keyword
        densities = {}
        for keyword in keywords:
            # Count occurrences of the keyword
            keyword_count = sum(1 for word in words if word == keyword.lower())
            # Calculate density as percentage
            density = (keyword_count / total_words) * 100 if total_words > 0 else 0
            densities[keyword] = density
            
        return densities
    
    @staticmethod
    def calculate_structure_score(content: str, subtopics: List[str]) -> float:
        """
        Calculate content structure score based on heading hierarchy and subtopic coverage.
        
        Args:
            content: The content to analyze
            subtopics: List of expected subtopics
            
        Returns:
            Structure score between 0 and 1
        """
        # Check for proper heading hierarchy
        h2_count = len(re.findall(r'^##\s', content, re.MULTILINE))
        h3_count = len(re.findall(r'^###\s', content, re.MULTILINE))
        
        # Score for heading hierarchy (0-0.5)
        heading_score = 0
        if h2_count >= len(subtopics):
            heading_score += 0.3
        if h3_count > 0:
            heading_score += 0.2
            
        # Score for subtopic coverage (0-0.5)
        coverage_score = 0
        for subtopic in subtopics:
            if subtopic.lower() in content.lower():
                coverage_score += 0.5 / len(subtopics)
                
        return heading_score + coverage_score
    
    @staticmethod
    def calculate_content_length_score(content: str, min_words: int = 500, max_words: int = 2000) -> float:
        """
        Calculate score based on content length.
        
        Args:
            content: The content to analyze
            min_words: Minimum acceptable word count
            max_words: Maximum acceptable word count
            
        Returns:
            Length score between 0 and 1
        """
        word_count = len(content.split())
        
        if word_count < min_words:
            return 0
        elif word_count > max_words:
            return 1
        else:
            return (word_count - min_words) / (max_words - min_words)
    
    @staticmethod
    def calculate_overall_quality(content: str, keywords: List[str], subtopics: List[str]) -> Dict[str, float]:
        """
        Calculate all quality metrics and return a comprehensive score.
        
        Args:
            content: The content to analyze
            keywords: List of keywords to check
            subtopics: List of expected subtopics
            
        Returns:
            Dictionary of all quality metrics
        """
        return {
            "readability": QualityMetrics.calculate_readability(content),
            "keyword_density": QualityMetrics.calculate_keyword_density(content, keywords),
            "structure": QualityMetrics.calculate_structure_score(content, subtopics),
            "length": QualityMetrics.calculate_content_length_score(content)
        } 