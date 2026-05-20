"""Text processing utility functions."""

import re
from typing import List


def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and stripping whitespace.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    return text.lower().strip()


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Handle common medical report format
    if text.endswith("."):
        sentences = text.split(".")[:-1]
    else:
        sentences = text.split(".")
    
    # Clean each sentence
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: Whether to keep basic punctuation
        
    Returns:
        Cleaned text
    """
    if keep_punctuation:
        # Keep alphanumeric, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,;:\-]', '', text)
    else:
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    return normalize_whitespace(text)


def join_sentences(sentences: List[str]) -> str:
    """Join sentences into a single text.
    
    Args:
        sentences: List of sentences
        
    Returns:
        Joined text
    """
    # Ensure each sentence ends with a period
    sentences = [s.strip() for s in sentences]
    sentences = [s if s.endswith('.') else s + '.' for s in sentences]
    
    return " ".join(sentences)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def count_words(text: str) -> int:
    """Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())

