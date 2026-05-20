"""Cluster sign (positive/negative) determination using LLMs."""

import os
import time
import random
from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np


class ClusterSignDeterminer:
    """Determine if clusters represent positive (normal) or negative (abnormal) findings."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "deepseek",
        model: str = "deepseek-chat",
        temperature: float = 0.9,
    ):
        """Initialize sign determiner.
        
        Args:
            api_key: API key for LLM provider
            provider: LLM provider ('deepseek', 'openai', 'anthropic')
            model: Model name to use
            temperature: Temperature for sampling (default 0.9 as in original)
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        
        if self.provider == "deepseek":
            try:
                import requests
                self.requests = requests
                self.api_url = "https://api.deepseek.com/v1/chat/completions"
            except ImportError:
                raise ImportError("requests package not installed. Install with: pip install requests")
        elif self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {self.provider}. Choose from: 'deepseek', 'openai', 'anthropic'")

    def determine_signs(
        self,
        cluster_to_texts: Dict[int, List[str]],
    ) -> Dict[int, str]:
        """Determine signs for all clusters.
        
        Args:
            cluster_to_texts: Dictionary mapping cluster ID to list of texts
            
        Returns:
            Dictionary mapping cluster ID to sign ('positive' or 'negative')
        """
        cluster_signs = {}
        cluster_ids = sorted(cluster_to_texts.keys())
        
        print(f"Determining signs for {len(cluster_ids)} clusters using {self.provider}...")
        
        for i, cluster_id in enumerate(cluster_ids):
            if i % 10 == 0:
                print(f"Processing cluster {i+1}/{len(cluster_ids)}...")
            
            texts = cluster_to_texts[cluster_id]
            
            # Sample one random text from cluster (as in original implementation)
            finding = random.choice(texts)
            
            # Determine sign
            sign = self._classify_finding(finding)
            cluster_signs[cluster_id] = sign
            
            # Rate limiting
            time.sleep(0.1)
        
        # Print summary
        n_positive = sum(1 for s in cluster_signs.values() if s == "positive")
        n_negative = sum(1 for s in cluster_signs.values() if s == "negative")
        
        print("\nSign determination summary:")
        print(f"  Positive: {n_positive}")
        print(f"  Negative: {n_negative}")
        
        return cluster_signs

    def _classify_finding(self, finding: str) -> str:
        """Classify a single finding as positive or negative.
        
        Args:
            finding: Medical finding text
            
        Returns:
            Sign: 'positive' or 'negative'
        """
        prompt = self._create_prompt(finding)
        
        try:
            if self.provider == "deepseek":
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                }
                
                response = self.requests.post(self.api_url, headers=headers, json=data)
                
                if response.status_code == 200:
                    result = response.json()["choices"][0]["message"]["content"].strip().lower()
                else:
                    print(f"DeepSeek API Error: {response.status_code}, {response.text}")
                    return "negative"  # Default to negative on error
            
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a medical expert analyzing radiology reports."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=10,
                )
                result = response.choices[0].message.content.strip().lower()
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                )
                result = response.content[0].text.strip().lower()
            
            # Parse result - only positive or negative
            if "positive" in result:
                return "positive"
            elif "negative" in result:
                return "negative"
            else:
                # If unclear, default to negative (conservative approach)
                print(f"Warning: Unclear response '{result}' for finding, defaulting to 'negative'")
                return "negative"
        
        except Exception as e:
            print(f"Error classifying finding: {e}")
            return "negative"  # Default to negative on error

    def _create_prompt(self, finding: str) -> str:
        """Create prompt for LLM classification (matching original implementation).
        
        Args:
            finding: Medical finding text
            
        Returns:
            Prompt string
        """
        # Original prompt from the repository
        prompt = f"""Evaluate whether the following sentence is positive or not. Any medical finding rather a normal finding should be considered as negative finding.

- If it includes any medical finding rather than a normal finding of a completely healthy person, respond with 'negative'.
- If this finding seems like a finding of a healthy person, respond with 'positive'.

Provide only 'positive' or 'negative' as the output, with no explanation.

The finding is: {finding}"""
        
        return prompt

    def determine_signs_rule_based(
        self,
        cluster_to_texts: Dict[int, List[str]],
    ) -> Dict[int, str]:
        """Determine signs using rule-based approach (no API needed).
        
        This is faster but less accurate than LLM-based approach.
        
        Args:
            cluster_to_texts: Dictionary mapping cluster ID to list of texts
            
        Returns:
            Dictionary mapping cluster ID to sign ('positive' or 'negative')
        """
        # Keywords for positive (normal) findings
        positive_keywords = [
            "normal", "clear", "unremarkable", "no", "within normal limits",
            "stable", "unchanged", "intact", "preserved",
            "no evidence", "no significant", "no acute", "negative for"
        ]
        
        # Keywords for negative (abnormal) findings
        negative_keywords = [
            "abnormal", "pathology", "lesion", "mass", "effusion", "fracture",
            "consolidation", "opacity", "infiltrate", "edema", "hemorrhage",
            "abnormality", "concerning", "suspicious", "significant",
            "enlarged", "thickening", "disease", "infection", "mild", "moderate",
            "severe", "acute", "chronic", "findings", "demonstrate"
        ]
        
        cluster_signs = {}
        
        for cluster_id, texts in cluster_to_texts.items():
            # Sample one random text (matching LLM approach)
            finding = random.choice(texts)
            finding_lower = finding.lower()
            
            # Count keyword occurrences
            positive_count = sum(1 for keyword in positive_keywords if keyword in finding_lower)
            negative_count = sum(1 for keyword in negative_keywords if keyword in finding_lower)
            
            # Classify based on counts
            # If more negative keywords or equal, classify as negative (conservative)
            if positive_count > negative_count:
                cluster_signs[cluster_id] = "positive"
            else:
                cluster_signs[cluster_id] = "negative"
        
        # Print summary
        n_positive = sum(1 for s in cluster_signs.values() if s == "positive")
        n_negative = sum(1 for s in cluster_signs.values() if s == "negative")
        
        print("\nRule-based sign determination summary:")
        print(f"  Positive: {n_positive}")
        print(f"  Negative: {n_negative}")
        
        return cluster_signs

    def get_cluster_examples(
        self,
        cluster_to_texts: Dict[int, List[str]],
        cluster_signs: Dict[int, str],
        n_examples: int = 5,
        sign_filter: Optional[str] = None,
    ) -> Dict[int, List[str]]:
        """Get example texts for clusters.
        
        Args:
            cluster_to_texts: Dictionary mapping cluster ID to list of texts
            cluster_signs: Dictionary mapping cluster ID to sign
            n_examples: Number of examples per cluster
            sign_filter: Only include clusters with this sign ('positive' or 'negative')
            
        Returns:
            Dictionary mapping cluster ID to example texts
        """
        examples = {}
        
        for cluster_id, texts in cluster_to_texts.items():
            sign = cluster_signs.get(cluster_id, "negative")
            
            if sign_filter and sign != sign_filter:
                continue
            
            # Sample texts
            if len(texts) > n_examples:
                sample_indices = np.random.choice(len(texts), n_examples, replace=False)
                sample_texts = [texts[idx] for idx in sample_indices]
            else:
                sample_texts = texts
            
            examples[cluster_id] = sample_texts
        
        return examples

