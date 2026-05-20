"""Base dataset class for medical report generation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset, ABC):
    """Abstract base class for medical report datasets."""

    def __init__(
        self,
        data_path: str,
        phase: str = "train",
        transform: Optional[Any] = None,
    ):
        """Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            phase: Dataset phase ('train', 'valid', 'test')
            transform: Image transformations
        """
        self.data_path = data_path
        self.phase = phase
        self.transform = transform
        self.data = self.load_data()

    @abstractmethod
    def load_data(self) -> List[Dict]:
        """Load data from files.
        
        Returns:
            List of data dictionaries
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Get item at index.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with sample data
        """
        pass

    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples
        """
        return len(self.data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "n_samples": len(self.data),
            "phase": self.phase,
        }

