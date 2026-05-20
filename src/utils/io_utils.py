"""Input/output utility functions."""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Union


def load_json(filepath: Union[str, Path]) -> Any:
    """Load data from a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save JSON file
        indent: Indentation for pretty printing
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)
    
    print(f"Saved JSON to {filepath}")


def load_npz(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load data from a .npz file.
    
    Args:
        filepath: Path to .npz file
        
    Returns:
        Dictionary of arrays
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.files}


def save_npz(data: Dict[str, np.ndarray], filepath: Union[str, Path]):
    """Save arrays to a .npz file.
    
    Args:
        data: Dictionary of arrays
        filepath: Path to save .npz file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(filepath, **data)
    print(f"Saved NPZ to {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """Load data from a pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data: Any, filepath: Union[str, Path]):
    """Save data to a pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save pickle file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved pickle to {filepath}")


def ensure_dir(dirpath: Union[str, Path]) -> Path:
    """Ensure a directory exists.
    
    Args:
        dirpath: Path to directory
        
    Returns:
        Path object
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def list_files(
    dirpath: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """List files in a directory.
    
    Args:
        dirpath: Path to directory
        pattern: Glob pattern for file matching
        recursive: Whether to search recursively
        
    Returns:
        List of file paths
    """
    dirpath = Path(dirpath)
    
    if recursive:
        files = list(dirpath.rglob(pattern))
    else:
        files = list(dirpath.glob(pattern))
    
    return [f for f in files if f.is_file()]

