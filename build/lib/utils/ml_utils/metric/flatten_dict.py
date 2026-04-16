# exception class import
from exception.exception import ProjectError

# library import
import sys
from typing import Dict

def flatten_dict(d: Dict, parent_key='', sep='/'):
    
    """
    
    """
    
    try:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # Recursively flatten
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    except Exception as e:
        raise ProjectError(e, sys)