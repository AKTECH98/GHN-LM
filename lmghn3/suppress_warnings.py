"""
Warning suppression module for GHN-3 training.
This module suppresses common warnings that appear during training.
"""

import warnings
import os

def suppress_all_warnings():
    """Suppress all common warnings during GHN-3 training."""
    
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Suppress specific warning categories
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress specific PyTorch warnings
    warnings.filterwarnings('ignore', message='enable_nested_tensor is True, but self.use_nested_tensor is False')
    warnings.filterwarnings('ignore', message='.*autocast.*is deprecated.*')
    
    # Suppress transformers warnings
    warnings.filterwarnings('ignore', module='transformers')
    warnings.filterwarnings('ignore', module='datasets')
    
    # Suppress PyTorch warnings
    warnings.filterwarnings('ignore', module='torch')
    
    # Set environment variables to suppress additional warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['PYTHONWARNINGS'] = 'ignore'

def suppress_training_warnings():
    """Suppress warnings specifically for training."""
    suppress_all_warnings()
    
    # Additional training-specific suppressions
    warnings.filterwarnings('ignore', message='.*parameter.*not matched.*')
    warnings.filterwarnings('ignore', message='.*ERROR.*NOT MATCHED.*')

# Auto-suppress warnings when this module is imported
suppress_all_warnings()
