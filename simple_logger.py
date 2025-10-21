#!/usr/bin/env python3
"""
Simple logger for model configuration and metadata.
"""

import os
import json
import time


class SimpleLogger:
    """Simple logger for model configuration and metadata."""
    
    def __init__(self, model_name, config, args):
        self.model_name = model_name
        self.config = config
        self.args = args
        self.timestamp = int(time.time())
        
        # Create logs directory (same structure as GHN training)
        self.logs_dir = "logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create log file
        self.log_file = os.path.join(self.logs_dir, f"{model_name}_{self.timestamp}.json")
        
        # Initialize log data
        self.log_data = {
            "model_name": model_name,
            "config": config,
            "args": vars(args),
            "timestamp": self.timestamp
        }
    
    def save_log(self):
        """Save the log to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def get_summary(self):
        """Get model configuration summary."""
        return {
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "config": self.config,
            "args": vars(self.args) if hasattr(self.args, '__dict__') else self.args
        }
