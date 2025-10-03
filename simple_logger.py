#!/usr/bin/env python3
"""
Simple logger for training metrics and model configuration.
"""

import os
import json
import csv
import time


class SimpleLogger:
    """Simple logger for training metrics and model configuration."""
    
    def __init__(self, model_name, config, args):
        self.model_name = model_name
        self.config = config
        self.args = args
        self.timestamp = int(time.time())
        
        # Create logs directory
        self.logs_dir = "training_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Create log files
        self.log_file = os.path.join(self.logs_dir, f"{model_name}_{self.timestamp}.json")
        self.csv_file = os.path.join(self.logs_dir, f"{model_name}_{self.timestamp}.csv")
        
        # Initialize log data
        self.log_data = {
            "model_name": model_name,
            "config": config,
            "args": vars(args),
            "timestamp": self.timestamp,
            "epochs": []
        }
        
        # Initialize CSV file
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
    
    def log_epoch(self, epoch, train_loss, val_loss, learning_rate):
        """Log metrics for one epoch."""
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": learning_rate
        }
        
        # Add to JSON log
        self.log_data["epochs"].append(epoch_data)
        
        # Add to CSV log
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, learning_rate])
    
    def save_log(self):
        """Save the complete log to JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
    
    def get_summary(self):
        """Get training summary."""
        if not self.log_data["epochs"]:
            return {}
        
        epochs = self.log_data["epochs"]
        return {
            "total_epochs": len(epochs),
            "final_train_loss": epochs[-1]["train_loss"],
            "final_val_loss": epochs[-1]["val_loss"],
            "best_val_loss": min(epoch["val_loss"] for epoch in epochs),
            "best_epoch": min(epochs, key=lambda x: x["val_loss"])["epoch"]
        }
