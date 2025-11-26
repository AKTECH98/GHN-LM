#!/usr/bin/env python3
"""
Evaluation script to extract metrics from completed training runs.

This script calculates three key metrics:
1. Perplexity at set intervals (from TensorBoard logs or checkpoints)
2. Convergence analysis (when validation loss stopped improving)
3. Test dataset score on the best model

Usage:
    python evaluate_metrics.py --config LM/configs/benchmark_1_tiny.yaml
    python evaluate_metrics.py --config LM/configs/benchmark_1_tiny.yaml --intervals 1,2,5,10
    python evaluate_metrics.py --config LM/configs/benchmark_1_tiny.yaml --init_method default
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from Dataloader.wikitext2_loader import build_wikitext2
from LM.create_model import create_model

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("⚠️  TensorBoard library not available. Will use checkpoints for perplexity extraction.")


def find_experiment_dirs_by_init_method(config_name: str) -> Dict[str, Path]:
    """
    Find experiment directories matching the config name, grouped by init method.
    
    Args:
        config_name: Base config name (e.g., "benchmark_1_tiny")
        
    Returns:
        Dictionary mapping init_method -> experiment_dir (most recent for each method)
    """
    experiment_base = Path("Experiment")
    if not experiment_base.exists():
        return {}
    
    # Normalize config name for matching (handle case variations)
    config_name_lower = config_name.lower()
    
    # Extract the part after "benchmark_" (e.g., "1_tiny" from "benchmark_1_tiny")
    if config_name_lower.startswith("benchmark_"):
        base_name = config_name_lower.replace("benchmark_", "", 1)
    else:
        base_name = config_name_lower
    
    # Search all patterns for all init methods
    patterns = {
        "default": f"Benchmark_{base_name}*",
        "ghn": f"GHN-T_{base_name}*",
        "ghn-i": f"GHN-I_{base_name}*"
    }
    
    results = {}
    for init_method, pattern in patterns.items():
        matches = list(experiment_base.glob(pattern))
        if matches:
            # Sort by modification time (newest first) and take the most recent
            matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
            results[init_method] = matches[0]
    
    return results


def find_tensorboard_dir(job_id: str) -> Optional[Path]:
    """Find TensorBoard log directory for a given job_id."""
    tensor_log_base = Path("Final_tensors")
    if not tensor_log_base.exists():
        return None
    
    tensor_log_dir = tensor_log_base / job_id
    if tensor_log_dir.exists():
        return tensor_log_dir
    
    return None


class MetricsEvaluator:
    """Evaluates training metrics from existing experiment data."""
    
    def __init__(self, config_file: Path, experiment_dir: Path, device: str = "cuda"):
        """
        Initialize the evaluator for a specific experiment directory.
        
        Args:
            config_file: Path to YAML config file (e.g., LM/configs/benchmark_1_tiny.yaml)
            experiment_dir: Path to specific experiment directory
            device: Device to run evaluation on
        """
        self.config_file = Path(config_file)
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Load YAML config
        with open(self.config_file, 'r') as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Extract config name from file path
        self.config_name = self.config_file.stem
        
        # Load experiment config.json
        config_json_path = self.experiment_dir / "config.json"
        if not config_json_path.exists():
            raise FileNotFoundError(f"Config JSON not found: {config_json_path}")
        
        with open(config_json_path, 'r') as f:
            self.config = json.load(f)
        
        self.job_id = self.config.get("job_id", self.experiment_dir.name)
        
        # Get convergence parameters from YAML config (same as training)
        training_config = self.yaml_config.get("training", {})
        self.default_convergence_patience = training_config.get("early_stopping_patience", 3)
        self.default_convergence_threshold = training_config.get("early_stopping_min_delta", 0.001)
        
        # Find TensorBoard log directory
        self.tensorboard_dir = find_tensorboard_dir(self.job_id)
        
        # Detect init method from experiment directory name
        exp_name = self.experiment_dir.name
        if exp_name.startswith("GHN-T_"):
            self.init_method = "ghn"
        elif exp_name.startswith("GHN-I_"):
            self.init_method = "ghn-i"
        elif exp_name.startswith("Benchmark_"):
            self.init_method = "default"
        else:
            self.init_method = "unknown"
    
    def extract_perplexity_from_tensorboard(self, target_epochs: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Extract perplexity values at target epochs from TensorBoard logs.
        
        Args:
            target_epochs: List of epochs to extract (e.g., [1, 2, 5, 10])
            
        Returns:
            Dictionary mapping epoch -> {train_perplexity, val_perplexity}
        """
        results = {}
        
        if not TENSORBOARD_AVAILABLE or self.tensorboard_dir is None:
            print("   ⚠️  TensorBoard not available or log dir not found. Cannot extract perplexity.")
            return results
        
        try:
            # Find event file
            event_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
            if not event_files:
                print("   ⚠️  No TensorBoard event files found. Cannot extract perplexity.")
                return results
            
            # Use the first event file
            event_file = event_files[0]
            print(f"   📊 Reading TensorBoard logs from: {event_file}")
            
            # Load event accumulator
            ea = EventAccumulator(str(self.tensorboard_dir))
            ea.Reload()
            
            # Get available scalar tags
            scalar_tags = ea.Tags()['scalars']
            
            # Extract train and val perplexity
            train_perplexity_tag = 'Epoch/Train_Perplexity'
            val_perplexity_tag = 'Epoch/Val_Perplexity'
            
            train_perplexities = {}
            val_perplexities = {}
            
            if train_perplexity_tag in scalar_tags:
                train_scalars = ea.Scalars(train_perplexity_tag)
                for scalar in train_scalars:
                    epoch = int(scalar.step) + 1  # TensorBoard uses 0-indexed, we use 1-indexed
                    train_perplexities[epoch] = scalar.value
            
            if val_perplexity_tag in scalar_tags:
                val_scalars = ea.Scalars(val_perplexity_tag)
                for scalar in val_scalars:
                    epoch = int(scalar.step) + 1
                    val_perplexities[epoch] = scalar.value
            
            # Extract values for target epochs
            for epoch in target_epochs:
                results[epoch] = {
                    "train_perplexity": train_perplexities.get(epoch, None),
                    "val_perplexity": val_perplexities.get(epoch, None)
                }
            
            print(f"   ✅ Extracted perplexity for {len(results)} epochs from TensorBoard")
            
        except Exception as e:
            print(f"   ⚠️  Error reading TensorBoard logs: {e}")
            print(f"   ❌ Cannot extract perplexity without TensorBoard logs.")
            return results
        
        return results
    
    def extract_perplexity_from_checkpoints(self, target_epochs: List[int]) -> Dict[int, Dict[str, float]]:
        """
        Extract perplexity values from checkpoint files.
        
        Args:
            target_epochs: List of epochs to extract
            
        Returns:
            Dictionary mapping epoch -> {train_perplexity, val_perplexity}
        """
        results = {}
        
        # Find all checkpoint files
        checkpoint_pattern = str(self.experiment_dir / "epoch_*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        print(f"   📦 Found {len(checkpoint_files)} checkpoint files")
        
        # Load checkpoints and extract losses
        for checkpoint_path in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                epoch = checkpoint.get('epoch', -1) + 1  # Convert to 1-indexed
                
                if epoch in target_epochs:
                    val_loss = checkpoint.get('val_loss', None)
                    if val_loss is not None:
                        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                        results[epoch] = {
                            "train_perplexity": None,  # Not stored in checkpoints
                            "val_perplexity": val_perplexity
                        }
            except Exception as e:
                print(f"   ⚠️  Error loading checkpoint {checkpoint_path}: {e}")
                continue
        
        # Also check best_model.pt
        best_model_path = self.experiment_dir / "best_model.pt"
        if best_model_path.exists():
            try:
                checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
                epoch = checkpoint.get('epoch', -1) + 1
                
                if epoch in target_epochs:
                    val_loss = checkpoint.get('val_loss', None)
                    if val_loss is not None:
                        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
                        if epoch not in results:
                            results[epoch] = {}
                        results[epoch]["val_perplexity"] = val_perplexity
            except Exception as e:
                print(f"   ⚠️  Error loading best model: {e}")
        
        print(f"   ✅ Extracted perplexity for {len(results)} epochs from checkpoints")
        return results
    
    def analyze_convergence(self, convergence_threshold: float = 0.0001, 
                           convergence_patience: int = 5) -> Dict:
        """
        Analyze when the model converged based on validation loss.
        
        Args:
            convergence_threshold: Minimum improvement to count as progress
            convergence_patience: Number of epochs without improvement to consider converged
            
        Returns:
            Dictionary with convergence metrics
        """
        print(f"\n🔍 Analyzing convergence...")
        print(f"   Threshold: {convergence_threshold}, Patience: {convergence_patience}")
        
        # Get validation loss history from TensorBoard only
        val_losses = []
        
        if not TENSORBOARD_AVAILABLE or self.tensorboard_dir is None:
            print("   ⚠️  TensorBoard not available or log dir not found. Cannot analyze convergence.")
            return {
                "converged": False,
                "convergence_epoch": None,
                "convergence_loss": None,
                "convergence_perplexity": None,
                "epochs_to_convergence": None
            }
        
        try:
            event_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
            if not event_files:
                print("   ⚠️  No TensorBoard event files found. Cannot analyze convergence.")
                return {
                    "converged": False,
                    "convergence_epoch": None,
                    "convergence_loss": None,
                    "convergence_perplexity": None,
                    "epochs_to_convergence": None
                }
            
            ea = EventAccumulator(str(self.tensorboard_dir))
            ea.Reload()
            
            val_loss_tag = 'Epoch/Val_Loss'
            if val_loss_tag in ea.Tags()['scalars']:
                val_scalars = ea.Scalars(val_loss_tag)
                for scalar in val_scalars:
                    epoch = int(scalar.step) + 1
                    val_losses.append((epoch, scalar.value))
                val_losses.sort(key=lambda x: x[0])
        except Exception as e:
            print(f"   ⚠️  Error reading TensorBoard for convergence: {e}")
            print("   ❌ Cannot analyze convergence without TensorBoard logs.")
            return {
                "converged": False,
                "convergence_epoch": None,
                "convergence_loss": None,
                "convergence_perplexity": None,
                "epochs_to_convergence": None
            }
        
        if not val_losses:
            print("   ⚠️  No validation loss data found in TensorBoard logs")
            return {
                "converged": False,
                "convergence_epoch": None,
                "convergence_loss": None,
                "convergence_perplexity": None,
                "epochs_to_convergence": None
            }
        
        # Analyze convergence
        best_loss = float('inf')
        best_epoch = None
        no_improvement_count = 0
        convergence_epoch = None
        
        for epoch, val_loss in val_losses:
            if val_loss < best_loss - convergence_threshold:
                best_loss = val_loss
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= convergence_patience and convergence_epoch is None:
                    convergence_epoch = epoch - convergence_patience
                    break
        
        # If never converged, use last epoch
        if convergence_epoch is None:
            convergence_epoch = val_losses[-1][0]
            best_loss = min(loss for _, loss in val_losses)
            best_epoch = min(epoch for epoch, loss in val_losses if loss == best_loss)
        
        convergence_perplexity = torch.exp(torch.tensor(best_loss)).item()
        
        result = {
            "converged": True,
            "convergence_epoch": convergence_epoch,
            "convergence_loss": float(best_loss),
            "convergence_perplexity": float(convergence_perplexity),
            "epochs_to_convergence": convergence_epoch,
            "best_epoch": best_epoch,
            "total_epochs": val_losses[-1][0] if val_losses else None
        }
        
        print(f"   ✅ Convergence detected at epoch {convergence_epoch}")
        print(f"      Best loss: {best_loss:.4f}, Perplexity: {convergence_perplexity:.2f}")
        
        return result
    
    def evaluate_test_dataset(self) -> Dict:
        """
        Evaluate the best model on the test dataset.
        
        Returns:
            Dictionary with test evaluation metrics
        """
        print(f"\n🧪 Evaluating best model on test dataset...")
        
        # Load best model checkpoint
        best_model_path = self.experiment_dir / "best_model.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_model_path}")
        
        print(f"   Loading best model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
        
        # Recreate model from YAML config
        model_config_dict = self.yaml_config["model"]
        from Dataloader.config_loader import ModelConfig
        model_config = ModelConfig(**model_config_dict)
        
        vocab_size = model_config_dict.get("vocab_size", 50257)
        model = create_model(model_config, vocab_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"   Model loaded: {model_config.model_type}")
        print(f"   Best model epoch: {checkpoint.get('epoch', -1) + 1}")
        print(f"   Best model val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # Load test data from YAML config
        data_config_dict = self.yaml_config["data"]
        training_config_dict = self.yaml_config["training"]
        
        print(f"   Loading test data...")
        data = build_wikitext2(
            tokenizer_name="gpt2",
            seq_len=data_config_dict["seq_len"],
            batch_size=training_config_dict["batch_size"],
            num_workers=data_config_dict["num_workers"],
            cache_dir="./data"
        )
        
        test_loader = data['test_loader']
        print(f"   Test batches: {len(test_loader)}")
        
        # Evaluate on test set
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating test set"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                try:
                    if hasattr(model, 'forward') and 'targets' in model.forward.__code__.co_varnames:
                        output = model(input_ids, targets=labels)
                        if isinstance(output, tuple):
                            logits, loss = output[:2]
                        else:
                            logits = output
                            loss = None
                    else:
                        output = model(input_ids)
                        if isinstance(output, tuple):
                            logits, loss = output[:2]
                        else:
                            logits = output
                            loss = None
                    
                    # Compute loss if not provided
                    if loss is None:
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        # If loss is per-token, multiply by number of tokens
                        num_valid_tokens = (labels != -100).sum().item()
                        loss = loss * num_valid_tokens
                    
                    total_loss += loss.item()
                    total_tokens += (labels != -100).sum().item()
                    num_batches += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Error in batch evaluation: {e}")
                    continue
        
        if total_tokens == 0:
            raise ValueError("No valid tokens found in test set")
        
        avg_loss = total_loss / total_tokens
        test_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        result = {
            "test_loss": float(avg_loss),
            "test_perplexity": float(test_perplexity),
            "total_tokens": int(total_tokens),
            "num_batches": num_batches
        }
        
        print(f"   ✅ Test evaluation complete:")
        print(f"      Test Loss: {avg_loss:.4f}")
        print(f"      Test Perplexity: {test_perplexity:.2f}")
        print(f"      Total Tokens: {total_tokens:,}")
        
        return result
    
    def evaluate_all_metrics(self, target_epochs: List[int] = None, 
                           convergence_threshold: Optional[float] = None,
                           convergence_patience: Optional[int] = None) -> Dict:
        """
        Evaluate all three metrics.
        
        Args:
            target_epochs: Epochs to extract perplexity for (default: [1, 2, 5, 10, 20, 50])
            convergence_threshold: Minimum improvement for convergence detection (default: from config)
            convergence_patience: Epochs without improvement to consider converged (default: from config)
            
        Returns:
            Dictionary with all metrics
        """
        if target_epochs is None:
            target_epochs = [1, 2, 5, 10, 20, 50]
        
        # Use config values if not provided
        if convergence_patience is None:
            convergence_patience = self.default_convergence_patience
        if convergence_threshold is None:
            convergence_threshold = self.default_convergence_threshold
        
        print(f"\n{'='*60}")
        print(f"📊 Evaluating All Metrics")
        print(f"{'='*60}")
        
        # 1. Perplexity at intervals
        print(f"\n1️⃣  Extracting Perplexity at Intervals")
        print(f"   Target epochs: {target_epochs}")
        perplexity_data = self.extract_perplexity_from_tensorboard(target_epochs)
        
        # Format for output
        perplexity_intervals = []
        for epoch in sorted(perplexity_data.keys()):
            perplexity_intervals.append({
                "epoch": epoch,
                "train_perplexity": perplexity_data[epoch].get("train_perplexity"),
                "val_perplexity": perplexity_data[epoch].get("val_perplexity")
            })
        
        # 2. Convergence analysis
        print(f"\n2️⃣  Analyzing Convergence")
        convergence_data = self.analyze_convergence(convergence_threshold, convergence_patience)
        
        # 3. Test evaluation
        print(f"\n3️⃣  Evaluating on Test Dataset")
        test_data = self.evaluate_test_dataset()
        
        # Get best model info
        best_model_path = self.experiment_dir / "best_model.pt"
        best_model_info = {}
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
            best_epoch = checkpoint.get('epoch', -1) + 1
            best_val_loss = checkpoint.get('val_loss', None)
            best_model_info = {
                "epoch": best_epoch,
                "val_loss": float(best_val_loss) if best_val_loss is not None else None,
                "val_perplexity": float(torch.exp(torch.tensor(best_val_loss)).item()) if best_val_loss is not None else None
            }
        
        # Compile all results
        results = {
            "config_file": str(self.config_file),
            "config_name": self.config_name,
            "init_method": self.init_method,
            "job_id": self.job_id,
            "experiment_dir": str(self.experiment_dir),
            "perplexity_at_intervals": perplexity_intervals,
            "convergence": convergence_data,
            "test_evaluation": test_data,
            "best_model": best_model_info
        }
        
        return results
    
    def save_metrics(self, metrics: Dict, output_path: Optional[Path] = None):
        """Save metrics to JSON file."""
        if output_path is None:
            output_path = self.experiment_dir / "metrics.json"
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n💾 Metrics saved to: {output_path}")
        return output_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate metrics from completed training runs")
    
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML config file (e.g., LM/configs/benchmark_1_tiny.yaml)")
    parser.add_argument("--intervals", type=str, default="1,2,5,10,20,50",
                       help="Comma-separated list of epochs to extract perplexity (default: 1,2,5,10,20,50)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation (default: cuda)")
    
    args = parser.parse_args()
    
    # Parse intervals
    target_epochs = [int(x.strip()) for x in args.intervals.split(",")]
    
    # Load config file
    config_file = Path(args.config)
    if not config_file.exists():
        parser.error(f"Config file not found: {config_file}")
    
    config_name = config_file.stem
    
    # Find all experiment directories for all init methods
    print(f"📋 Finding experiment directories for config: {config_name}")
    experiment_dirs = find_experiment_dirs_by_init_method(config_name)
    
    if not experiment_dirs:
        parser.error(
            f"No experiment directories found for config: {config_name}\n"
            f"  Searched for patterns matching: {config_name}\n"
            f"  In directory: Experiment/"
        )
    
    print(f"   Found experiments for {len(experiment_dirs)} init methods:")
    for init_method, exp_dir in experiment_dirs.items():
        print(f"      {init_method}: {exp_dir.name}")
    
    # Load YAML config once to get convergence defaults
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
    training_config = yaml_config.get("training", {})
    convergence_patience = training_config.get("early_stopping_patience", 3)
    convergence_threshold = training_config.get("early_stopping_min_delta", 0.001)
    
    # Evaluate metrics for each init method
    all_results = {}
    
    for init_method, experiment_dir in experiment_dirs.items():
        print(f"\n{'='*60}")
        print(f"📊 Evaluating {init_method.upper()} Init Method")
        print(f"{'='*60}")
        
        # Create evaluator for this init method
        evaluator = MetricsEvaluator(
            config_file=config_file,
            experiment_dir=experiment_dir,
            device=args.device
        )
        
        # Evaluate all metrics
        metrics = evaluator.evaluate_all_metrics(
            target_epochs=target_epochs,
            convergence_threshold=convergence_threshold,
            convergence_patience=convergence_patience
        )
        
        # Store results (don't save individual files)
        all_results[init_method] = metrics
    
    # Print summary for all init methods
    print(f"\n{'='*60}")
    print(f"📊 Summary for All Init Methods")
    print(f"{'='*60}")
    print(f"Config: {config_name}\n")
    
    # Print comparison table for perplexity at intervals
    print("Perplexity at Intervals:")
    print(f"{'Epoch':<8} {'Default':<12} {'GHN-T':<12} {'GHN-I':<12}")
    print("-" * 50)
    
    # Get all epochs from all results
    all_epochs = set()
    for results in all_results.values():
        for entry in results['perplexity_at_intervals']:
            all_epochs.add(entry['epoch'])
    
    for epoch in sorted(all_epochs):
        row = [f"{epoch}"]
        for init_method in ["default", "ghn", "ghn-i"]:
            if init_method in all_results:
                val_ppl = None
                for entry in all_results[init_method]['perplexity_at_intervals']:
                    if entry['epoch'] == epoch:
                        val_ppl = entry['val_perplexity']
                        break
                if val_ppl is not None:
                    row.append(f"{val_ppl:.2f}")
                else:
                    row.append("N/A")
            else:
                row.append("N/A")
        print(f"{row[0]:<8} {row[1]:<12} {row[2]:<12} {row[3]:<12}")
    
    # Print convergence comparison
    print(f"\nConvergence:")
    print(f"{'Init Method':<15} {'Converged':<12} {'Epoch':<8} {'Perplexity':<12}")
    print("-" * 50)
    for init_method in ["default", "ghn", "ghn-i"]:
        if init_method in all_results:
            conv = all_results[init_method]['convergence']
            epoch_str = str(conv['convergence_epoch']) if conv['convergence_epoch'] is not None else "N/A"
            perplexity_str = f"{conv['convergence_perplexity']:.2f}" if conv['convergence_perplexity'] is not None else "N/A"
            print(f"{init_method:<15} {str(conv['converged']):<12} {epoch_str:<8} {perplexity_str:<12}")
    
    # Print test evaluation comparison
    print(f"\nTest Evaluation:")
    print(f"{'Init Method':<15} {'Test Loss':<12} {'Test Perplexity':<15}")
    print("-" * 45)
    for init_method in ["default", "ghn", "ghn-i"]:
        if init_method in all_results:
            test = all_results[init_method]['test_evaluation']
            print(f"{init_method:<15} {test['test_loss']:.4f}      {test['test_perplexity']:.2f}")
    
    # Save combined results in Evaluations folder
    evaluations_dir = Path("Evaluations")
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    combined_output = evaluations_dir / f"{config_name}_all_metrics.json"
    
    with open(combined_output, 'w') as f:
        json.dump({
            "config_file": str(config_file),
            "config_name": config_name,
            "results_by_init_method": all_results
        }, f, indent=2)
    print(f"\n💾 Combined metrics saved to: {combined_output}")


if __name__ == "__main__":
    main()
