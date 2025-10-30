#!/usr/bin/env python3
"""
Checkpoint Evaluation Script

This script evaluates language models at different checkpoints for both benchmark
and GHN-initialized experiments. It loads models from different epochs and computes
test loss on WikiText-2 validation data.

Usage:
    python evaluate_checkpoints.py --config benchmark_1_tiny --experiment_dir Experiment/
    
    # Evaluate specific epochs
    python evaluate_checkpoints.py --config benchmark_1_tiny --epochs 2,5,10 --experiment_dir Experiment/
    
    # Compare benchmark vs GHN init
    python evaluate_checkpoints.py --config benchmark_1_tiny --compare --experiment_dir Experiment/
"""

import argparse
import os
import sys
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Dataloader.config_loader import load_config_file, list_benchmark_configs
from Dataloader.wikitext2_loader import build_wikitext2
from LM.create_model import create_model
from GHN.nn import from_pretrained


class CheckpointEvaluator:
    """Evaluates models at different checkpoints."""
    
    def __init__(self, config_name: str, experiment_dir: str = "Experiment/", device: str = "cuda"):
        """
        Initialize the evaluator.
        
        Args:
            config_name: Name of the benchmark config (e.g., 'benchmark_1_tiny')
            experiment_dir: Path to the Experiment directory (default: "Experiment/")
            device: Device to run evaluation on
        """
        self.config_name = config_name
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device(device)
        
        # Load configuration
        config_path = f"LM/configs/{config_name}.yaml"
        self.model_config, self.training_config, self.data_config = load_config_file(config_path)
        
        # Auto-discover experiment directories based on config name
        self.benchmark_dir, self.ghn_init_dir = self._discover_experiment_dirs()
        
        print(f"📋 Configuration: {config_name}")
        print(f"   Model: {self.model_config.model_type}")
        print(f"   Epochs: {self.training_config.epochs}")
        print(f"   Device: {self.device}")
        print(f"   Benchmark dir: {self.benchmark_dir}")
        print(f"   GHN init dir: {self.ghn_init_dir}")
        
    def _discover_experiment_dirs(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Auto-discover experiment directories based on config name.
        
        Returns:
            Tuple of (benchmark_dir, ghn_init_dir) - either can be None if not found
        """
        import glob
        
        # Extract the config number and type from config name
        # e.g., "benchmark_1_tiny" -> "1_tiny"
        config_parts = self.config_name.split('_')
        if len(config_parts) >= 3:
            config_suffix = '_'.join(config_parts[1:])  # "1_tiny"
        else:
            config_suffix = self.config_name
            
        # Look for benchmark directory
        benchmark_pattern = str(self.experiment_dir / f"Benchmark_{config_suffix}_*")
        benchmark_dirs = glob.glob(benchmark_pattern)
        
        # Look for GHN init directory  
        ghn_pattern = str(self.experiment_dir / f"GHN_init_{config_suffix}_*")
        ghn_dirs = glob.glob(ghn_pattern)
        
        # Select the most recent directories (highest timestamp)
        benchmark_dir = None
        if benchmark_dirs:
            benchmark_dir = Path(max(benchmark_dirs, key=lambda x: int(x.split('_')[-1])))
            
        ghn_init_dir = None
        if ghn_dirs:
            ghn_init_dir = Path(max(ghn_dirs, key=lambda x: int(x.split('_')[-1])))
        
        return benchmark_dir, ghn_init_dir
        
    def load_test_data(self) -> Dict:
        """Load WikiText-2 test data."""
        print(f"\n📚 Loading WikiText-2 test data...")
        
        data = build_wikitext2(
            tokenizer_name="gpt2",
            seq_len=self.data_config.seq_len,
            batch_size=self.training_config.batch_size,
            num_workers=self.data_config.num_workers,
            cache_dir="./data"
        )
        
        print(f"   Vocab size: {data['vocab_size']}")
        print(f"   Test batches: {len(data['test_loader'])}")
        
        return data
    
    def find_checkpoints(self, experiment_dir: Path) -> List[Tuple[int, Path]]:
        """
        Find all available checkpoints in an experiment directory.
        
        Returns:
            List of (epoch, checkpoint_path) tuples sorted by epoch
        """
        checkpoints = []
        
        if not experiment_dir.exists():
            print(f"⚠️  Experiment directory not found: {experiment_dir}")
            return checkpoints
        
        # Find epoch checkpoints (epoch_X.pt)
        epoch_pattern = str(experiment_dir / "epoch_*.pt")
        epoch_files = glob.glob(epoch_pattern)
        
        for file_path in epoch_files:
            filename = os.path.basename(file_path)
            try:
                epoch = int(filename.replace("epoch_", "").replace(".pt", ""))
                checkpoints.append((epoch, Path(file_path)))
            except ValueError:
                continue
        
        # Add best model checkpoint if it exists
        best_model_path = experiment_dir / "best_model.pt"
        if best_model_path.exists():
            # Load the checkpoint to get the epoch
            try:
                checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
                best_epoch = checkpoint.get('epoch', -1)
                checkpoints.append((best_epoch, best_model_path))
            except Exception as e:
                print(f"⚠️  Could not load best model checkpoint: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints
    
    def load_model_checkpoint(self, checkpoint_path: Path, is_ghn_init: bool = False) -> torch.nn.Module:
        """
        Load a model from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            is_ghn_init: Whether this is a GHN-initialized model
            
        Returns:
            Loaded model
        """
        print(f"   Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Create model
            model = create_model(self.model_config, vocab_size=50257)
            model = model.to(self.device)
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("No model state dict found in checkpoint")
            
            model.eval()
            
            # Print checkpoint info
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('val_loss', 'unknown')
            print(f"     Epoch: {epoch}, Val Loss: {val_loss}")
            
            return model
            
        except Exception as e:
            print(f"❌ Error loading checkpoint {checkpoint_path}: {e}")
            raise
    
    def evaluate_model(self, model: torch.nn.Module, test_loader) -> Dict[str, float]:
        """
        Evaluate a model on test data.
        
        Args:
            model: The model to evaluate
            test_loader: Test data loader
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                # Move batch to device
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
                    
                    # Compute loss if not provided (match training exactly)
                    if loss is None:
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            labels.view(-1),
                            ignore_index=-100,
                            reduction='mean'  # Match training loss calculation
                        )
                    
                    # Count total tokens for averaging (including ignored ones, like training)
                    total_tokens += labels.numel()
                    
                    total_loss += loss.item() * labels.numel()  # Scale by total tokens
                    num_batches += 1
                    
                except Exception as e:
                    print(f"⚠️  Error in batch evaluation: {e}")
                    continue
        
        if num_batches == 0:
            return {"test_loss": float('inf'), "perplexity": float('inf')}
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "test_loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_batches": num_batches
        }
    
    def evaluate_experiment(self, experiment_dir: Path, experiment_name: str, 
                          target_epochs: Optional[List[int]] = None) -> Dict:
        """
        Evaluate all checkpoints in an experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory
            experiment_name: Name of the experiment (for logging)
            target_epochs: Specific epochs to evaluate (None for all)
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n🔍 Evaluating {experiment_name} experiment...")
        
        # Find checkpoints
        checkpoints = self.find_checkpoints(experiment_dir)
        
        if not checkpoints:
            print(f"❌ No checkpoints found in {experiment_dir}")
            return {}
        
        print(f"   Found {len(checkpoints)} checkpoints")
        
        # Filter by target epochs if specified
        if target_epochs is not None:
            checkpoints = [(epoch, path) for epoch, path in checkpoints if epoch in target_epochs]
            print(f"   Filtered to {len(checkpoints)} checkpoints for epochs: {target_epochs}")
        
        # Load test data
        test_data = self.load_test_data()
        test_loader = test_data['test_loader']
        
        # Evaluate each checkpoint
        results = {}
        for epoch, checkpoint_path in checkpoints:
            print(f"\n   📊 Evaluating epoch {epoch}...")
            
            try:
                # Load model
                model = self.load_model_checkpoint(checkpoint_path, is_ghn_init="GHN_init" in experiment_name)
                
                # Evaluate
                metrics = self.evaluate_model(model, test_loader)
                results[epoch] = metrics
                
                print(f"     Test Loss: {metrics['test_loss']:.4f}")
                print(f"     Perplexity: {metrics['perplexity']:.2f}")
                
                # Clean up
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"❌ Error evaluating epoch {epoch}: {e}")
                results[epoch] = {"error": str(e)}
        
        return results
    
    def compare_experiments(self, target_epochs: Optional[List[int]] = None) -> Dict:
        """
        Compare benchmark vs GHN init experiments.
        
        Args:
            target_epochs: Specific epochs to compare (None for all)
            
        Returns:
            Dictionary with comparison results
        """
        print(f"\n🔄 Comparing Benchmark vs GHN Init experiments...")
        
        # Check if both directories exist
        if not self.benchmark_dir or not self.benchmark_dir.exists():
            print(f"❌ Benchmark directory not found: {self.benchmark_dir}")
            return {}
            
        if not self.ghn_init_dir or not self.ghn_init_dir.exists():
            print(f"❌ GHN init directory not found: {self.ghn_init_dir}")
            return {}
        
        # Evaluate both experiments
        benchmark_results = self.evaluate_experiment(
            self.benchmark_dir, "Benchmark", target_epochs
        )
        
        ghn_init_results = self.evaluate_experiment(
            self.ghn_init_dir, "GHN Init", target_epochs
        )
        
        # Create comparison
        comparison = {
            "benchmark": benchmark_results,
            "ghn_init": ghn_init_results,
            "comparison": {}
        }
        
        # Compare common epochs
        common_epochs = set(benchmark_results.keys()) & set(ghn_init_results.keys())
        
        for epoch in sorted(common_epochs):
            if "error" not in benchmark_results[epoch] and "error" not in ghn_init_results[epoch]:
                bm_loss = benchmark_results[epoch]["test_loss"]
                ghn_loss = ghn_init_results[epoch]["test_loss"]
                
                improvement = ((bm_loss - ghn_loss) / bm_loss) * 100
                
                comparison["comparison"][epoch] = {
                    "benchmark_loss": bm_loss,
                    "ghn_init_loss": ghn_loss,
                    "improvement_percent": improvement,
                    "better_model": "GHN Init" if ghn_loss < bm_loss else "Benchmark"
                }
        
        return comparison
    
    def save_results(self, results: Dict, output_file: str = None):
        """Save results to JSON file in Experiment folder with config name."""
        if output_file is None:
            # Create results directory in Experiment folder
            results_dir = self.experiment_dir / f"results_{self.config_name}"
            results_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            import time
            timestamp = int(time.time())
            output_file = results_dir / f"evaluation_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: {output_file}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate language models at different checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all checkpoints for a config
  python evaluate_checkpoints.py --config benchmark_1_tiny --experiment_dir Experiment/
  
  # Evaluate specific epochs
  python evaluate_checkpoints.py --config benchmark_1_tiny --epochs 2,5,10 --experiment_dir Experiment/
  
  # Compare benchmark vs GHN init
  python evaluate_checkpoints.py --config benchmark_1_tiny --compare --experiment_dir Experiment/
  
  # List available configs
  python evaluate_checkpoints.py --list_configs
        """
    )
    
    parser.add_argument("--config", type=str, help="Benchmark configuration name")
    parser.add_argument("--experiment_dir", type=str, default="Experiment/", 
                       help="Path to experiment directory (default: Experiment/)")
    parser.add_argument("--epochs", type=str, help="Comma-separated list of epochs to evaluate")
    parser.add_argument("--compare", action="store_true", 
                       help="Compare benchmark vs GHN init experiments")
    parser.add_argument("--device", type=str, default="cuda", 
                       help="Device to run evaluation on")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--list_configs", action="store_true", 
                       help="List available benchmark configurations")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # List configs if requested
    if args.list_configs:
        print("Available benchmark configurations:")
        configs = list_benchmark_configs()
        for config in configs:
            print(f"  - {config}")
        return
    
    # Validate arguments
    if not args.config:
        print("❌ Error: --config is required. Use --list_configs to see available configs.")
        return
    
    # Parse epochs if provided
    target_epochs = None
    if args.epochs:
        try:
            target_epochs = [int(e.strip()) for e in args.epochs.split(',')]
        except ValueError:
            print("❌ Error: --epochs must be comma-separated integers (e.g., '2,5,10')")
            return
    
    # Initialize evaluator
    try:
        evaluator = CheckpointEvaluator(args.config, args.experiment_dir, args.device)
    except Exception as e:
        print(f"❌ Error initializing evaluator: {e}")
        return
    
    # Run evaluation
    try:
        if args.compare:
            # Compare experiments
            results = evaluator.compare_experiments(target_epochs)
            
            # Print comparison summary
            print(f"\n📊 Comparison Summary:")
            print(f"{'Epoch':<8} {'Benchmark Loss':<15} {'GHN Init Loss':<15} {'Improvement':<12} {'Better':<10}")
            print("-" * 70)
            
            for epoch, comp in results.get("comparison", {}).items():
                print(f"{epoch:<8} {comp['benchmark_loss']:<15.4f} {comp['ghn_init_loss']:<15.4f} "
                      f"{comp['improvement_percent']:<12.2f}% {comp['better_model']:<10}")
        
        else:
            # Evaluate single experiment
            if evaluator.benchmark_dir and evaluator.benchmark_dir.exists():
                results = evaluator.evaluate_experiment(
                    evaluator.benchmark_dir, "Benchmark", target_epochs
                )
            elif evaluator.ghn_init_dir and evaluator.ghn_init_dir.exists():
                results = evaluator.evaluate_experiment(
                    evaluator.ghn_init_dir, "GHN Init", target_epochs
                )
            else:
                print(f"❌ No experiment directories found for {args.config}")
                print(f"   Searched for:")
                if evaluator.benchmark_dir:
                    print(f"   - Benchmark: {evaluator.benchmark_dir}")
                if evaluator.ghn_init_dir:
                    print(f"   - GHN Init: {evaluator.ghn_init_dir}")
                return
        
        # Save results (use provided output file or default location)
        evaluator.save_results(results, args.output)
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

