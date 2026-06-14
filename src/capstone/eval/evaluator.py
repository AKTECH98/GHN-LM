"""Metrics evaluation from completed training runs."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from capstone.data.wikitext2_loader import build_wikitext2
from capstone.eval.discovery import find_tensorboard_dir
from capstone.lm.create_model import create_model
from capstone.paths import DATA_DIR

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class MetricsEvaluator:
    """Evaluates training metrics from existing experiment data."""

    def __init__(self, config_file: Path, experiment_dir: Path, device: str = "cuda"):
        self.config_file = Path(config_file)
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        with open(self.config_file) as f:
            self.yaml_config = yaml.safe_load(f)

        self.config_name = self.config_file.stem

        config_json_path = self.experiment_dir / "config.json"
        if not config_json_path.exists():
            raise FileNotFoundError(f"Config JSON not found: {config_json_path}")

        with open(config_json_path) as f:
            self.config = json.load(f)

        self.job_id = self.config.get("job_id", self.experiment_dir.name)

        training_config = self.yaml_config.get("training", {})
        self.default_convergence_patience = training_config.get("early_stopping_patience", 3)
        self.default_convergence_threshold = training_config.get("early_stopping_min_delta", 0.001)

        self.tensorboard_dir = find_tensorboard_dir(self.job_id)

        from capstone.eval.discovery import read_experiment_init_method

        self.init_method = read_experiment_init_method(self.experiment_dir)

    def extract_perplexity_from_tensorboard(self, target_epochs: List[int]) -> Dict[int, Dict[str, float]]:
        results: Dict[int, Dict[str, float]] = {}

        if not TENSORBOARD_AVAILABLE or self.tensorboard_dir is None:
            print("   ⚠️  TensorBoard not available or log dir not found. Cannot extract perplexity.")
            return results

        try:
            event_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
            if not event_files:
                print("   ⚠️  No TensorBoard event files found. Cannot extract perplexity.")
                return results

            print(f"   📊 Reading TensorBoard logs from: {event_files[0]}")
            ea = EventAccumulator(str(self.tensorboard_dir))
            ea.Reload()
            scalar_tags = ea.Tags()["scalars"]

            train_perplexities = {}
            val_perplexities = {}

            if "Epoch/Train_Perplexity" in scalar_tags:
                for scalar in ea.Scalars("Epoch/Train_Perplexity"):
                    train_perplexities[int(scalar.step) + 1] = scalar.value

            if "Epoch/Val_Perplexity" in scalar_tags:
                for scalar in ea.Scalars("Epoch/Val_Perplexity"):
                    val_perplexities[int(scalar.step) + 1] = scalar.value

            for epoch in target_epochs:
                results[epoch] = {
                    "train_perplexity": train_perplexities.get(epoch),
                    "val_perplexity": val_perplexities.get(epoch),
                }

            print(f"   ✅ Extracted perplexity for {len(results)} epochs from TensorBoard")
        except Exception as e:
            print(f"   ⚠️  Error reading TensorBoard logs: {e}")

        return results

    def extract_perplexity_from_checkpoints(self, target_epochs: List[int]) -> Dict[int, Dict[str, float]]:
        results: Dict[int, Dict[str, float]] = {}
        checkpoint_files = glob.glob(str(self.experiment_dir / "epoch_*.pt"))
        print(f"   📦 Found {len(checkpoint_files)} checkpoint files")

        for checkpoint_path in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                epoch = checkpoint.get("epoch", -1) + 1
                if epoch in target_epochs:
                    val_loss = checkpoint.get("val_loss")
                    if val_loss is not None:
                        results[epoch] = {
                            "train_perplexity": None,
                            "val_perplexity": torch.exp(torch.tensor(val_loss)).item(),
                        }
            except Exception as e:
                print(f"   ⚠️  Error loading checkpoint {checkpoint_path}: {e}")

        best_model_path = self.experiment_dir / "best_model.pt"
        if best_model_path.exists():
            try:
                checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
                epoch = checkpoint.get("epoch", -1) + 1
                if epoch in target_epochs:
                    val_loss = checkpoint.get("val_loss")
                    if val_loss is not None:
                        results.setdefault(epoch, {})
                        results[epoch]["val_perplexity"] = torch.exp(torch.tensor(val_loss)).item()
            except Exception as e:
                print(f"   ⚠️  Error loading best model: {e}")

        print(f"   ✅ Extracted perplexity for {len(results)} epochs from checkpoints")
        return results

    def analyze_convergence(
        self,
        convergence_threshold: float = 0.0001,
        convergence_patience: int = 5,
    ) -> Dict:
        print("\n🔍 Analyzing convergence...")
        print(f"   Threshold: {convergence_threshold}, Patience: {convergence_patience}")

        empty = {
            "converged": False,
            "convergence_epoch": None,
            "convergence_loss": None,
            "convergence_perplexity": None,
            "epochs_to_convergence": None,
        }

        if not TENSORBOARD_AVAILABLE or self.tensorboard_dir is None:
            print("   ⚠️  TensorBoard not available or log dir not found. Cannot analyze convergence.")
            return empty

        val_losses = []
        try:
            event_files = list(self.tensorboard_dir.glob("events.out.tfevents.*"))
            if not event_files:
                print("   ⚠️  No TensorBoard event files found. Cannot analyze convergence.")
                return empty

            ea = EventAccumulator(str(self.tensorboard_dir))
            ea.Reload()
            if "Epoch/Val_Loss" in ea.Tags()["scalars"]:
                for scalar in ea.Scalars("Epoch/Val_Loss"):
                    val_losses.append((int(scalar.step) + 1, scalar.value))
                val_losses.sort(key=lambda x: x[0])
        except Exception as e:
            print(f"   ⚠️  Error reading TensorBoard for convergence: {e}")
            return empty

        if not val_losses:
            print("   ⚠️  No validation loss data found in TensorBoard logs")
            return empty

        best_loss = float("inf")
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

        if convergence_epoch is None:
            convergence_epoch = val_losses[-1][0]
            best_loss = min(loss for _, loss in val_losses)
            best_epoch = min(epoch for epoch, loss in val_losses if loss == best_loss)

        convergence_perplexity = torch.exp(torch.tensor(best_loss)).item()
        print(f"   ✅ Convergence detected at epoch {convergence_epoch}")
        print(f"      Best loss: {best_loss:.4f}, Perplexity: {convergence_perplexity:.2f}")

        return {
            "converged": True,
            "convergence_epoch": convergence_epoch,
            "convergence_loss": float(best_loss),
            "convergence_perplexity": float(convergence_perplexity),
            "epochs_to_convergence": convergence_epoch,
            "best_epoch": best_epoch,
            "total_epochs": val_losses[-1][0],
        }

    def evaluate_test_dataset(self) -> Dict:
        print("\n🧪 Evaluating best model on test dataset...")

        best_model_path = self.experiment_dir / "best_model.pt"
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model not found: {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)

        from capstone.data.config_loader import ModelConfig

        model_config_dict = self.yaml_config["model"]
        model_config = ModelConfig(**model_config_dict)
        vocab_size = model_config_dict.get("vocab_size", 50257)

        model = create_model(model_config, vocab_size)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        data_config_dict = self.yaml_config["data"]
        training_config_dict = self.yaml_config["training"]

        data = build_wikitext2(
            tokenizer_name="gpt2",
            seq_len=data_config_dict["seq_len"],
            batch_size=training_config_dict["batch_size"],
            num_workers=data_config_dict["num_workers"],
            cache_dir=str(DATA_DIR),
        )

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")

        with torch.no_grad():
            for batch in tqdm(data["test_loader"], desc="Evaluating test set"):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                try:
                    if hasattr(model, "forward") and "targets" in model.forward.__code__.co_varnames:
                        output = model(input_ids, targets=labels)
                    else:
                        output = model(input_ids)

                    if isinstance(output, tuple):
                        logits, loss = output[:2]
                    else:
                        logits, loss = output, None

                    if loss is None:
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        loss = loss * (labels != -100).sum().item()

                    total_loss += loss.item()
                    total_tokens += (labels != -100).sum().item()
                    num_batches += 1
                except Exception as e:
                    print(f"   ⚠️  Error in batch evaluation: {e}")

        if total_tokens == 0:
            raise ValueError("No valid tokens found in test set")

        avg_loss = total_loss / total_tokens
        test_perplexity = torch.exp(torch.tensor(avg_loss)).item()

        print(f"   ✅ Test Perplexity: {test_perplexity:.2f}")
        return {
            "test_loss": float(avg_loss),
            "test_perplexity": float(test_perplexity),
            "total_tokens": int(total_tokens),
            "num_batches": num_batches,
        }

    def evaluate_all_metrics(
        self,
        target_epochs: Optional[List[int]] = None,
        convergence_threshold: Optional[float] = None,
        convergence_patience: Optional[int] = None,
    ) -> Dict:
        if target_epochs is None:
            target_epochs = [1, 2, 5, 10, 20, 50]
        if convergence_patience is None:
            convergence_patience = self.default_convergence_patience
        if convergence_threshold is None:
            convergence_threshold = self.default_convergence_threshold

        print(f"\n{'=' * 60}")
        print("📊 Evaluating All Metrics")
        print(f"{'=' * 60}")

        perplexity_data = self.extract_perplexity_from_tensorboard(target_epochs)
        perplexity_intervals = [
            {
                "epoch": epoch,
                "train_perplexity": perplexity_data[epoch].get("train_perplexity"),
                "val_perplexity": perplexity_data[epoch].get("val_perplexity"),
            }
            for epoch in sorted(perplexity_data.keys())
        ]

        convergence_data = self.analyze_convergence(convergence_threshold, convergence_patience)
        test_data = self.evaluate_test_dataset()

        best_model_info = {}
        best_model_path = self.experiment_dir / "best_model.pt"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location="cpu", weights_only=False)
            best_val_loss = checkpoint.get("val_loss")
            best_model_info = {
                "epoch": checkpoint.get("epoch", -1) + 1,
                "val_loss": float(best_val_loss) if best_val_loss is not None else None,
                "val_perplexity": float(torch.exp(torch.tensor(best_val_loss)).item())
                if best_val_loss is not None
                else None,
            }

        return {
            "config_file": str(self.config_file),
            "config_name": self.config_name,
            "init_method": self.init_method,
            "job_id": self.job_id,
            "experiment_dir": str(self.experiment_dir),
            "perplexity_at_intervals": perplexity_intervals,
            "convergence": convergence_data,
            "test_evaluation": test_data,
            "best_model": best_model_info,
        }
