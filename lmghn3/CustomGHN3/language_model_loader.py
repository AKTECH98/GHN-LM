"""
Language Model Loader for GHN-3 training.
Follows the same pattern as DeepNets-1M loader.
"""

import torch
import torch.utils.data
from functools import partial
from .graph import Graph, GraphBatch
from .graph_adapter import LanguageModelWrapper
from .utils import log


class LanguageModelDataset(torch.utils.data.Dataset):
    """Dataset that yields Graph objects for language models with lazy loading."""
    
    def __init__(self, model_configs, device='cpu', is_train=True, lazy=True):
        """
        Args:
            model_configs: List of model configuration dictionaries
            device: Device to create models on
            is_train: Whether this is for training (needed for NetBatchSamplerDDP)
            lazy: If True, create graphs on-demand. If False, pre-create all graphs.
        """
        self.model_configs = model_configs
        self.device = device
        self.is_train = is_train
        self.lazy = lazy
        self.graphs = []
        self.nodes = []  # Number of nodes for each graph (needed for NetBatchSamplerDDP)
        
        if not lazy:
            # Pre-create all graphs (memory intensive)
            self._create_graphs()
        else:
            # For lazy loading, we'll create graphs on-demand
            # Pre-calculate node counts for NetBatchSamplerDDP
            self._calculate_node_counts()
    
    def _create_graphs(self):
        """Pre-create all graphs."""
        log(f"Creating {len(self.model_configs)} language model graphs...")
        
        for i, config in enumerate(self.model_configs):
            try:
                # Import here to avoid circular imports
                from lmghn3.models import GPTEncoderLayerLM, GPTEncoderConfig
                
                # Create model based on type
                model_type = config.get('model_type', 'gpt_encoder').lower()
                config_copy = config.copy()
                model_type = config_copy.pop('model_type', 'gpt_encoder')
                config_copy.pop('name', None)  # Remove name if present
                
                if model_type == 'gpt_encoder':
                    # Force tie_weights=False for GHN-3 compatibility
                    config_copy['tie_weights'] = False
                    model_config = GPTEncoderConfig(**config_copy)
                    model = GPTEncoderLayerLM(model_config)
                else:
                    log(f"Warning: Unknown model type {model_type}, skipping")
                    continue
                
                # Wrap model
                wrapped_model = LanguageModelWrapper(model)
                
                # Create graph with net_args
                # For language models, we need to provide net_args that can be used to reconstruct the model
                net_args = {
                    'model_type': model_type,
                    'config': config_copy,
                    'vocab_size': config_copy.get('vocab_size', 1000),
                    'd_model': config_copy.get('d_model', 64),
                    'n_layer': config_copy.get('n_layer', 2),
                    'n_head': config_copy.get('n_head', 2),
                    'd_ff': config_copy.get('d_ff', 128),
                    'max_seq_len': config_copy.get('max_seq_len', 32),
                    'p_drop': config_copy.get('p_drop', 0.0),
                    'tie_weights': config_copy.get('tie_weights', True)
                }
                
                graph = Graph(wrapped_model, net_args=net_args)
                graph.net = wrapped_model  # Set the net attribute so GraphBatch can collect it
                self.graphs.append(graph)
                
                # Store number of nodes for this graph
                num_nodes = graph.node_feat.shape[0] if hasattr(graph, 'node_feat') else 0
                self.nodes.append(num_nodes)
                
                if (i + 1) % 10 == 0:
                    log(f"Created {i + 1}/{len(self.model_configs)} graphs")
                    
            except Exception as e:
                log(f"Warning: Failed to create graph for config {i}: {e}")
                continue
        
        log(f"Successfully created {len(self.graphs)} graphs out of {len(self.model_configs)} configs")
        
        # Convert nodes list to tensor for proper indexing
        import torch
        self.nodes = torch.tensor(self.nodes, dtype=torch.long)
    
    def _calculate_node_counts(self):
        """Pre-calculate node counts for NetBatchSamplerDDP without creating full graphs."""
        log(f"Calculating node counts for {len(self.model_configs)} model configurations...")
        
        for i, config in enumerate(self.model_configs):
            try:
                # Estimate node count based on model architecture
                model_type = config.get('model_type', 'gpt_encoder').lower()
                
                if model_type == 'gpt_encoder':
                    # For GPT encoder: embedding + n_layer * (attention + feedforward) + output
                    n_layer = config.get('n_layer', 2)
                    d_model = config.get('d_model', 64)
                    # Rough estimate: embedding + layers + output
                    estimated_nodes = 2 + n_layer * 4 + 1  # embedding, layers, output
                else:
                    # Default estimate for other model types
                    estimated_nodes = 10
                
                self.nodes.append(estimated_nodes)
                
            except Exception as e:
                log(f"Error calculating node count for config {i}: {e}")
                self.nodes.append(10)  # Default fallback
        
        # Convert nodes list to tensor for proper indexing
        import torch
        self.nodes = torch.tensor(self.nodes, dtype=torch.long)
        log(f"Calculated node counts for {len(self.nodes)} configurations")
    
    def _create_single_graph(self, idx):
        """Create a single graph on-demand."""
        config = self.model_configs[idx]
        
        try:
            # Import here to avoid circular imports
            from lmghn3.models import GPTEncoderLayerLM, GPTEncoderConfig
            
            # Create model based on type
            model_type = config.get('model_type', 'gpt_encoder').lower()
            config_copy = config.copy()
            model_type = config_copy.pop('model_type', 'gpt_encoder')
            config_copy.pop('name', None)  # Remove name if present
            
            if model_type == 'gpt_encoder':
                # Force tie_weights=False for GHN-3 compatibility
                config_copy['tie_weights'] = False
                model_config = GPTEncoderConfig(**config_copy)
                model = GPTEncoderLayerLM(model_config)
            elif model_type == 'mini_gpt':
                # Import MiniGPT components
                from lmghn3.models import GPTDecoderLM, MiniGPTConfig
                # Force tie_weights=False for GHN-3 compatibility
                config_copy['tie_weights'] = False
                model_config = MiniGPTConfig(**config_copy)
                model = GPTDecoderLM(model_config)
            else:
                log(f"Warning: Unknown model type {model_type}, skipping")
                return None
            
            # Wrap model
            wrapped_model = LanguageModelWrapper(model)
            
            # Create graph with net_args
            net_args = {
                'model_type': model_type,
                'config': config_copy,
                'vocab_size': config_copy.get('vocab_size', 1000),
                'd_model': config_copy.get('d_model', 64),
                'n_layer': config_copy.get('n_layer', 2),
                'n_head': config_copy.get('n_head', 2),
                'd_ff': config_copy.get('d_ff', 128),
                'max_seq_len': config_copy.get('max_seq_len', 32),
                'p_drop': config_copy.get('p_drop', 0.0),
                'tie_weights': config_copy.get('tie_weights', True)
            }
            
            graph = Graph(wrapped_model, net_args=net_args)
            graph.net = wrapped_model  # Set the net attribute so GraphBatch can collect it
            
            return graph
            
        except Exception as e:
            log(f"Error creating graph for config {idx}: {e}")
            return None
    
    def __len__(self):
        if self.lazy:
            return len(self.model_configs)
        else:
            return len(self.graphs)
    
    def __getitem__(self, idx):
        if self.lazy:
            # Create graph on-demand
            graph = self._create_single_graph(idx)
            if graph is None:
                # Skip this configuration and try the next one
                log(f"Warning: Failed to create graph for config {idx}, skipping")
                # Return a simple dummy graph to avoid None issues
                try:
                    from lmghn3.models import GPTEncoderLayerLM, GPTEncoderConfig
                    dummy_config = GPTEncoderConfig(
                        vocab_size=1000, d_model=64, n_layer=1, n_head=2, 
                        d_ff=128, max_seq_len=32, p_drop=0.0, tie_weights=False
                    )
                    dummy_model = GPTEncoderLayerLM(dummy_config)
                    wrapped_model = LanguageModelWrapper(dummy_model)
                    graph = Graph(wrapped_model, net_args={'model_type': 'gpt_encoder'})
                    graph.net = wrapped_model
                    return graph
                except Exception as e:
                    log(f"Error creating dummy graph: {e}")
                    # If even dummy creation fails, skip this index
                    return self.__getitem__((idx + 1) % len(self.model_configs))
            return graph
        else:
            # Return pre-created graph
            return self.graphs[idx]


class LanguageModelLoader:
    """Language Model Loader following DeepNets-1M pattern."""
    
    @staticmethod
    def loader(model_configs, meta_batch_size=1, dense=True, device='cpu', num_workers=0, lazy=True):
        """
        Create a DataLoader that yields GraphBatch objects.
        
        Args:
            model_configs: List of model configuration dictionaries
            meta_batch_size: Number of graphs per batch
            dense: Whether to use dense graphs
            device: Device to create models on
            num_workers: Number of worker processes
            lazy: If True, create graphs on-demand. If False, pre-create all graphs.
            
        Returns:
            DataLoader that yields GraphBatch objects
        """
        dataset = LanguageModelDataset(model_configs, device=device, is_train=True, lazy=lazy)
        
        # Use NetBatchSampler if available, otherwise use regular batch sampler
        try:
            from .deepnets1m import NetBatchSamplerDDP
            sampler = NetBatchSamplerDDP(dataset, meta_batch_size)
        except ImportError:
            # Fallback to regular batch sampler
            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(dataset),
                batch_size=meta_batch_size,
                drop_last=False
            )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            batch_size=1,
            pin_memory=False,
            collate_fn=partial(GraphBatch, dense=dense),
            num_workers=num_workers
        )
        
        return loader
