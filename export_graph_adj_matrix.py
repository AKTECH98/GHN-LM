#!/usr/bin/env python3
"""
Export graph adjacency matrix and node list to text files for the smallest mini_gpt config.
"""

import torch
import numpy as np
from Dataloader.config_loader import load_config_file
from LM.create_model import create_model
from GHN.graph import Graph

def merge_weight_bias_nodes(graph, adj_binary):
    """
    Merge weight and bias nodes into single nodes for display purposes.
    Returns merged nodes list and merged adjacency matrix.
    """
    nodes = graph._nodes
    n_nodes = len(nodes)
    
    # Find weight/bias pairs
    weight_nodes = {}  # base_name -> node_index
    bias_nodes = {}    # base_name -> node_index
    standalone_nodes = []  # nodes that don't have a weight/bias pair
    
    for i, node in enumerate(nodes):
        param_name = node.get('param_name', '')
        
        # Skip special nodes
        if param_name in ['input'] or 'Backward' in param_name:
            standalone_nodes.append(i)
            continue
        
        # Check if it's a weight or bias
        if param_name.endswith('.weight'):
            base_name = param_name[:-7]  # Remove '.weight'
            weight_nodes[base_name] = i
        elif param_name.endswith('.bias'):
            base_name = param_name[:-5]  # Remove '.bias'
            bias_nodes[base_name] = i
        else:
            standalone_nodes.append(i)
    
    # Create mapping: old_index -> new_index
    old_to_new = {}
    merged_nodes = []
    new_index = 0
    
    # Process pairs
    processed = set()
    for base_name in weight_nodes:
        weight_idx = weight_nodes[base_name]
        bias_idx = bias_nodes.get(base_name, None)
        
        if bias_idx is not None:
            # Merge weight and bias into one node
            weight_node = nodes[weight_idx]
            bias_node = nodes[bias_idx]
            
            # Create merged node representation
            merged_param_name = base_name  # Without .weight or .bias
            merged_module = weight_node.get('module') or bias_node.get('module')
            merged_module_type = type(merged_module).__name__ if merged_module is not None else 'None'
            
            merged_nodes.append({
                'old_indices': [weight_idx, bias_idx],
                'param_name': merged_param_name,
                'module': merged_module,
                'module_type': merged_module_type,
                'id': f"{weight_node.get('id')}_{bias_node.get('id')}"
            })
            
            old_to_new[weight_idx] = new_index
            old_to_new[bias_idx] = new_index
            processed.add(weight_idx)
            processed.add(bias_idx)
            new_index += 1
        else:
            # Only weight, no bias
            standalone_nodes.append(weight_idx)
    
    # Add remaining bias nodes (no matching weight)
    for base_name, bias_idx in bias_nodes.items():
        if bias_idx not in processed:
            standalone_nodes.append(bias_idx)
    
    # Add standalone nodes
    for old_idx in standalone_nodes:
        node = nodes[old_idx]
        merged_nodes.append({
            'old_indices': [old_idx],
            'param_name': node.get('param_name', 'N/A'),
            'module': node.get('module', None),
            'module_type': type(node.get('module', None)).__name__ if node.get('module') is not None else 'None',
            'id': str(node.get('id', old_idx))
        })
        old_to_new[old_idx] = new_index
        new_index += 1
    
    # Create merged adjacency matrix
    n_merged = len(merged_nodes)
    merged_adj = np.zeros((n_merged, n_merged), dtype=int)
    
    # For each edge in original graph, map to merged graph
    for i in range(n_nodes):
        if i not in old_to_new:
            continue
        new_i = old_to_new[i]
        
        for j in range(n_nodes):
            if j not in old_to_new:
                continue
            if adj_binary[i, j] > 0:
                new_j = old_to_new[j]
                # Skip self-loops (when weight and bias nodes that were connected get merged)
                if new_i != new_j:
                    merged_adj[new_i, new_j] = 1
    
    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(merged_adj, 0)
    
    return merged_nodes, merged_adj

def export_graph_to_files(config_path, output_prefix="graph"):
    """Export graph adjacency matrix and node list to text files."""
    
    print(f"📋 Loading configuration from: {config_path}")
    try:
        model_config, training_config, data_config = load_config_file(config_path)
        print(f"   Model: {model_config.model_type}")
        print(f"   D_model: {model_config.d_model}")
        print(f"   N_layers: {model_config.n_layer}")
        print(f"   N_heads: {model_config.n_head}")
    except Exception as e:
        print(f"❌ Error loading config file: {e}")
        return
    
    # Create model
    print(f"\n🏗️  Creating {model_config.model_type} model...")
    device = torch.device('cpu')
    model = create_model(model_config, vocab_size=model_config.vocab_size)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Create graph WITH embeddings
    print(f"\n🕸️  Creating graph from model (including embeddings)...")
    graph = Graph(
        model, 
        ve_cutoff=50, 
        dense=False, 
        verbose=False, 
        exclude_embeddings=False  # Include embeddings
    )
    
    print(f"   Number of nodes (original): {graph.n_nodes}")
    print(f"   Adjacency matrix shape: {graph._Adj.shape}")
    
    # Get adjacency matrix and filter to only direct edges (weight = 1)
    # Exclude virtual edges (weight > 1)
    adj_matrix = graph._Adj.numpy() if isinstance(graph._Adj, torch.Tensor) else graph._Adj
    adj_direct = (adj_matrix == 1).astype(int)  # Only direct edges, no virtual edges
    
    print(f"   Number of direct edges (weight=1): {adj_direct.sum()}")
    print(f"   Number of virtual edges (weight>1): {(adj_matrix > 1).sum()}")
    print(f"   (Virtual edges will be excluded from output)")
    
    # Merge weight and bias nodes for display
    print(f"\n🔗 Merging weight and bias nodes...")
    merged_nodes, merged_adj = merge_weight_bias_nodes(graph, adj_direct)  # Use direct edges only
    print(f"   Number of nodes (merged): {len(merged_nodes)}")
    print(f"   Number of edges (merged, direct only): {merged_adj.sum()}")
    
    # Save adjacency matrix to text file (using merged nodes)
    adj_file = f"{output_prefix}_adjacency_matrix.txt"
    print(f"\n💾 Saving adjacency matrix to: {adj_file}")
    with open(adj_file, 'w') as f:
        # Write header
        f.write(f"# Adjacency Matrix for {config_path}\n")
        f.write(f"# Shape: {merged_adj.shape[0]}x{merged_adj.shape[1]}\n")
        f.write(f"# Number of nodes (merged): {len(merged_nodes)}\n")
        f.write(f"# Number of nodes (original): {graph.n_nodes}\n")
        f.write(f"# Number of edges (direct only): {merged_adj.sum()}\n")
        f.write(f"# Format: Binary matrix (0 = no edge, 1 = direct edge)\n")
        f.write("# Each row represents a source node, each column represents a target node\n")
        f.write("# Note: Weight and bias nodes have been merged into single nodes\n")
        f.write("# Note: Only direct edges (weight=1) are included, virtual edges excluded\n")
        f.write("#\n")
        
        # Write matrix
        for i in range(merged_adj.shape[0]):
            row_str = ' '.join(str(merged_adj[i, j]) for j in range(merged_adj.shape[1]))
            f.write(row_str + '\n')
    
    print(f"   ✅ Adjacency matrix saved!")
    
    # Save node list to text file (using merged nodes)
    nodes_file = f"{output_prefix}_nodes.txt"
    print(f"\n💾 Saving node list to: {nodes_file}")
    with open(nodes_file, 'w') as f:
        # Write header
        f.write(f"# Node List for {config_path}\n")
        f.write(f"# Total nodes (merged): {len(merged_nodes)}\n")
        f.write(f"# Total nodes (original): {len(graph._nodes)}\n")
        f.write(f"# Format: node_index | param_name | module_type | (original_indices)\n")
        f.write("# Note: Weight and bias nodes have been merged into single nodes\n")
        f.write("#\n")
        
        # Write merged nodes
        for i, merged_node in enumerate(merged_nodes):
            param_name = merged_node['param_name']
            module_type = merged_node['module_type']
            old_indices = merged_node['old_indices']
            
            if len(old_indices) > 1:
                indices_str = f"({old_indices[0]},{old_indices[1]})"
            else:
                indices_str = f"({old_indices[0]})"
            
            f.write(f"{i:4d} | {param_name:50s} | {module_type:15s} | {indices_str}\n")
    
    print(f"   ✅ Node list saved!")
    
    # Also save a summary file
    summary_file = f"{output_prefix}_summary.txt"
    print(f"\n💾 Saving summary to: {summary_file}")
    with open(summary_file, 'w') as f:
        f.write(f"Graph Summary for {config_path}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Model Type: {model_config.model_type}\n")
        f.write(f"  D_model: {model_config.d_model}\n")
        f.write(f"  N_layers: {model_config.n_layer}\n")
        f.write(f"  N_heads: {model_config.n_head}\n")
        f.write(f"  D_ff: {model_config.d_ff}\n")
        f.write(f"  Vocab Size: {model_config.vocab_size}\n")
        f.write(f"  Max Seq Len: {model_config.max_seq_len}\n")
        f.write(f"  Total Parameters: {total_params:,}\n\n")
        f.write(f"Graph Statistics:\n")
        f.write(f"  Number of Nodes (original): {graph.n_nodes}\n")
        f.write(f"  Number of Nodes (merged): {len(merged_nodes)}\n")
        f.write(f"  Adjacency Matrix Shape (merged): {merged_adj.shape}\n")
        f.write(f"  Number of Direct Edges (merged): {merged_adj.sum()}\n")
        f.write(f"  Edge Density (merged): {merged_adj.sum() / (merged_adj.shape[0] * merged_adj.shape[1]):.6f}\n")
        f.write(f"  Note: Only direct edges (weight=1) included, virtual edges excluded\n\n")
        f.write(f"Node Types (merged):\n")
        node_types = {}
        for merged_node in merged_nodes:
            module_type = merged_node['module_type']
            node_types[module_type] = node_types.get(module_type, 0) + 1
        for node_type, count in sorted(node_types.items()):
            f.write(f"  {node_type}: {count}\n")
    
    print(f"   ✅ Summary saved!")
    
    print(f"\n✅ Export completed!")
    print(f"   Files created:")
    print(f"     - {adj_file}")
    print(f"     - {nodes_file}")
    print(f"     - {summary_file}")

def main():
    """Main function."""
    # Use the smallest mini_gpt config
    config_path = "LM/configs/benchmark_6_mini_gpt_tiny.yaml"
    output_prefix = "mini_gpt_tiny_graph"
    
    export_graph_to_files(config_path, output_prefix)

if __name__ == "__main__":
    main()

