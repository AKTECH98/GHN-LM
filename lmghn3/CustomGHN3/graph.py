import numpy as np
import heapq
import torch
import torch.nn as nn
import networkx as nx
import transformers
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Scatter as _scatter

import sys
sys.setrecursionlimit(10000)  # for large models like efficientnet_v2_l

t_long = torch.long

class GraphBatch:
    r"""
    Container for a batch of Graph objects.

    Example:

        batch = GraphBatch([Graph(torchvision.models.resnet50())])

    """

    def __init__(self, graphs, dense=False):
        r"""
        :param graphs: iterable, where each item is a Graph object.
        :param dense: create dense node and adjacency matrices (e.g. for transformer)
        """
        self.n_nodes, self.node_feat, self.node_info, self.edges, self.net_args, self.net_inds = [], [], [], [], [], []
        self._n_edges = []
        self.graphs = graphs
        self.dense = dense
        if self.dense:
            self.mask = []

        if graphs is not None:
            if not isinstance(graphs, (list, tuple)):
                graphs = [graphs]
            for graph in graphs:
                self.append(graph)

    def append(self, graph):
        graph_offset = len(self.n_nodes)                    # current index of the graph in a batch
        self.n_nodes.append(len(graph.node_feat))           # number of nodes

        if self.dense:
            self.node_feat.append(graph.node_feat)
            self.edges.append(graph._Adj)
        else:
            self._n_edges.append(len(graph.edges))              # number of edges
            self.node_feat.append(torch.cat((graph.node_feat,   # primitive type
                                             graph_offset + torch.zeros(len(graph.node_feat), 1, dtype=torch.long)),
                                            dim=1))             # graph index for each node
            self.edges.append(torch.cat((graph.edges,
                                         graph_offset + torch.zeros(len(graph.edges), 1, dtype=torch.long)),
                                        dim=1))                 # graph index for each edge

        self.node_info.append(graph.node_info)      # op names, ids, etc.
        self.net_args.append(graph.net_args)        # a dictionary of arguments to construct a Network object
        self.net_inds.append(graph.net_idx)         # network integer identifier (optional)
        if hasattr(graph, 'net'):
            if not hasattr(self, 'nets'):
                self.nets = []
            self.nets.append(graph.net)

    def scatter(self, device_ids, nets):
        """
        Distributes the batch of graphs and networks to multiple CUDA devices.
        :param device_ids: list of CUDA devices
        :param nets: list of networks
        :return: list of tuples of networks and corresponding graphs
        """
        n_graphs = len(self.n_nodes)  # number of graphs in a batch
        gpd = int(np.ceil(n_graphs / len(device_ids)))  # number of graphs per device

        if len(device_ids) > 1:
            sorted_idx = self._sort_by_nodes(len(device_ids), gpd)
            nets = [nets[i] for i in sorted_idx]

        chunks_iter = np.arange(0, n_graphs, gpd)
        n_nodes_chunks = [len(self.n_nodes[i:i + gpd]) for i in chunks_iter]
        if self.dense:
            if not isinstance(self.n_nodes, torch.Tensor):
                self.n_nodes = torch.tensor(self.n_nodes, dtype=t_long)
            self.n_nodes = _scatter.apply(device_ids, n_nodes_chunks, 0, self.n_nodes)
        else:
            node_chunks = [sum(self.n_nodes[i:i + gpd]) for i in chunks_iter]
            edge_chunks = [sum(self._n_edges[i:i + gpd]) for i in chunks_iter]
            self._cat()
            self.node_feat = _scatter.apply(device_ids, node_chunks, 0, self.node_feat)
            self.edges = _scatter.apply(device_ids, edge_chunks, 0, self.edges)
            self.n_nodes = _scatter.apply(device_ids, n_nodes_chunks, 0, self.n_nodes)

        batch_lst = []  # each item in the list is a GraphBatch instance
        for device, i in enumerate(chunks_iter):
            # update graph_offset for each device
            graphs = GraphBatch([], dense=self.dense)
            graphs.n_nodes = self.n_nodes[device]

            if self.dense:
                max_nodes = max(graphs.n_nodes)
                graphs.node_feat = [None] * gpd
                graphs.edges = [None] * gpd
                graphs.mask = torch.zeros(gpd, max_nodes, 1, dtype=torch.bool, device=device)
                for k, j in enumerate(range(i, i + gpd)):
                    n = graphs.n_nodes[k]

                    assert n == len(self.node_feat[j]) == len(self.edges[j]), \
                        (i, j, k, n, len(self.node_feat[j]), len(self.edges[j]))

                    graphs.node_feat[k] = F.pad(self.node_feat[j], (0, 0, 0, max_nodes - n), mode='constant')
                    graphs.edges[k] = F.pad(self.edges[j], (0, max_nodes - n, 0, max_nodes - n), mode='constant')
                    graphs.mask[k, :n] = 1

                graphs.node_feat = torch.stack(graphs.node_feat, dim=0).to(device)
                graphs.edges = torch.stack(graphs.edges, dim=0).to(device)

            else:
                self.node_feat[device][:, -1] = self.node_feat[device][:, -1] - gpd * device
                self.edges[device][:, -1] = self.edges[device][:, -1] - gpd * device
                graphs.node_feat = self.node_feat[device]
                graphs.edges = self.edges[device]

            graphs.node_info = self.node_info[i:i + gpd]
            graphs.net_args = self.net_args[i:i + gpd]
            graphs.net_inds = self.net_inds[i:i + gpd]
            batch_lst.append((nets[i:i + gpd], graphs))  # match signature of the GHN forward pass

        return batch_lst

    def to_device(self, device):
        if isinstance(device, (tuple, list)):
            device = device[0]

        if self.on_device(device):
            print('WARNING: GraphBatch is already on device %s.' % str(device))

        self._cat(device)
        self.node_feat = self.node_feat.to(device, non_blocking=True)
        self.edges = self.edges.to(device, non_blocking=True)
        return self

    def on_device(self, device):
        if isinstance(device, (tuple, list)):
            device = device[0]
        return isinstance(self.n_nodes, torch.Tensor) and self.node_feat.device == device

    def to_dense(self, x=None):
        if x is None:
            x = self.node_feat
        B, M, C = len(self.n_nodes), max(self.n_nodes), x.shape[-1]
        node_feat = torch.zeros(B, M, C, device=x.device)
        offset = [0]
        for b in range(B):
            node_feat[b, :self.n_nodes[b]] = x[offset[-1]: offset[-1] + self.n_nodes[b]]
            offset.append(offset[-1] + self.n_nodes[b])
        return node_feat, offset

    def to_sparse(self, x):
        node_feat = torch.cat([x[b, :self.n_nodes[b]] for b in range(len(self.n_nodes))])
        return node_feat

    def _sort_by_nodes(self, num_devices, gpd):
        """
        Sorts graphs and associated attributes in a batch by the number of nodes such
        that the memory consumption is more balanced across GPUs.
        :param num_devices: number of GPU devices (must be more than 1)
        :param gpd: number of graphs per GPU
                                (all GPUs are assumed to receive the same number of graphs)
        :return: indices of sorted graphs
        """
        n_nodes = np.array(self.n_nodes)
        sorted_idx = np.argsort(n_nodes)[::-1]  # decreasing order
        n_nodes = n_nodes[sorted_idx]

        heap = [(0, idx) for idx in range(num_devices)]
        heapq.heapify(heap)
        idx_groups = {}
        for i in range(num_devices):
            idx_groups[i] = []

        for idx, n in enumerate(n_nodes):
            while True:
                set_sum, set_idx = heapq.heappop(heap)
                if len(idx_groups[set_idx]) < gpd:
                    break
            idx_groups[set_idx].append(sorted_idx[idx])
            heapq.heappush(heap, (set_sum + n, set_idx))

        idx = np.concatenate([np.array(v) for v in idx_groups.values()])
        idx = idx[::-1]  # to make fewer nodes on the first device (which is using more)

        # Sort everything according to the idx order
        self.n_nodes = [self.n_nodes[i] for i in idx]
        self.node_info = [self.node_info[i] for i in idx]
        self.net_args = [self.net_args[i] for i in idx]
        self.net_inds = [self.net_inds[i] for i in idx]
        if self.dense:
            self.node_feat = [self.node_feat[i] for i in idx]
            self.edges = [self.edges[i] for i in idx]
            if len(self.mask) > 0:
                self.mask = [self.mask[i] for i in idx]
        else:
            self._n_edges = [self._n_edges[i] for i in idx]
            # update graph_offset for each graph
            node_feat, edges = [], []
            for graph_offset, i in enumerate(idx):
                node_feat_i = self.node_feat[i]
                edges_i = self.edges[i]
                node_feat_i[:, -1] = graph_offset
                edges_i[:, -1] = graph_offset
                node_feat.append(node_feat_i)
                edges.append(edges_i)
            self.node_feat = node_feat
            self.edges = edges

        return idx

    def _cat(self, device='cpu'):
        if not isinstance(self.n_nodes, torch.Tensor):
            self.n_nodes = torch.tensor(self.n_nodes, dtype=t_long, device=device)
        else:
            self.n_nodes = self.n_nodes.to(device, non_blocking=True)

        max_nodes = max(self.n_nodes)

        if not isinstance(self.node_feat, torch.Tensor):

            if self.dense:
                self.mask = torch.zeros(len(self.n_nodes), max_nodes, 1, dtype=torch.bool, device=device)
                for i, x in enumerate(self.node_feat):
                    self.node_feat[i] = F.pad(x, (0, 0, 0, max_nodes - len(x)), mode='constant')
                    self.mask[i, :len(x)] = 1
                    assert self.n_nodes[i] == len(x), (self.n_nodes[i], len(x))
                self.node_feat = torch.stack(self.node_feat, dim=0)
            else:
                self.node_feat = torch.cat(self.node_feat)

        if not isinstance(self.edges, torch.Tensor):
            if self.dense:
                for i, x in enumerate(self.edges):
                    self.edges[i] = F.pad(x, (0, max_nodes - len(x), 0, max_nodes - len(x)), mode='constant')
                self.edges = torch.stack(self.edges, dim=0)
            else:
                self.edges = torch.cat(self.edges)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.n_nodes)

    def __iter__(self):
        for graph in self.graphs:
            yield graph


class Graph:
    r"""
    Container for a computational graph of a neural network.

    Example:

        graph = Graph(torchvision.models.resnet50())

    """

    def __init__(self, model=None, node_feat=None, node_info=None, A=None, edges=None, net_args=None, net_idx=None,
                 ve_cutoff=50, list_all_nodes=False, reduce_graph=True, fix_weight_edges=True, fix_softmax_edges=True,
                 dense=False, verbose=True):
        r"""
        Pass either model or node/edge arguments.
        :param model: Neural Network inherited from nn.Module
        :param node_feat: node features (optional, only if model is None)
        :param node_info: node meta-information (optional, only if model is None)
        :param A: adjacency matrix in the dense format (optional, only if model is None)
        :param edges: adjacency matrix in the sparse format (optional, only if model is None)
        :param net_args: network arguments (optional, only if model is None)
        :param net_idx: network index in the DeepNets-1M dataset (optional, only if model is None)
        :param ve_cutoff: virtual edge cutoff
        :param list_all_nodes: for dataset generation
        :param reduce_graph: remove redundant/unsupported nodes
        :param fix_weight_edges: rewire edges to/from the weight nodes to make it a correct DAG
        :param fix_softmax_edges: rewire edges to/from the softmax nodes to make it consistent with DeepNets-1M DAGs
        :param verbose: print warnings
        """

        assert node_feat is None or model is None, 'either model or other arguments must be specified'

        self.model = model
        self._list_all_nodes = list_all_nodes  # True in case of dataset generation
        self._verbose = verbose
        self._reduce_graph = reduce_graph
        self._fix_weight_edges = fix_weight_edges
        self._fix_softmax_edges = fix_softmax_edges
        self.nx_graph = None  # NetworkX DiGraph instance

        if model is not None:
            self._build_graph()
            self._add_virtual_edges(ve_cutoff=ve_cutoff)
            self._construct_features()
        else:
            self.n_nodes = len(node_feat)
            self.node_feat = node_feat
            self.node_info = node_info

            if dense:
                self._Adj = A
            else:
                if edges is None:
                    if not isinstance(A, torch.Tensor):
                        A = torch.from_numpy(A).long()
                    ind = torch.nonzero(A)
                    self.edges = torch.cat((ind, A[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)
                else:
                    self.edges = edges

        self.net_args = net_args
        self.net_idx = net_idx

    def _owner_module(self, qname: str):
        if not qname:
            return self.model
        parts = qname.split(".")[:-1]
        try:
            mod = self.model
            for p in parts:
                mod = getattr(mod, p)
            return mod
        except AttributeError:
            return None

    def _build_param_map_with_aliases(self):
        """
        Returns:
          param_map: dict[id(param)] -> (primary_name, owner_module, meta)
            where meta contains {"aliases": [..], "is_buffer": bool}
        """
        # Primary/canonical names from named_parameters (unique by id)
        param_map = {}
        for name, p in self.model.named_parameters(recurse=True):
            pid = id(p)
            if pid not in param_map:
                param_map[pid] = (name, self._owner_module(name), {"aliases": [], "is_buffer": False})

        # Add buffers (rarely appear as leaves, but include for completeness)
        for name, b in self.model.named_buffers(recurse=True):
            pid = id(b)
            if pid not in param_map:
                param_map[pid] = (name, self._owner_module(name), {"aliases": [], "is_buffer": True})
            else:
                param_map[pid][2]["is_buffer"] = True

        # Collect alias names from each submodule's direct params
        for mod_name, m in self.model.named_modules():
            for p_name, p in m.named_parameters(recurse=False):
                if p is None:
                    continue
                alias = f"{mod_name}.{p_name}" if mod_name else p_name
                pid = id(p)
                if pid in param_map:
                    primary_name, owner, meta = param_map[pid]
                    if alias != primary_name and alias not in meta["aliases"]:
                        meta["aliases"].append(alias)
                else:
                    # Very rare: direct param not in named_parameters() (shouldn’t happen)
                    param_map[pid] = (alias, self._owner_module(alias), {"aliases": [], "is_buffer": False})

        return param_map

    def _build_graph(self):


        param_map = self._build_param_map_with_aliases()
        self.abc = param_map
        nodes, edges, seen = {}, [], {}

        def get_attr(fn):
            """
            Get extra attributes of a node in a computational graph that can help identify the node.
            :param fn:
            :return:
            """
            attrs = dict()
            for attr in dir(fn):
                if not attr.startswith('_saved_'):
                    continue
                val = getattr(fn, attr)
                attr = attr[len('_saved_'):]
                if torch.is_tensor(val):
                    attrs[attr] = "[saved tensor]"
                elif isinstance(val, tuple) and any(torch.is_tensor(t) for t in val):
                    attrs[attr] = "[saved tensors]"
                else:
                    attrs[attr] = str(val)
            return attrs

        def traverse_graph(fn):
            r"""
            Traverse the computational graph of a neural network in the backward direction starting
            from the output node (var).
            :param fn:
            :return:
            """
            assert not torch.is_tensor(fn)
            if fn in seen:
                return seen[fn]

            fn_name = str(type(fn).__name__)
            node_link, link_start = None, None
            if fn_name.find('AccumulateGrad') < 0:
                leaf_nodes = []
                for u in fn.next_functions:
                    for i_u, uu in enumerate(u):
                        if uu is not None:  # so it's okay to keep uu=u[0] since u[1] never has variable field
                            if hasattr(uu, 'variable'):
                                var = uu.variable
                                primary, module, meta = param_map.get(id(var), (f"<unnamed id={id(var)}>", None, {}))
                                name = primary
                                leaf_nodes.append({'id': uu,
                                                   'param_name': name,
                                                   'attrs': {'size': var.size(), **get_attr(var)},
                                                   'module': module})

                                assert len(uu.next_functions) == 0

                if len(leaf_nodes) == 0:
                    leaf_nodes.append({'id': fn,
                                       'param_name': fn_name,
                                       'attrs': get_attr(fn),
                                       'module': None})

                assert not hasattr(fn, 'variable'), fn.variable

                for leaf in leaf_nodes:
                    node_link = str(id(leaf['id']))
                    if link_start is None:
                        link_start = node_link

                    seen[leaf['id']] = (node_link, leaf['param_name'])
                    nodes[node_link] = {'param_name': leaf['param_name'],
                                        'attrs': leaf['attrs'],
                                        'module': leaf['module']}

            seen[fn] = (node_link, fn_name)

            # recurse
            if hasattr(fn, 'next_functions'):
                for u in fn.next_functions:
                    for uu in u:
                        if uu is not None and not isinstance(uu, int):
                            link_, name_ = traverse_graph(uu)
                            if link_ is not None and link_start != link_:
                                edges.append((link_start, link_) if name_.find('bias') >= 0 else (link_, link_start))

            return node_link, fn_name

        with torch.enable_grad():
            device = next(self.model.parameters()).device
            # Heuristic defaults for LMs:
            V = getattr(getattr(self.model, "tok", None), "num_embeddings", None)
            if V is None:
                # try standard Embedding under common names
                emb = None
                for m in self.model.modules():
                    if isinstance(m, nn.Embedding):
                        emb = m;
                        break
                V = emb.num_embeddings if emb is not None else 50257  # GPT2 default-ish

            cfg = getattr(self.model, "config", None)
            T = getattr(self.model, "max_len",
                        getattr(cfg, "n_positions",
                                getattr(cfg, "max_position_embeddings", 64)))

            B = 2
            input_ids = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

            out = self.model(input_ids) if callable(getattr(self.model, "__call__", None)) else self.model(
                input_ids=input_ids)
            if isinstance(out, dict): out = list(out.values())
            if not isinstance(out, (list, tuple)): out = [out]

            for v in out:
                if isinstance(v, torch.Tensor) and v.grad_fn is not None:
                    traverse_graph(v.grad_fn)

        nodes_lookup = {key: i for i, key in enumerate(nodes)}
        nodes = [{'id': key, **nodes[key]} for key in nodes_lookup]
        A = np.zeros((len(nodes), len(nodes)))
        for out_node_id, in_node_id in edges:
            A[nodes_lookup[out_node_id], nodes_lookup[in_node_id]] = 1

        self._Adj = A
        self._nodes = nodes
        if self._reduce_graph:
            A, nodes = self._filter_graph(exclude_embeddings=True)  # Filter graph first time to remove most of the redundant/unsupported nodes

        if self._fix_weight_edges:
            # The weight tensor is often incorrectly placed as a leaf node with a wrong edge direction.
            # For example, for a two layer network like:
            # self.fc = nn.Sequential(
            #             nn.Linear(in_features, in_features),
            #             nn.ReLU(),
            #             nn.Linear(in_features, out_features),
            #         )
            # the edges can be layer0.bias->layer1.bias and layer1.weight->layer1.bias, so layer1.weight does not have
            # incoming edges and is unreachable if we traverse the graph in the forward direction.
            # The loop below corrects the graph by making the edges like layer0.bias->layer1.weight->layer1.bias.

            pattern = 'weight'  # we assume the leaf modules should have a weight attribute
            for i, node in enumerate(nodes):
                if A[:, i].sum() > 0:  # if there are incoming edges to the node, then it already should be correct
                    continue
                if node['param_name'].find(pattern) < 0:  # if no 'weight' string in the name, assume graph is correct
                    continue

                for out_neigh in np.where(A[i, :])[0]:  # all nodes with an edge from the weight node, e.g. bias

                    is_same_layer = node['module'] == nodes[out_neigh]['module']
                    qkv = len(np.where(A[:, i])[0]) == 0 and nodes[out_neigh]['param_name'].lower().find('softmax') >= 0
                    if is_same_layer or qkv:

                        n_out = len(np.where(A[i, :])[0])  # number of out neighbors the weight node has

                        in_out = np.setdiff1d(np.where(A[:, out_neigh])[0], i)  # incoming to the bias except the i node
                        if len(in_out) == 0:  # if the w (i) is the only incoming to the b, then it should be correct
                            continue

                        nodes[i], nodes[out_neigh] = nodes[out_neigh], nodes[i]
                        A[i, out_neigh], A[out_neigh, i] = 0, 1

                        if n_out == 1:
                            # out_neigh is the weight node after swapping, while i is the bias node
                            out_new = np.setdiff1d(np.where(A[out_neigh, :])[0], i)  # outcoming from w except the bias
                            if len(out_new) == 0:
                                continue
                            A[out_neigh, out_new] = 0  # remove the edges from the weight to out_new
                            A[i, out_new] = 1  # add edges from the bias to out_new

        if self._fix_softmax_edges:
            # Fix softmax/msa edges to be consistent with the GHN/DeepNets-1M code
            pattern = 'softmax'
            self.nx_graph = self._nx_graph_from_adj(A=A)
            for i, node in enumerate(nodes):
                if node['param_name'].lower().find(pattern) < 0:
                    continue
                for out_neigh in np.where(A[i, :])[0]:  # all nodes following i (msa/softmax), usually just one node
                    in_out = np.setdiff1d(np.where(A[:, out_neigh])[0], i)  # nodes coming to out_neigh, except from i
                    for j in in_out:
                        # remove all edges coming to the node next to msa
                        n_paths = 0
                        for _ in nx.all_simple_paths(self.nx_graph, j, out_neigh):
                            n_paths += 1
                            if n_paths > 1:
                                break
                        # if n_paths == 1 and A[i, j] == 1, then do not change anything
                        if n_paths > 1 or A[i, j] == 0:
                            A[j, out_neigh] = 0  # For ViTs, there should be 2 paths, so remove the 2nd edge to softmax
                        if n_paths == 1 and A[i, j] == 0:
                            # if only one path from j to out_neigh, then the edge (j, i) will replace (j, out_neigh)
                            A[j, i] = 1

        if sum(A[np.diag_indices_from(A)]) > 0 and self._verbose:
            print('WARNING: diagonal elements of the adjacency matrix should be zero', sum(A[np.diag_indices_from(A)]))

        if self._reduce_graph:
            # Filter the graph one more time, since above manipulations could lead to redundant add/concat nodes
            A, nodes = self._filter_graph(unsupported_modules=['Add', 'Cat'])

        # Add input node
        try:
            A = np.pad(A, ((0, 1), (0, 1)), mode='constant')
            nodes.append({'id': 'input', 'param_name': 'input', 'attrs': None, 'module': None})
            # Should work for multiple inputs
            for ind in np.where(A.sum(0) == 0)[0]:  # nodes that do not have any incoming edges
                if nodes[ind]['param_name'].find('weight') >= 0:
                    A[-1, ind] = 1
        except Exception as e:
            print('WARNING: adding input node failed:', e)

        # After topo-sort produced A, nodes
        has_add = any('AddBackward' in n['param_name'] for n in nodes)
        has_embed = any(isinstance(n.get('module'), nn.Embedding) for n in nodes)
        if has_embed and not has_add:
            insert_at = 1
            nodes.insert(insert_at,
                         {'id': 'sum_pos_tok', 'param_name': 'AddBackward0', 'attrs': None, 'module': None})
            A = np.insert(A, insert_at, 0, axis=0);
            A = np.insert(A, insert_at, 0, axis=1)
            emb_idxs = [i for i, n in enumerate(nodes) if isinstance(n.get('module'), nn.Embedding)]
            for ei in emb_idxs:
                # redirect: emb -> sum -> original succs
                succ = np.where(A[ei, :])[0]
                for s in succ:
                    A[ei, s] = 0
                    A[insert_at, s] = 1
                A[ei, insert_at] = 1

        # Sort nodes in a topological order consistent with forward propagation
        try:
            A[np.diag_indices_from(A)] = 0
            ind = np.array(list(nx.topological_sort(nx.DiGraph(A))))
            nodes = [nodes[i] for i in ind]
            A = A[ind, :][:, ind]
        except Exception as e:
            print('WARNING: topological sort failed:', e)

        self._Adj = A
        self._nodes = nodes

        return

    def _filter_graph(self, unsupported_modules=None, exclude_embeddings=True):
        """
        Keep: Linear, LayerNorm, MultiheadAttention, TransformerEncoder/Layer,
              Softmax (as msa), Add (sum), Cat (if merges >1 input), input.
        Drop: mean/pooling, embeddings (if exclude_embeddings=True), and anything not recognized.
        """
        if unsupported_modules is None:
            unsupported_modules = ['Mean', 'AdaptiveAvgPool', 'MaxPool', 'AvgPool']

        # precompute indegrees
        indeg = [int(self._Adj[:, i].sum()) for i in range(len(self._nodes))]
        keep_idx = []
        for i, node in enumerate(self._nodes):
            pname = node['param_name']
            base = pname.split('Backward')[0]
            name = MODULES_LM.get(base, None)
            m = node.get('module', None)
            if name is None and m is not None:
                for k, v in MODULES_LM.items():
                    if not isinstance(k, str) and isinstance(m, k):
                        name = v(m, pname);
                        break

            keep = False
            if name in ('linear', 'ln', 'msa', 'input'):
                # Check if this is an embedding-related layer that should be excluded
                if exclude_embeddings and ('lm_head' in pname or 'tok' in pname or 'pos' in pname):
                    keep = False
                else:
                    keep = True
            elif name == 'embed':
                # Exclude embedding layers from GHN prediction if requested
                keep = not exclude_embeddings
            elif name == 'sum':
                keep = indeg[i] > 1
            elif name == 'concat':
                keep = indeg[i] > 1

            if keep:
                keep_idx.append(i)
            else:
                # rewire predecessors directly to successors
                pred = np.where(self._Adj[:, i])[0]
                succ = np.where(self._Adj[i, :])[0]
                for p in pred:
                    for s in succ:
                        if p != s:
                            self._Adj[p, s] = 1

        keep_idx = np.array(sorted(set(keep_idx)), dtype=int)
        if len(keep_idx) < self._Adj.shape[0]:
            self._Adj = self._Adj[:, keep_idx][keep_idx, :]
            self._nodes = [self._nodes[i] for i in keep_idx]
        return self._Adj, self._nodes

    def _add_virtual_edges(self, ve_cutoff=50):
        r"""
        Add virtual edges with weights equal the shortest path length between the nodes.
        :param ve_cutoff: maximum shortest path length between the nodes
        :return:
        """

        self.n_nodes = len(self._nodes)

        assert self._Adj[np.diag_indices_from(self._Adj)].sum() == 0, (
            'no loops should be in the graph', self._Adj[np.diag_indices_from(self._Adj)].sum())

        # Check that the graph is connected and all nodes reach the final output
        self._nx_graph_from_adj()

        if self._verbose:
            length = nx.shortest_path(self.nx_graph, target=self.n_nodes - 1)
            for node in range(self.n_nodes):
                if node not in length and not self._nodes[node]['param_name'].lower().startswith('aux'):
                    print('WARNING: node={}-{} does not have a path to node={}-{}'.format(
                        node, self._nodes[node]['param_name'], len(self._nodes) - 1, self._nodes[-1]['param_name']))

            # Check that all nodes have a path to the input
            length = nx.shortest_path(self.nx_graph, source=0)
            for node in range(self.n_nodes):
                if node in length:
                    continue
                source_name = self._nodes[0]['param_name']
                target_name = self._nodes[node]['param_name']
                if not (target_name.startswith('pos_enc') or
                        target_name.find('pos_emb') >= 0 or
                        target_name.find('position_bias') >= 0 or
                        source_name.find('position_bias') >= 0):
                    print('WARNING: node={}-{} does not have a path to node={}-{}'.format(
                        0, self._nodes[0]['param_name'], node, self._nodes[node]['param_name']))

        if ve_cutoff > 1:
            length = dict(nx.all_pairs_shortest_path_length(self.nx_graph, cutoff=ve_cutoff))
            for node1 in length:
                for node2 in length[node1]:
                    if length[node1][node2] > 0 and self._Adj[node1, node2] == 0:
                        self._Adj[node1, node2] = length[node1][node2]
            assert (self._Adj > ve_cutoff).sum() == 0, ((self._Adj > ve_cutoff).sum(), ve_cutoff)
        return self._Adj

    def _construct_features(self):
        """
            LM-only node features: [type_id, in_dim, out_dim, act_id] (int64)
            Edge features: [src, dst, weight, edge_type] where edge_type: 0=fwd, 1=residual, 2=posenc
        """

        self.n_nodes = len(self._nodes)
        self.node_feat = torch.empty(self.n_nodes, 4, dtype=t_long)
        self.node_info = [[]]
        self._param_shapes = []

        PRIM2IDX_LM = {op: i for i, op in enumerate(PRIMITIVES_LM)}

        def module_to_name(node, pname):
            lowp = (pname or "").lower()
            # Heuristic: GPT-2 attention params live under ".attn."
            # Count them as an 'msa' concept instead of 'linear'
            if ".attn." in lowp and any(k in lowp for k in ("c_attn", "c_proj", "attn.bias", "masked_bias", "softmax")):
                return "msa"

            m = node.get('module', None)
            if m is not None:
                for k, v in MODULES_LM.items():
                    if not isinstance(k, str) and isinstance(m, k):
                        return v(m, pname)

            base = (pname or "").split('Backward')[0]
            return MODULES_LM.get(base, None)

        for i, node in enumerate(self._nodes):
            pname = node['param_name']
            name = module_to_name(node, pname) or 'linear'
            m = node.get('module', None)

            sz = None
            attrs = node.get('attrs')
            if isinstance(attrs,dict) and 'size' in attrs:
                sz = attrs['size']
            elif m is not None:
                if '.weight' in pname and hasattr(m, 'weight') and m.weight is not None:
                    sz = tuple(m.weight.shape)
                elif '.bias' in pname and hasattr(m, 'bias') and m.bias is not None:
                    sz = tuple(m.bias.shape)

            self._param_shapes.append(sz)

            in_dim = out_dim = 0
            if sz is not None  and len(sz)==2:
                if name=='embed':
                    out_dim = int(sz[1])
                else:
                    out_dim, in_dim = int(sz[0]), int(sz[1])

            # crude activation tag from name (works with autograd fn names or layer paths)
            act_id = 0
            low = pname.lower()
            if 'gelu' in low:
                act_id = PRIM2IDX_LM['gelu']
            elif 'relu' in low:
                act_id = PRIM2IDX_LM['relu']

            type_id = PRIM2IDX_LM.get(name, PRIM2IDX_LM['linear'])
            self.node_feat[i, 0] = type_id
            self.node_feat[i, 1] = in_dim
            self.node_feat[i, 2] = out_dim
            self.node_feat[i, 3] = act_id

            # Keep a human-friendly node_info row
            self.node_info[0].append([i, pname, name, sz, False, False])

        # adjacency & edges (virtual edges kept as weights)
        self._Adj = torch.tensor(self._Adj, dtype=t_long)
        ind = torch.nonzero(self._Adj)  # [E, 2]
        self.edges = torch.cat((ind, self._Adj[ind[:, 0], ind[:, 1]].view(-1, 1)), dim=1)  # [E, 3]

        # ----- edge types: forward/residual/posenc -----
        E = self.edges.size(0)
        edge_types = torch.zeros(E, 1, dtype=t_long)

        # residual if dst is an Add node with >1 incoming
        add_nodes = set()
        for j, n in enumerate(self._nodes):
            base = n['param_name'].split('Backward')[0]
            if MODULES_LM.get(base, None) == 'sum':
                add_nodes.add(j)
        indeg = torch.bincount(self.edges[:, 1], minlength=self.n_nodes)
        is_res = torch.tensor([1 if (int(d) in add_nodes and int(indeg[int(d)]) > 1) else 0
                               for d in self.edges[:, 1]], dtype=t_long).view(-1, 1)
        edge_types += is_res  # residual=1

        # positional-encoding links: from pos-like embedding nodes into Add
        # replace your current pos_like block with this
        def _is_pos_like(n):
            pname = (n.get('param_name') or '').lower()
            # catch common variants: GPT-2 (wpe), position_bias, RoPE/rotary
            if any(k in pname for k in ('pos', 'position', 'wpe', 'position_bias', 'rope', 'rotary')):
                return True
            m = n.get('module', None)
            if isinstance(m, nn.Embedding):
                # heuristic: positional vocab is "small", token vocab is large
                num_emb = getattr(m, 'num_embeddings', 10 ** 9)
                return num_emb <= 8192 and 'wte' not in pname  # avoid token emb
            return False

        pos_like = {i for i, n in enumerate(self._nodes) if _is_pos_like(n)}
        mask_pos = torch.tensor(
            [1 if (int(s) in pos_like and int(d) in add_nodes) else 0
             for s, d in self.edges[:, :2]],
            dtype=t_long
        ).view(-1, 1)
        edge_types[mask_pos.squeeze(1) == 1] = 2  # posenc = 2

        self.edges = torch.cat([self.edges, edge_types], dim=1)  # [E, 4]

    def _named_modules(self):
        r"""
        Helper function to automatically build the graphs.
        :return: dictionary of named modules, where the key is the module_name.parameter_name and
        the value is a tuple of (parameter, module)
        """
        modules = {}
        for n, m in self.model.named_modules():
            for np, p in m.named_parameters(recurse=False):
                if p is None:
                    continue
                key = n + '.' + np
                if key in modules:
                    assert id(p) == id(modules[key][0]), (n, np, p.shape, modules[key][0].shape)
                    continue
                modules[key] = (p, m)

        n_tensors = len(list(self.model.named_parameters()))
        params = dict(self.model.named_parameters())

        if len(modules) > n_tensors:
            if self._verbose:
                print('WARNING: number of tensors found in all submodules ({}) > number of unique tensors ({}). '
                      'This is fine in some models with tied weights.'.format(len(modules), n_tensors))
                for m in modules:
                    if m not in params:
                        print('\t module {} ({}) not in params'.format(m, modules[m][0].shape))
        else:
            assert len(modules) == n_tensors, (len(modules), n_tensors)

        return modules

    def _nx_graph_from_adj(self, A=None, remove_ve=True):
        A0 = self._Adj if A is None else A
        A = A0.data.cpu().numpy() if isinstance(A0, torch.Tensor) else A0.copy()  # <- copy()
        if remove_ve:
            A[A > 1] = 0
        else:
            A = A.astype(np.float32)
            ind = A > 1
            A[ind] = 1. / A[ind]
        self.nx_graph = nx.DiGraph(A)
        return self.nx_graph

    def properties(self, undirected=True, key=('avg_degree', 'avg_path')):
        """
        Computes graph properties.
        :param undirected: ignore edge direction when computing graph properties.
        :param key: a tuple/list of graph properties to estimate.
        :return: dictionary with property names and values.
        """
        G = self._nx_graph_from_adj()
        if undirected:
            G = G.to_undirected()
        props = {}
        for prop in key:
            if prop == 'avg_degree':
                degrees = dict(G.degree())
                assert len(degrees) == self._Adj.shape[0] == self.n_nodes, 'invalid graph'
                props[prop] = sum(degrees.values()) / self.n_nodes
            elif prop == 'avg_path':
                props[prop] = nx.average_shortest_path_length(G)
            else:
                raise NotImplementedError(prop)

        return props

    def visualize(self, node_size=50, figname=None, figsize=(10, 10), with_labels=False, **nx_args):
        import matplotlib
        if figname is not None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        G = self._nx_graph_from_adj(remove_ve=True)
        pos = nx.spring_layout(G, seed=0)
        fig = plt.figure(figsize=figsize)
        nx.draw(G, pos, node_size=node_size, arrows=True, **nx_args)

        if with_labels:
            labels = {i: self._nodes[i]['param_name'] for i in range(len(self._nodes))}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.axis('off')
        if figname is not None:
            plt.savefig(figname + '.png', dpi=180, bbox_inches='tight')
        else:
            plt.show()

def get_conv_name(module, op_name):
    if op_name.find('bias') >= 0:
        return 'bias'
    elif isinstance(module, nn.Conv2d) and module.groups > 1:
        return 'dil_conv' if min(module.dilation) > 1 else 'sep_conv'
    return 'conv'

_CONV1D_CLS = getattr(transformers, 'Conv1D', None)
MODULES_LM = {
    nn.Embedding:              lambda m, op: 'embed',
    nn.MultiheadAttention:     lambda m, op: 'msa',
    nn.LayerNorm:              lambda m, op: 'ln',
    nn.Linear:                 lambda m, op: 'linear',
    nn.TransformerEncoderLayer:lambda m, op: 'msa',
    nn.TransformerEncoder:     lambda m, op: 'msa',
    **({_CONV1D_CLS: (lambda m, op: 'linear')} if _CONV1D_CLS is not None else {}),
    'Softmax':                 'msa',
    'ScaledDotProductAttention': 'msa',
    'ScaledDotProductEager':    'msa',
    'FlashAttention':           'msa',  # optional
    'Add':                     'sum',
    'Cat':                     'concat',
    'input':                   'input',
}

PRIMITIVES_LM = [
    'input',    # 0
    'embed',    # token/pos embeddings
    'msa',      # multi-head self-attention (concept)
    'linear',   # linear (q/k/v/out, FFN linears)
    'mlp',      # optional: treat FFN block as 'mlp' if you want to collapse two linears
    'ln',       # layer norm
    'gelu',
    'relu',
    'sum',      # residual/add
    'concat',   # rare but keep
]