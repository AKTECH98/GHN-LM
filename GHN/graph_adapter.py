"""
Language Model Wrapper for GHN compatibility.

This module provides the LanguageModelWrapper class that makes language models
compatible with GHN's graph representation by filtering parameters and handling
tied weights properly.
"""

import torch
import torch.nn as nn


class LanguageModelWrapper:
    """Wrapper for language models to make them compatible with GHN's named_layered_modules."""
    
    def __init__(self, model):
        self.model = model
        # Store only tensor parameters, filtering out boolean/other types
        self._parameters = {}
        self._named_parameters = {}
        
        # Track tied weights to avoid None issues
        self._tied_weights = {}
        
        # Filter parameters to only include tensors
        for name, param in model.named_parameters():
            if isinstance(param, torch.Tensor):
                self._parameters[name] = param
                self._named_parameters[name] = param
                
                # Check for tied weights (same tensor object)
                param_id = id(param)
                if param_id in self._tied_weights:
                    self._tied_weights[param_id].append(name)
                else:
                    self._tied_weights[param_id] = [name]
    
    def named_parameters(self, recurse=True, prefix='', remove_duplicate=True):
        """Return only tensor parameters."""
        for name, param in self._named_parameters.items():
            yield prefix + name, param
    
    def parameters(self, recurse=True):
        """Return only tensor parameters."""
        for param in self._parameters.values():
            yield param
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Return modules, but only those with tensor parameters."""
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            
            # Only include modules that have tensor parameters
            for name, module in self.model.named_modules():
                if hasattr(module, 'parameters'):
                    has_tensor_params = any(isinstance(p, torch.Tensor) for p in module.parameters(recurse=False))
                    if has_tensor_params:
                        submodule_prefix = prefix + ('.' if prefix else '') + name
                        yield submodule_prefix, module
    
    def __getattr__(self, name):
        """Delegate all other attributes to the original model."""
        return getattr(self.model, name)
    
    def __call__(self, *args, **kwargs):
        """Delegate forward calls to the original model."""
        # Fix any None weights that might have been set by GHN
        self._fix_none_weights()
        return self.model(*args, **kwargs)
    
    def _fix_none_weights(self):
        """Fix None weights that might have been set by GHN, especially for tied weights."""
        for param_id, param_names in self._tied_weights.items():
            if len(param_names) > 1:  # This is a tied weight
                # Find the first non-None parameter
                non_none_param = None
                for name in param_names:
                    param = self._get_param_by_name(name)
                    if param is not None and isinstance(param, torch.Tensor):
                        non_none_param = param
                        break
                
                # If we found a non-None parameter, set all tied parameters to it
                if non_none_param is not None:
                    for name in param_names:
                        param = self._get_param_by_name(name)
                        if param is None or not isinstance(param, torch.Tensor):
                            self._set_param_by_name(name, non_none_param)
    
    def _get_param_by_name(self, name):
        """Get parameter by name from the original model."""
        parts = name.split('.')
        obj = self.model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        return getattr(obj, parts[-1], None)
    
    def _set_param_by_name(self, name, value):
        """Set parameter by name in the original model."""
        parts = name.split('.')
        obj = self.model
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)


