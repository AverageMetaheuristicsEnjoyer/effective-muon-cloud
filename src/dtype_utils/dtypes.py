import torch
import torch.nn as nn
from typing import Dict, Any, Optional


def get_model_classes():
    classes = [nn.Linear, nn.Embedding]
    
    try:
        from models.llama import RMSNorm, LlamaBlock
        classes.extend([RMSNorm, LlamaBlock])
    except ImportError:
        pass

    try:
        from models.base import LayerNorm
        classes.append(LayerNorm)
    except ImportError:
        pass
    
    return tuple(classes)


def register_activation_hooks(model, distributed_backend) -> Dict[str, Dict[str, Any]]:
    if not distributed_backend.is_master_process():
        return {}
    
    activation_dtypes = {}
    model_classes = get_model_classes()
    
    def make_hook(name: str):
        def hook(module, input, output):
            if isinstance(input, tuple):
                inp = input[0]
            else:
                inp = input
            
            if isinstance(output, dict):
                out = output.get("logits", list(output.values())[0])
            elif isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            
            activation_dtypes[name] = {
                "input_dtype": inp.dtype if hasattr(inp, "dtype") else "N/A",
                "output_dtype": out.dtype if hasattr(out, "dtype") else "N/A",
                "input_shape": list(inp.shape) if hasattr(inp, "shape") else "N/A",
                "output_shape": list(out.shape) if hasattr(out, "shape") else "N/A",
            }
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, model_classes):
            module.register_forward_hook(make_hook(name))
    
    return activation_dtypes


def print_model_dtypes(model, distributed_backend, max_print: int = 10):
    if not distributed_backend.is_master_process():
        return
    
    print("\n=== PARAMETER DTYPES ===")
    dtype_counts = {}
    printed = 0
    
    for name, param in model.named_parameters():
        dtype = param.dtype
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        if printed < max_print or param.numel() > 1e6:
            print(f"  {name}: {dtype}  shape: {list(param.shape)}")
            printed += 1
    
    print("\n  Summary:")
    for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {dtype}: {count} parameters")
    print("=" * 60 + "\n")


def print_optimizer_dtypes(opt, distributed_backend):
    if not distributed_backend.is_master_process():
        return
    
    print("\n=== OPTIMIZER DTYPES ===")
    print(f"  Type: {opt.__class__.__name__}")
    
    if hasattr(opt, "qargs"):
        print(f"  FP8 Config:")
        print(f"    first_order_bit: {opt.qargs.first_order_bit}")
        print(f"    second_order_bit: {opt.qargs.second_order_bit}")
    
    if hasattr(opt, "state"):
        state_dtypes = {}
        for i, param_group in enumerate(opt.param_groups):
            for param in param_group["params"][:3]:
                param_state = opt.state.get(param, {})
                for state_name, state_val in param_state.items():
                    if isinstance(state_val, torch.Tensor):
                        dtype = state_val.dtype
                        key = f"Group {i} — {state_name}"
                        state_dtypes[key] = dtype
        
        print("  State Tensors:")
        for name, dtype in state_dtypes.items():
            print(f"    {name}: {dtype}")
    print("=" * 60 + "\n")


def print_gradient_dtypes(model, distributed_backend, max_print: int = 10):
    if not distributed_backend.is_master_process():
        return
    
    print("\n=== GRADIENT DTYPES ===")
    dtype_counts = {}
    printed = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            dtype = param.grad.dtype
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            
            if printed < max_print:
                print(f"  {name}: {dtype}  shape: {list(param.grad.shape)}")
                printed += 1
        else:
            if printed < max_print:
                print(f"  {name}: NO GRAD")
                printed += 1
    
    print("\n  Summary:")
    for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
        print(f"    {dtype}: {count} gradients")
    print("=" * 60 + "\n")


def print_activation_dtypes(activation_dtypes: Dict[str, Dict[str, Any]], 
                            distributed_backend, 
                            max_print: int = 10):
    if not distributed_backend.is_master_process():
        return
    
    if not activation_dtypes:
        print("\n=== ACTIVATION DTYPES ===\n  No hooks registered\n" + "=" * 60 + "\n")
        return
    
    print("\n=== ACTIVATION DTYPES (from hooks) ===")
    for i, (name, info) in enumerate(list(activation_dtypes.items())[:max_print]):
        print(f"  [{i+1}] {name}:")
        print(f"      input:  {info['input_dtype']}  {info['input_shape']}")
        print(f"      output: {info['output_dtype']}  {info['output_shape']}")
    
    if len(activation_dtypes) > max_print:
        print(f"  ... and {len(activation_dtypes) - max_print} more modules")
    print("=" * 60 + "\n")


def print_memory_usage(distributed_backend, device: str = "cuda", label: str = ""):
    if not distributed_backend.is_master_process():
        return
    if device != "cuda" or not torch.cuda.is_available():
        return
    
    label_str = f" ({label})" if label else ""
    print(f"\n=== MEMORY USAGE{label_str} ===")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    print(f"  Peak:      {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
    print("=" * 60 + "\n")