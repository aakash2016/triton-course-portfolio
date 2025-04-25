import torch
import time
import numpy as np
from typing import Callable, Dict, List, Tuple, Union
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad
from fused_ops import matmul_layernorm, matmul_softmax

@dataclass
class TestCase:
    """Represents a test configuration"""
    name: str
    shapes: Union[Tuple[Tuple[int, ...], Tuple[int, ...]], List[Tuple[Tuple[int, ...], Tuple[int, ...]]]]
    num_runs: int = 100

def generate_tensors(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate test tensors with given shapes."""
    return torch.randn(*shape1), torch.randn(*shape2)

def time_operation(op_fn: Callable, num_runs: int = 100, warmup: int = 10) -> Tuple[float, float]:
    """Time an operation with warmup runs and return mean and std of execution time."""
    # Warmup runs
    for _ in range(warmup):
        op_fn()
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        op_fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times), np.std(times)

def run_benchmark(test_case: TestCase) -> None:
    """Run benchmark for a given test case."""
    print(f"\n{'='*80}\nRunning benchmark: {test_case.name}\n{'='*80}")
    
    if not isinstance(test_case.shapes[0], tuple):
        # Single shape pair case
        shapes = [test_case.shapes]
    else:
        shapes = test_case.shapes

    for shape1, shape2 in shapes:
        print(f"\nTesting shapes: {shape1} @ {shape2}")
        
        # Generate test data
        x1, x2 = generate_tensors(shape1, shape2)
        if torch.cuda.is_available():
            x1, x2 = x1.cuda(), x2.cuda()
        
        # Determine gradient shape based on matmul output shape
        grad_shape = x1.shape[:-1] + (x2.shape[-1],)
        y_grad = torch.ones(*grad_shape)
        if torch.cuda.is_available():
            y_grad = y_grad.cuda()

        # Create nodes
        node_x1 = ad.Variable("x1")
        node_x2 = ad.Variable("x2")
        node_grad = ad.Variable("grad")

        # Test LayerNorm fusion
        print("\nTesting MatMul + LayerNorm:")
        normalized_shape = [grad_shape[-1]]
        
        # Create computation graphs
        fused_y = matmul_layernorm(node_x1, node_x2, normalized_shape=normalized_shape)
        unfused_y = ad.layernorm(ad.matmul(node_x1, node_x2), normalized_shape=normalized_shape)
        
        # Create evaluators
        fused_eval = ad.Evaluator([fused_y])
        unfused_eval = ad.Evaluator([unfused_y])
        
        # Forward pass timing
        print("\nForward Pass:")
        fused_mean, fused_std = time_operation(
            lambda: fused_eval.run({node_x1: x1, node_x2: x2}),
            test_case.num_runs
        )
        unfused_mean, unfused_std = time_operation(
            lambda: unfused_eval.run({node_x1: x1, node_x2: x2}),
            test_case.num_runs
        )
        print(f"Fused:    {fused_mean*1000:.3f} ms ± {fused_std*1000:.3f} ms")
        print(f"Unfused:  {unfused_mean*1000:.3f} ms ± {unfused_std*1000:.3f} ms")
        print(f"Speedup:  {unfused_mean/fused_mean:.2f}x")
        
        # Backward pass timing
        fused_grads = fused_y.op.gradient(fused_y, node_grad)
        unfused_grads = ad.gradients(unfused_y, [node_x1, node_x2])
        
        fused_grad_eval = ad.Evaluator(fused_grads)
        unfused_grad_eval = ad.Evaluator(unfused_grads)
        
        print("\nBackward Pass:")
        fused_mean, fused_std = time_operation(
            lambda: fused_grad_eval.run({node_x1: x1, node_x2: x2, node_grad: y_grad}),
            test_case.num_runs
        )
        unfused_mean, unfused_std = time_operation(
            lambda: unfused_grad_eval.run({node_x1: x1, node_x2: x2, node_grad: y_grad}),
            test_case.num_runs
        )
        print(f"Fused:    {fused_mean*1000:.3f} ms ± {fused_std*1000:.3f} ms")
        print(f"Unfused:  {unfused_mean*1000:.3f} ms ± {unfused_std*1000:.3f} ms")
        print(f"Speedup:  {unfused_mean/fused_mean:.2f}x")

        # Test Softmax fusion
        print("\nTesting MatMul + Softmax:")
        
        # Create computation graphs
        fused_y = matmul_softmax(node_x1, node_x2, dim=-1)
        unfused_y = ad.softmax(ad.matmul(node_x1, node_x2), dim=-1)
        
        # Create evaluators
        fused_eval = ad.Evaluator([fused_y])
        unfused_eval = ad.Evaluator([unfused_y])
        
        # Forward pass timing
        print("\nForward Pass:")
        fused_mean, fused_std = time_operation(
            lambda: fused_eval.run({node_x1: x1, node_x2: x2}),
            test_case.num_runs
        )
        unfused_mean, unfused_std = time_operation(
            lambda: unfused_eval.run({node_x1: x1, node_x2: x2}),
            test_case.num_runs
        )
        print(f"Fused:    {fused_mean*1000:.3f} ms ± {fused_std*1000:.3f} ms")
        print(f"Unfused:  {unfused_mean*1000:.3f} ms ± {unfused_std*1000:.3f} ms")
        print(f"Speedup:  {unfused_mean/fused_mean:.2f}x")
        
        # Backward pass timing
        fused_grads = fused_y.op.gradient(fused_y, node_grad)
        unfused_grads = ad.gradients(unfused_y, [node_x1, node_x2])
        
        fused_grad_eval = ad.Evaluator(fused_grads)
        unfused_grad_eval = ad.Evaluator(unfused_grads)
        
        print("\nBackward Pass:")
        fused_mean, fused_std = time_operation(
            lambda: fused_grad_eval.run({node_x1: x1, node_x2: x2, node_grad: y_grad}),
            test_case.num_runs
        )
        unfused_mean, unfused_std = time_operation(
            lambda: unfused_grad_eval.run({node_x1: x1, node_x2: x2, node_grad: y_grad}),
            test_case.num_runs
        )
        print(f"Fused:    {fused_mean*1000:.3f} ms ± {fused_std*1000:.3f} ms")
        print(f"Unfused:  {unfused_mean*1000:.3f} ms ± {unfused_std*1000:.3f} ms")
        print(f"Speedup:  {unfused_mean/fused_mean:.2f}x")

if __name__ == "__main__":
    # Define test cases
    test_cases = [
        TestCase(
            name="2D Matrix Multiplication",
            shapes=[
                ((128, 64), (64, 128)),    # Small
                ((512, 256), (256, 512)),  # Medium
                ((1024, 512), (512, 1024)) # Large
            ],
            num_runs=100
        ),
        TestCase(
            name="3D Batch Matrix Multiplication",
            shapes=[
                ((32, 128, 64), (32, 64, 128)),      # Small batch
                ((64, 256, 128), (64, 128, 256)),    # Medium batch
                ((128, 512, 256), (128, 256, 512))   # Large batch
            ],
            num_runs=50
        ),
    ]

    # Run benchmarks
    for test_case in test_cases:
        run_benchmark(test_case)