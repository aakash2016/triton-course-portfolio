import numpy as np
from mpi4py import MPI
from rng import get_rng, rng_context, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time

# Example usage
def run_moe(
    moe_type="tp", 
    batch_size=8, 
    feature_dim=32, 
    hidden_dim=128,
    output_dim=64,
    topk=2
):
    """
    Unified function to run different types of MoE models
    
    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    # Get number of experts based on MPI world size
    num_experts = mpi.get_size()
    
    # Generate input data
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        # Synchronize the input data across all processes
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)
    
    # Create appropriate MoE model
    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)
    
    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )
    
    # Run forward pass
    # Warm up
    _ = moe(X)
    
    # Measure time
    N = 10
    start_time = time.time()
    for _ in range(N):
        outputs = moe(X)
    end_time = time.time()
    avg_duration_ms = 1000 * (end_time - start_time) / N
    avg_throughput = batch_size / (avg_duration_ms / 1000)  # samples/sec

    # Print timing information
    if mpi.get_rank() == 0:
        print(f"Forward pass time for {moe_type} MoE: {avg_duration_ms} ms")

    return dict(
        avg_duration_ms=avg_duration_ms,
        avg_throughput=avg_throughput,
        output_shape=outputs[0].shape # rank 0 output shape for consistency
    )

    
def benchmark_moe():
    k=4
    batch_size=8
    output_dim=64

    # Test simple MoE
    simple_result = run_moe(moe_type="simple", batch_size=batch_size, output_dim=output_dim, topk=k)
    assert simple_result['output_shape'] == (output_dim,)
    print(f"Simple MoE: {simple_result['avg_duration_ms']} ms")

    # Test TP MoE
    tp_result = run_moe(moe_type="tp", batch_size=batch_size, output_dim=output_dim, topk=k)
    assert tp_result['output_shape'] == (output_dim,)
    print(f"TP MoE: {tp_result['avg_duration_ms']} ms")

    # Test EP MoE
    ep_result = run_moe(moe_type="ep", batch_size=batch_size, output_dim=output_dim, topk=k)
    assert ep_result['output_shape'] == (output_dim,)
    print(f"EP MoE: {ep_result['avg_duration_ms']} ms")


if __name__ == "__main__":
    benchmark_moe()
