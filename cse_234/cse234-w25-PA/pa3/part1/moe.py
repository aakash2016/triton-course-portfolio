import numpy as np

from mpiwrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b"""

    def __init__(self, in_features, out_features):
        # Use default RNG for all other operations - no need for context
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Expert network with one hidden layer and ReLU activation"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context('expert'):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Routes inputs to experts using softmax-based gating"""

    def __init__(self, input_dim, num_experts):
        # Router should be consistent across all ranks, so use default RNG
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)

        # Softmax for routing probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Select top-k experts
        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)

        # Normalize gates to sum to 1
        gates = gates / np.sum(gates, axis=1, keepdims=True)

        return indices, gates


class ShardedLinear:
    """
    Linear layer that is sharded across processes
    Each process only holds a portion of the weight matrix
    
    Requires that out_features is evenly divisible by the world size
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Assert that out_features is evenly divisible by world_size
        assert out_features % self.world_size == 0, f"Output features ({out_features}) must be evenly divisible by world size ({self.world_size})"

        # Calculate the local output dimension
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size

        # Calculate output offset for this rank (simple with even division)
        self.output_offset = self.rank * self.local_out_features

        # Initialize local weights and bias
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        # Handle empty batch case
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global))

        # Create a buffer for the full output
        # result = np.zeros((x.shape[0], self.out_features_global), dtype=np.float32)

        # TODO: Produce the result of sharded linear layer.
        send_data = np.dot(x, self.weight) + self.bias
        recv_outputs = mpi.allgather(send_data) # collects these partial outputs from all processes
        return np.concatenate(recv_outputs, axis=1)


class ShardedExpert:
    """Expert network with one hidden layer and ReLU activation, sharded across processes"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context('expert'):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class MoE_TP:
    """
    Distributed Mixture of Experts using MPI for tensor parallelism
    
    TP-style MoE:
    - Each process holds a portion of every expert (sharded experts)
    - Router is replicated on all processes
    - All-to-all and all-gather communication patterns for processing
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Total number of experts in the model
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Create router (replicated on all processes)
        with rng_context('router'):
            self.router = Router(input_dim, num_experts)

        # Create sharded experts - each expert is sharded across all processes
        with rng_context('expert'):
            self.experts = [ShardedExpert(input_dim, hidden_dim, output_dim)
                            for _ in range(num_experts)]

        # print(f"[Rank {self.rank}] Holding portions of all {num_experts} experts")

    def forward(self, x):
        """
        Distributed forward pass through the MoE model using tensor parallelism
        with optimized batch processing

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        # outputs = np.zeros((batch_size, self.output_dim))

        # TODO: Implement the TP-style MoE forward pass.
        # 1. Compute the routing indices and gates for each input
        indices, gates = self.router(x, self.topk)  # 8 experts, top k = 2
        # 2. Process experts parallel with TP style.
        # I will split x along batch dim (Megatron-LM like)
        local_x = np.array_split(x, self.num_experts, axis=0)[self.rank]  # static split, (1, 32)

        # local adjust indices and gates for local slice
        local_bs = local_x.shape[0]
        local_start = self.rank * local_bs
        local_gates = gates[local_start:local_start+local_bs]
        local_indices = indices[local_start:local_start+local_bs]
        expert_outputs = np.zeros((local_bs, self.output_dim))
        for b in range(local_bs):
            for k in range(self.topk):
                expert_idx = local_indices[b, k]
                gate = local_gates[b, k]
                expert_out = self.experts[expert_idx](local_x[b:b + 1])
                expert_outputs[b] += gate * expert_out[0]

        outputs = mpi.allreduce(expert_outputs)
        return outputs

    def __call__(self, x):
        return self.forward(x)


class SimpleMoE:
    """
    Simple reference implementation of Mixture of Experts.
    
    This class implements a basic MoE model that routes inputs to a subset
    of experts and combines their outputs using learned gating weights.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Number of expert networks
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        # Create router network
        with rng_context('router'):
            self.router = Router(input_dim, num_experts)

        # Create expert networks
        with rng_context('expert'):
            self.experts = [Expert(input_dim, hidden_dim, output_dim)
                            for _ in range(num_experts)]

    def forward(self, x):
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Get expert assignments and gates
        indices, gates = self.router(x, self.topk)

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # Compute weighted combination of expert outputs
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i:i + 1]  # (1, input_dim)
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_EP:
    """
    Distributed Mixture of Experts using MPI for expert parallelism
    
    EP-style MoE: 
    Each process hosts exactly one expert. Router is replicated on all processes.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # Total number of processes = number of experts
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.get_rank()

        # Create router (replicated on all processes)
        with rng_context('router'):
            self.router = Router(input_dim, self.num_experts)

        # Create only one expert per process
        with rng_context('expert_with_rank'):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Distributed forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # TODO: Implement the forward pass.
        # 1. Compute the routing indices and gates for each input
        indices, gates = self.router(x, self.topk)

        # 2. Process local inputs with this expert (within the device)
        local_inputs = []
        for i in range(batch_size):
            for k in range(self.topk):
                if indices[i, k] == self.rank:
                    local_inputs.append(x[i])
        local_x = np.array(local_inputs) if local_inputs else np.empty((0, self.input_dim))
        expert_output = self.expert(local_x) if local_x.size > 0 else np.empty((0, self.output_dim))

        # 3. Communicate between devices to get the outputs from all experts
        send_data = [expert_output] * self.num_experts
        recv_outputs = mpi.alltoall(send_data) # each process ends up with all expert outputs, which is what you need for MoE.

        expert_output_pos = [0] * self.num_experts
        for i in range(batch_size):
            for k in range(self.topk):
                expert_idx = indices[i, k]
                outputs[i] += gates[i, k] * recv_outputs[expert_idx][expert_output_pos[expert_idx]]
                expert_output_pos[expert_idx] += 1

        return outputs

    def __call__(self, x):
        return self.forward(x)
