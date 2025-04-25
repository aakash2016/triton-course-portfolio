import argparse
import json
import math
import numpy as np

fp_map = {
    "float8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8
}

def model_training_cost_analysis_llama(model_config_path):
    with open(model_config_path, 'r') as file:
        config = json.load(file)

    ## PARAMETERS
    total_params = 0
    per_layer_params = 0
    total_params += config['vocab_size'] * config['hidden_size'] # input embeddings
    per_layer_params += config['hidden_size'] # rms norm
    per_layer_params += 4 * config['hidden_size'] ** 2 # self-attention
    per_layer_params += config['hidden_size']  # rms norm
    per_layer_params += 2 * config['hidden_size'] * config['intermediate_size']  # mlp (silu)
    total_params += per_layer_params * config['num_hidden_layers']
    total_params += config['hidden_size']  # rms norm
    if not config['tie_word_embeddings']:
        total_params += config['vocab_size'] * config['hidden_size']  # reverse embeddings

    ## FLOPs -> single transformer layer
    flops_layer_TF = 0
    flops_layer_TF += 6 * 1 * config['max_sequence_length'] * config['hidden_size'] ** 2 # QKV projection
    flops_layer_TF += 2 * 1 * (config['max_sequence_length'] ** 2) * config['hidden_size']  # matmul
    flops_layer_TF += 3 * 1 * (config['max_sequence_length'] ** 2) * config['num_attention_heads']  # softmax
    flops_layer_TF += 2 * 1 * (config['max_sequence_length'] ** 2) * config['hidden_size'] # matmul (values)
    flops_layer_TF += 2 * 1 * config['max_sequence_length'] * config['hidden_size'] ** 2  # attention output
    flops_layer_TF += 1 * config['max_sequence_length'] * config['hidden_size'] # residual connection
    flops_layer_TF += 4 * 1 * config['max_sequence_length'] * config['hidden_size'] * config['intermediate_size'] # up and down proj
    flops_layer_TF += 5 * 1 * config['max_sequence_length'] * config['intermediate_size'] # silu fn and x
    flops_layer_TF /= 1e12  # TFLOPs

    ## MEMORY -> single transformer layer, FP16 precision
    bytes = fp_map[config['torch_dtype']]
    peak_memory_GB = 0

    # Activation Memory with checkpoint between layers.
    peak_memory_GB += 1 * config['max_sequence_length'] * config['hidden_size'] * bytes # 2 bytes

    # model weights transformer layer
    peak_memory_GB += per_layer_params * bytes

    peak_memory_GB /= 1e9  # GBs
    return total_params, flops_layer_TF, peak_memory_GB

def model_training_cost_analysis_deepseek(model_config_path):
    with open(model_config_path, 'r') as file:
        config = json.load(file)

    # Deepseek only differs in MHA and MoE parts - rest all same
    ## PARAMETERS - 671B total parameters with 37B activated for each token.
    total_params = 0
    active_params_per_layer = 0
    mhla_per_layer_params = 0
    total_params += config['vocab_size'] * config['hidden_size'] # input embeddings

    # MHLA
    mhla_per_layer_params += config['hidden_size'] # rms norm
    mhla_per_layer_params += config['hidden_size'] * config['q_lora_rank'] # query-projection
    mhla_per_layer_params += config['num_attention_heads'] * config['qk_nope_head_dim'] * config['q_lora_rank']  # q up-projection matrices
    mhla_per_layer_params += config['q_lora_rank'] * config['qk_rope_head_dim'] * config['num_attention_heads']  # rope q

    mhla_per_layer_params += config['hidden_size'] * config['kv_lora_rank'] # kv-projection
    mhla_per_layer_params += config['hidden_size'] * config['qk_rope_head_dim'] # rope k
    mhla_per_layer_params += 2 * config['num_attention_heads'] * config['qk_nope_head_dim'] * config['kv_lora_rank'] # kv up-projection matrices
    mhla_per_layer_params += config['hidden_size'] * config['num_attention_heads'] * config['qk_nope_head_dim']  # W_o
    active_params_per_layer += mhla_per_layer_params

    # first 3 dense layers
    dense_layer_params = 2 * config['hidden_size'] * config['intermediate_size']  # Dense FFN (up + down)
    total_params += dense_layer_params * config['first_k_dense_replace']

    # MoE
    total_moe_layes = int((config['num_hidden_layers'] - config['first_k_dense_replace']) / config['moe_layer_freq'])
    mlp_params = 3 * config['hidden_size'] * config['moe_intermediate_size'] + config['hidden_size'] # (mlp (silu), gate) + rms
    moe_params = (config['n_shared_experts'] + config['n_routed_experts']) * mlp_params # one expert
    total_params += moe_params * total_moe_layes # all experts

    # if interleaved MoE --> if moe_layer_freq > 1 -> more dense
    remaining_dense_layers = config['num_hidden_layers'] - config['first_k_dense_replace'] - total_moe_layes
    total_params += dense_layer_params * remaining_dense_layers

    # MoE active params per layer
    active_moe_params = (config['num_experts_per_tok'] + config['n_shared_experts']) * mlp_params
    active_params_per_layer += active_moe_params / config['moe_layer_freq'] + dense_layer_params * (1 - 1/config['moe_layer_freq'])

    total_params += mhla_per_layer_params * config['num_hidden_layers']
    total_params += config['hidden_size']  # rms norm
    if not config['tie_word_embeddings']:
        total_params += config['vocab_size'] * config['hidden_size']  # reverse embeddings

    ## FLOPs -> single transformer layer = MoE FLOPs + MHLA Flops
    flops_layer_TF = 0

    # MHLA
    flops_layer_TF += 2 * config['max_position_embeddings'] * config['hidden_size'] * config['q_lora_rank']  # Q down proj
    flops_layer_TF += 2 * config['q_lora_rank'] * config['num_attention_heads'] * (config['qk_nope_head_dim'] + config['qk_rope_head_dim'])  # Q up proj

    flops_layer_TF += 2 * config['max_position_embeddings'] * config['hidden_size'] * (
                config['kv_lora_rank'] + config['qk_rope_head_dim'])  # KV down proj + rope
    flops_layer_TF += (2 * config['max_position_embeddings'] * config['kv_lora_rank'] *
                       config['num_attention_heads'] * 2 * config['qk_nope_head_dim'])  # KV_b
    flops_layer_TF += 2 * (config['max_position_embeddings'] ** 2) * config['num_attention_heads'] * (
                config['qk_nope_head_dim'] + config['qk_rope_head_dim'])  # QK matmul
    flops_layer_TF += 3 * (config['max_position_embeddings'] ** 2) * config['num_attention_heads']  # softmax
    flops_layer_TF += 2 * (config['max_position_embeddings'] ** 2) * config['num_attention_heads'] * config['qk_nope_head_dim']  # values matmul
    flops_layer_TF += 2 * config['max_position_embeddings'] * config['hidden_size'] * config['num_attention_heads'] * config['qk_nope_head_dim']  # W_o

    # MoE
    flops_layer_TF += 2 * config['max_position_embeddings'] * config['hidden_size'] * config['n_routed_experts']  # gate
    flops_layer_TF += (4 * config['max_position_embeddings'] * config['hidden_size'] * config['moe_intermediate_size'] *
                       (config['num_experts_per_tok'] + config['n_shared_experts']))  # active experts
    flops_layer_TF += 5 * config['max_position_embeddings'] * config['moe_intermediate_size'] * (
                config['num_experts_per_tok'] + config['n_shared_experts'])  # silu

    flops_layer_TF += 2 * config['max_position_embeddings'] * config['hidden_size']  # residual connections
    flops_layer_TF /= 1e12  # TFLOPs

    ## MEMORY -> single transformer layer, FP16 precision
    bytes = fp_map[config['torch_dtype']]
    peak_memory_GB = 0
    peak_memory_GB += config['max_position_embeddings'] * config['hidden_size'] * bytes # 2 bytes # activation Memory with checkpoint
    peak_memory_GB += active_params_per_layer * bytes # model weights transformer layer
    peak_memory_GB /= 1e9  # GBs

    return total_params, flops_layer_TF, peak_memory_GB

def get_optimal_N_D_from_cost(cost_budget):
    """
    cost_budget:  a monetary training budget (in dollars)
    Returns:
        N: Optimal total model parameters (in absolute numbers)
        D: Optimal number of training tokens (in absolute numbers)
        training_budget_flops: Effective total training FLOPs (in FLOPs)
        best_gpu: name of the selected GPU (one of 'A100', 'V100', 'T4')
    """
    # GPU specs
    gpu_options = {
        'A100': {'cost_per_hour': 4.0, 'peak_flops': 312},
        'V100': {'cost_per_hour': 2.5, 'peak_flops': 125},
        'T4': {'cost_per_hour': 1.0, 'peak_flops': 65}
    }

    min_loss = float('inf')
    best_gpu, training_budget_flops = None, None
    N_array = 10 ** np.arange(8, 13, step=2**-6) # model sizes
    for gpu in gpu_options:
        compute = get_total_compute_available(
            cost_budget,
            gpu_options[gpu]["cost_per_hour"],
            gpu_options[gpu]["peak_flops"]
        )

        # num training-tokens range
        D_array = compute / (6 * N_array)

        # evaluate the loss in each case
        losses = loss_scaling_law(N_array, D_array)
        best_idx = np.argmin(losses)

        if losses[best_idx] < min_loss:
            min_loss = losses[best_idx]
            N, D = int(N_array[best_idx]), int(D_array[best_idx])
            training_budget_flops = compute
            best_gpu = gpu

    return N, D, training_budget_flops, best_gpu

def get_total_compute_available(budget, cost_hour, peak_performance, mfu=0.4):
    hours = budget / cost_hour
    flops_per_second = peak_performance * mfu * 1e12
    return flops_per_second * hours * 3600

def loss_scaling_law(Ns, Ds):
    losses = 406.4/np.power(Ns, 0.34) + 410.7/np.power(Ds, 0.29) + 1.69
    return losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
