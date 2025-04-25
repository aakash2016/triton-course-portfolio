# Advantages of MoE

### MoE Performance in DeepSeek  
I experiment with the `moe_layer_freq` key in the deepseek config for my analyses.

#### Full MoE (MoE Stride = 1)  
- **Number of parameters**: 670.63 billion  
- **TFLOPs per transformer layer**: 2,345.48  
- **Peak memory cost per transformer layer**: 3.52 GB (only MoE, no dense) 

#### Interleaved MoE (MoE Stride = 2)  
- **Number of parameters**: 350.01 billion  
- **TFLOPs per transformer layer**: 2,345.48  
- **Peak memory cost per transformer layer**: 3.38 GB (using MoE + dense average)

### Key Insights
- Both configurations have **identical TFLOPs per transformer layer** (~2,345.48 TFLOPs). However, the **Parameter count is reduced by ~48%** (from 670.63B to 350.01B) when using stride 2. This suggests that MoE models maintain similar computational workloads despite using fewer active parameters per token. 
- Hence, MoE achieves a better scaling law, meaning more capability per unit of compute, which aligns with DeepSeek’s cost-effective $5M claim.  
- We can say MoE models are more compute-efficient as it follows a better scaling law—while the number of parameters increases drastically, the actual compute demand only rises mildly.  
