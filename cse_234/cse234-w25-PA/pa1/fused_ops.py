from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        norm_dims = tuple(range(-len(node.normalized_shape), 0))

        # matmul
        mat_mul = torch.matmul(input_values[0], input_values[1])

        # layernorm
        mean_val = mat_mul.mean(dim=norm_dims, keepdim=True)
        diff = mat_mul - mean_val
        var_val = (diff ** 2).mean(dim=norm_dims, keepdim=True)
        return diff / torch.sqrt(var_val + node.eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # matmul op
        matrix_mul = matmul(node.inputs[0], node.inputs[1])

        # layernorm op
        dim = -len(node.normalized_shape)
        size = sum_op(ones_like(matrix_mul), dim=(dim,), keepdim=True)

        mean_val = mean(matrix_mul, dim=dim, keepdim=True)
        var_val = variance(matrix_mul, dim=dim, keepdim=True)
        c = power(sqrt(add_by_const(var_val, node.eps)), -1)
        normalised_mean = (matrix_mul - mean_val) / mul_by_const(power(var_val + node.eps, 1.5), 2)

        grad_mu = sum_op(output_grad, dim=tuple(range(dim, 0)), keepdim=True) * div(c, size)
        grad_var = sum_op(output_grad * normalised_mean, dim=tuple(range(dim, 0)), keepdim=True)

        # now we use chain rule
        dz_dt = (output_grad * c) - grad_mu - (grad_var * 2 * (matrix_mul - mean_val) / size)
        return [
            matmul(dz_dt, transpose(node.inputs[1], dim0=-1, dim1=-2)),
            matmul(transpose(node.inputs[0], dim0=-2, dim1=-1), dz_dt)
        ]

class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2

        if input_values[0].dim() == 3:
            mat_mul = torch.bmm(input_values[0], input_values[1])
        else:
            mat_mul = torch.matmul(input_values[0], input_values[1])

        max_value, _ = torch.max(mat_mul, dim=node.dim, keepdim=True)
        exp_mat = torch.exp(mat_mul - max_value)
        sum_exp = torch.sum(exp_mat, dim=node.dim, keepdim=True)
        return exp_mat / sum_exp

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # matmul
        matrix_mul = matmul(node.inputs[0], node.inputs[1])

        # softmax grad
        softmax_vals = softmax(matrix_mul, dim=node.dim)
        s_grad = softmax_vals * output_grad
        dz_dt = s_grad - softmax_vals * sum_op(s_grad, dim=node.dim, keepdim=True)

        # employing chain rule
        return [
            matmul(dz_dt, transpose(node.inputs[1], dim0=-1, dim1=-2)),
            matmul(transpose(node.inputs[0], dim0=-2, dim1=-1), dz_dt)
        ]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()