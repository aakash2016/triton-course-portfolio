from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)


def test_graph():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")
    trans_x2 = ad.transpose(x2, 1, 0)
    y = ad.matmul(x1, trans_x2) / 10 * x3
    x1_grad, x2_grad, x3_grad = ad.gradients(y, nodes=[x1, x2, x3])
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad, x3_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            x3: torch.tensor([[2.71, 3.14], [3.87, -4.0]]),
        },
        expected_outputs=[
            torch.tensor(
                [[0.9472, 2.2621, 0.9777, 0.9734], [0.8436, -2.3691, -1.3187, -1.24]]
            ),
            torch.tensor(
                [[-0.1549, 0.542, -2.1091, 2.1211], [-0.434, 0.628, 2.477, -0.1724]]
            ),
            torch.tensor([[-0.145, 2.474], [0.142, -0.877]]),
        ],
    )


def test_gradient_of_gradient():
    x1 = ad.Variable(name="x1")
    x2 = ad.Variable(name="x2")
    y = x1 * x1 + x1 * x2

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
    grad_x1_x1, grad_x1_x2 = ad.gradients(grad_x1, [x1, x2])
    grad_x2_x1, grad_x2_x2 = ad.gradients(grad_x2, [x1, x2])

    evaluator = ad.Evaluator(
        [y, grad_x1, grad_x2, grad_x1_x1, grad_x1_x2, grad_x2_x1, grad_x2_x2]
    )
    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-1.8, 5.4, 0.2, 11.56], [0.27, 0.0, 15.08, 19.22]]),
            torch.tensor([[0.8, 4.7, 0.9, 6.8], [1.2, 6.6, -8.4, 9.3]]),
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            2 * torch.ones((2, 4), dtype=torch.float32),
            1 * torch.ones((2, 4), dtype=torch.float32),
            1 * torch.ones((2, 4), dtype=torch.float32),
            torch.zeros((2, 4), dtype=torch.float32),
        ],
    )


if __name__ == "__main__":
    test_graph()
    test_gradient_of_gradient()