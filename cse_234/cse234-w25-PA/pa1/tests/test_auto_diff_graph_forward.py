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


def test_identity():
    x = ad.Variable("x1")
    evaluator = ad.Evaluator(eval_nodes=[x])

    x_val = torch.rand(4, 5, dtype=torch.float32)
    check_evaluator_output(evaluator, input_values={x: x_val}, expected_outputs=[x_val])


def test_add():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.add(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[torch.tensor([[1.8, 2.7, 0.4, 3.4], [0.9, 6.6, -2.6, 6.2]])],
    )


def test_add_by_const():
    x1 = ad.Variable("x1")
    y = ad.add_by_const(x1, 2.7)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[torch.tensor([[1.7, 4.7, 3.2, 6.1], [3.0, 2.7, -3.1, 5.8]])],
    )


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]])
        ],
    )


def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 2.7)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[
            torch.tensor([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]])
        ],
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        },
        expected_outputs=[
            torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]])
        ],
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]])],
    )

def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    evaluator = ad.Evaluator(eval_nodes=[y])

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])


    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
        },
        expected_outputs=[
            torch.tensor([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]])
        ],
    )


def test_graph():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    x3 = ad.Variable("x3")
    trans_x2 = ad.transpose(x2, 1, 0)
    y = ad.matmul(x1, trans_x2) / 10 + x3
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            x3: torch.tensor([[2.71, 3.14], [3.87, -4.0]]),
        },
        expected_outputs=[torch.tensor([[2.565, 5.614], [4.012, -4.877]])],
    )


if __name__ == "__main__":
    test_identity()
    test_add()
    test_add_by_const()
    test_mul()
    test_mul_by_const()
    test_div()
    test_div_by_const()
    test_matmul()

    test_graph()