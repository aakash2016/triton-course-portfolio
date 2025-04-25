import ssl
ssl._create_default_https_context = ssl._create_stdlib_context  # for downloading mnist dataset

import torch
import numpy as np

from sklearn.utils import shuffle
from typing import Callable, Tuple, List
from sklearn.preprocessing import OneHotEncoder
from torchvision import datasets, transforms

import auto_diff as ad

max_len = 28

def linear_transform(
        ip: ad.Node,
        weight: ad.Node,
        bias: ad.Node
) -> ad.Node:
    """linear transformation: output = input @ weight + bias"""
    return ad.add(ad.matmul(ip, weight), bias)

def single_head_attention(
        X: ad.Node,
        W_q: ad.Node,
        W_k: ad.Node,
        W_v: ad.Node,
        W_o: ad.Node,
        model_dim: int
) -> ad.Node:
    """single-head attention mechanism"""
    Q = ad.matmul(X, W_q) # X [batch_size, seq_length, model_dim] * w [input_dim, model_dim]
    K = ad.matmul(X, W_k)
    V = ad.matmul(X, W_v)

    # compute attention weights, we will do softmax on dim 1 (sequence length dim)
    attention_scores = ad.div_by_const(
        ad.matmul(Q, ad.transpose(K, dim0=1, dim1=2)),
        pow(model_dim, 0.5)
    )
    attention_weights = ad.softmax(attention_scores, dim=1) # attention_weights [batch_size, seq_length, input_dim]

    # compute the attention output
    attention_output = ad.matmul(attention_weights, V) # attention_output [batch_size, seq_length, model_dim]

    # project the attention output using W_o
    attention_projected = ad.matmul(attention_output, W_o) # attention_projected [batch_size, seq_length, model_dim]
    return attention_projected

def encoder_layer(
        X: ad.Node,
        W_q: ad.Node,
        W_k: ad.Node,
        W_v: ad.Node,
        W_o: ad.Node,
        W_1: ad.Node,
        b_1: ad.Node,
        model_dim: int,
        eps
) -> ad.Node:
    """encoder layer"""
    attention_output = single_head_attention(X, W_q, W_k, W_v, W_o, model_dim) # attention [batch_size, seq_length, model_dim]

    # post-attention layernorm
    normalized_output = ad.layernorm(attention_output, normalized_shape=[model_dim], eps=eps)

    # linear + activation
    ffn_output = linear_transform(normalized_output, W_1, b_1) # ffn_output [batch_size, seq_length, model_dim]
    ffn_output = ad.relu(ffn_output)

    # post-ffn layernorm
    normalized_ffn_output = ad.layernorm(ffn_output, normalized_shape=[model_dim], eps=eps)
    return normalized_ffn_output

def transformer(
        X: ad.Node,
        nodes: List[ad.Node],
        model_dim: int,
        seq_length: int,
        eps,
        batch_size,
        num_classes,
        input_dim
) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, model_dim), denoting the input data.
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """

    # Encoder Layer
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes
    encoder_output = encoder_layer(X, W_Q, W_K, W_V, W_O, W_1, b_1, model_dim, eps)

    # average over the sequence length
    averaged = ad.div_by_const(ad.sum_op(encoder_output, dim=1), seq_length)  # averaged [batch_size, model_dim]

    # classification layer
    output = linear_transform(averaged, W_2, b_2) # output [batch_size, n_Classes]
    return output


def softmax_loss(
    Z: ad.Node,
    y_one_hot: ad.Node,
    batch_size: int
) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """

    softmax_vals = ad.softmax(Z, dim=1) # softmax_vals [batch_size, num_classes]
    loss = ad.div_by_const(ad.sum_op(ad.sum_op(y_one_hot * ad.log(softmax_vals), dim=1, keepdim=True), dim=0, keepdim=True), -batch_size)
    return loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Run an epoch of SGD for the logistic regression model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for logistic regression model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size> num_examples:continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
        
        # Compute forward and backward passes
        y_predict, loss, *grads = f_run_model(X_batch, y_batch, model_weights)
        grad_W_Q, grad_W_K, grad_W_V, grad_W_O, grad_W_1, grad_W_2, grad_b_1, grad_b_2 = grads

        # Update weights and biases
        model_weights[0] -= lr * grad_W_Q.sum(dim=0)  # W_Q
        model_weights[1] -= lr * grad_W_K.sum(dim=0)  # W_K
        model_weights[2] -= lr * grad_W_V.sum(dim=0)  # W_V
        model_weights[3] -= lr * grad_W_O.sum(dim=0)  # W_O
        model_weights[4] -= lr * grad_W_1.sum(dim=0)  # W_1
        model_weights[5] -= lr * grad_W_2            # W_2
        model_weights[6] -= lr * grad_b_1.sum(dim=(0, 1))  # b_1
        model_weights[7] -= lr * grad_b_2.sum(dim=0)   # b_2

        # Accumulate the loss
        total_loss += loss

    # Compute the average loss
    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)

    # return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params

    # TODO: Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-5 

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # Define the forward graph.
    X = ad.Variable("X")
    W_Q = ad.Variable("W_Q")
    W_K = ad.Variable("W_K")
    W_V = ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1 = ad.Variable("W_1")
    W_2 = ad.Variable("W_2")
    b_1 = ad.Variable("b_1")
    b_2 = ad.Variable("b_2")
    nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_op = transformer(
        X,
        nodes,
        model_dim,
        seq_length,
        eps,
        batch_size,
        num_classes,
        input_dim
    )

    y_predict: ad.Node = y_op # The output of the forward pass
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)
    
    # Construct the backward graph.
    # Create the evaluator.
    grads: List[ad.Node] = ad.gradients(loss, nodes=nodes) # Define the gradient nodes here
    evaluator = ad.Evaluator(eval_nodes=[y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(ip, Y, weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        # Fill in the mapping from variable to tensor

        result = evaluator.run(
            input_values={
                X: ip,
                y_groundtruth: Y,
                W_Q: weights[0],
                W_K: weights[1],
                W_V: weights[2],
                W_O: weights[3],
                W_1: weights[4],
                W_2: weights[5],
                b_1: weights[6],
                b_2: weights[7]
            }
        )
        return result

    def f_eval_model(X_val, weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                # Fill in the mapping from variable to tensor
                X: X_batch,
                W_Q: weights[0],
                W_K: weights[1],
                W_V: weights[2],
                W_O: weights[3],
                W_1: weights[4],
                W_2: weights[5],
                b_1: weights[6],
                b_2: weights[7]
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test = torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val),
        torch.tensor(W_K_val),
        torch.tensor(W_V_val),
        torch.tensor(W_O_val),
        torch.tensor(W_1_val),
        torch.tensor(W_2_val),
        torch.tensor(b_1_val),
        torch.tensor(b_2_val)
    ]
    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
