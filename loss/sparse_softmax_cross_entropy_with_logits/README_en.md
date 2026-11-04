# SparseSoftmaxCrossEntropyWithLogits

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term>     |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Computes the Softmax cross-entropy loss in the sparse class label scenario and outputs the gradient required for backpropagation at the same time. Unlike dense labels, labels directly provide the class index of each sample, so one-hot expansion is not required.

- Formula:

  Let features be $x$ (with the shape $[N, C]$) and labels be $y$ (with the shape $[N]$ and the value range $[0, C)$). For the $n$-th sample:

  $$
  \text{softmax}(x)_{n,c} = \frac{\exp(x_{n,c})}{\sum_{k=0}^{C-1}\exp(x_{n,k})}
  $$

  $$
  \text{loss}_n = -\log\big(\text{softmax}(x)_{n,y_n}\big)
  $$

  $$
  \text{backprop}_{n,c} = \text{softmax}(x)_{n,c} - \mathbb{1}\{c = y_n\}
  $$

  where:

  - $N$ is the batch count, and $C$ is the number of classes.
  - loss is the loss of each sample (a vector of length $N$).
  - backprop is the gradient of backpropagation (a matrix with the shape $[N, C]$).

## Parameters

| Parameter name   | Input/Output/Attribute | Description                                                         | Data type                 | Data format |
| -------- | -------------- | ------------------------------------------------------------ | ------------------------ | -------- |
| features | Input           | Predicted logits, corresponding to $x$ in the formula. It is a matrix of batch_size * num_classes. | FLOAT16, FLOAT, BFLOAT16 | ND       |
| labels   | Input           | Class labels, corresponding to $y$ in the formula. It is a vector of length batch_size, with the value range [0, num_classes). | INT32, INT64             | ND       |
| loss     | Output           | Loss value of each sample, a vector of length batch_size. The data type is consistent with features. | FLOAT16, FLOAT, BFLOAT16 | ND       |
| backprop | Output           | Gradient of backpropagation, with the shape consistent with features. The data type is consistent with features. | FLOAT16, FLOAT, BFLOAT16 | ND       |

## Constraints

- The data type of features is consistent with loss and backprop.
- The number of dimensions of labels must equal the number of dimensions of features minus 1, and the sizes of all dimensions except the last one must be the same as the corresponding dimensions of features.
- The last dimension (num_classes) of features must be greater than 0, that is, it contains at least one class.
- The values of labels must fall in the range [0, num_classes).
- The operator is compatible with the TensorFlow SparseSoftmaxCrossEntropyWithLogits operator.

## Invocation Description

| Invocation method | Sample code                                                     | Description                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Graph-mode invocation | [test_geir_sparse_softmax_cross_entropy_with_logits](./examples/test_geir_sparse_softmax_cross_entropy_with_logits.cpp) | Calls the SparseSoftmaxCrossEntropyWithLogits operator by building a graph through the [operator IR](./op_graph/sparse_softmax_cross_entropy_with_logits_proto.h). |
