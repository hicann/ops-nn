# SigmoidCrossEntropyWithLogits

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    √     |

## Function

- Operator function: Computes the Sigmoid Cross Entropy loss between input logits and label target.

- Formula:

  $$
  \text{loss} = \max(\text{predict}, 0) - \text{predict} \times \text{target} + \log(1 + \exp(-|\text{predict}|))
  $$

  Here:

  - predict is the input logits
  - target is the label value
  - loss is the computed loss value

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| ----- | ----- |----- |----- |----- |
| predict | Input | Prediction logits. | FLOAT16, FLOAT, BFLOAT16 | ND |
| target | Input | Label value. | FLOAT16, FLOAT, BFLOAT16 | ND |
| loss | Output | Loss value. The shape is the same as input predict. | FLOAT16, FLOAT, BFLOAT16 | ND |

## Constraints

- predict and target must have the same data type and shape.
- The data types FLOAT16, FLOAT, and BFLOAT16 are supported.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation | - | Calls the SigmoidCrossEntropyWithLogits operator by building a graph with the [operator IR](op_graph/sigmoid_cross_entropy_with_logits_proto.h). |
