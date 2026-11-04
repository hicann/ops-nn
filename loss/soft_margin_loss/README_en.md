# SoftMarginLoss

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: computes the binary classification logistic loss function of the input self and the target target.
- Formula:

When `reduction` is 'none':

$$
\text{out}(self,target)=L=log(1+\exp(-target[i]*self[i]))
$$

If `reduction` is `mean` or `sum`,

$$
\text{out}(self,target)=\begin{cases}
mean(L), & \text{if reduction} = \text{'mean'}\\
sum(L), & \text{if reduction} = \text{'sum'}
\end{cases}
$$

- Here:
- self: the input tensor
- target: the ground-truth label

## Parameters

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">Parameter name</th>
    <th style="white-space: nowrap">Input/Output/Attribute</th>
    <th style="white-space: nowrap">Description</th>
    <th style="white-space: nowrap">Data type</th>
    <th style="white-space: nowrap">Data format</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>self</td>
    <td>Input</td>
    <td>'self' in the formula, denoting the input tensor.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>target</td>
    <td>Input</td>
    <td>'target' in the formula, denoting the ground-truth label.</td>
    <td>BFLOAT16, FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>loss</td>
    <td>Output</td>
    <td>Loss value output computed by soft_margin_loss from self and target</td>
    <td>FLOAT16, BFLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reduction</td>
    <td>Attribute</td>
    <td>Reduction parameter specifying mean/sum/no reduction of the element-wise loss values, supporting 'none', 'mean', and 'sum'</td>
    <td>STRING</td>
    <td>-</td>
    </tr>

</tbody></table>

## Constraints

- Numerical boundaries (the operator computes in float32 precision, which also applies to FLOAT16/BFLOAT16 inputs):
  - Overflow: when an element satisfies $-target \times self > \ln(FLT\_MAX) \approx 88.72$, $\exp(-target \times self)$ overflows to +inf, and the output is $log(1+inf)=inf$; when `reduction` is 'mean' or 'sum', any element being inf makes the scalar output inf. A typical scenario is a severely wrong prediction direction (for example, target=1 and self takes a large negative value).
  - Underflow: when an element satisfies $target \times self > \ln(2^{24}) \approx 16.64$, $\exp(-target \times self) < 2^{-24}$, $1+\exp(-target \times self)$ rounds to 1, and the output is 0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_soft_margin_loss.cpp](examples/test_aclnn_soft_margin_loss.cpp) | Calls the SoftMarginLoss operator through the [aclnnSoftMarginLoss](docs/aclnnSoftMarginLoss_en.md) interface. |
