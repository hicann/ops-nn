# BNLL

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>     |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Activation function BNLL (Binomial Normal Log Likelihood), that is, the numerically stable version of softplus.
- Formula:

$$
BNLL(x) = \ln(1 + e^x)
$$

Equivalent piecewise implementation (to ensure numerical stability):

$$
y_i = \begin{cases} x_i + \ln(1 + e^{-x_i}) & \text{if } x_i > 0 \\ \ln(1 + e^{x_i}) & \text{otherwise} \end{cases}
$$

## Parameters

<table style="table-layout: auto; width: 100%">
<thead>
    <tr>
    <th style="white-space: nowrap">Parameter name</th>
    <th style="white-space: nowrap">Input/Output</th>
    <th style="white-space: nowrap">Description</th>
    <th style="white-space: nowrap">Data type</th>
    <th style="white-space: nowrap">Data format</th>
    </tr>
</thead>
<tbody>
    <tr>
    <td>x</td>
    <td>Input</td>
    <td>'x' in the formula, the input tensor of the BNLL activation function. Empty tensors are supported, and 0 to 8 dimensions are supported.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>Output</td>
    <td>Output tensor of x computed by BNLL, with the shape and dtype the same as the input x.</td>
    <td>FLOAT16, FLOAT32</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode  | [test_geir_bnll.cpp](examples/test_geir_bnll.cpp) | Calls the BNLL operator through the GE IR API. |


## Golden Host Regression

The NumPy Kernel golden uses `max(x, 0) + log1p(exp(-abs(x)))`. The exponent input is always non-positive, which avoids overflow when `np.where` evaluates both the selected and unselected branches. The original float64 intermediate precision and the input dtype/shape are kept, and positive infinity, negative infinity, and NaN are propagated according to the formula.

Run `python3 activation/bnll/tests/assets/test_bnll_golden_branch_overflow.py` to check the actually registered golden. It verifies large values, normal values, scalars, matrices, and empty inputs under the RuntimeWarning-as-error and NumPy overflow/invalid strict modes, using the CPU Torch `logaddexp` as an independent reference. Only the Host golden is verified; it does not represent TTK device execution, CANN build, or NPU operator precision or performance verification.
