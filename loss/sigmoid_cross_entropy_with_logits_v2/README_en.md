# SigmoidCrossEntropyWithLogitsV2

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

- Description: Computes the BCELoss between the input logits and the label target.

- Formula:

  - Single-label scenario:

    $$
    \ell(self, target) = L = \{l_{1},..., l_{n}\}^{T}
    $$

    $$
    \ell_{n} = -weight_{n}[target_{n} \cdot log(\sigma(self_{n})) + (1 - target_{n}) \cdot log(1 - \sigma(self_{n}))]
    $$

    $$
    \ell(self, target) =
    \begin{cases}
    L, & if\ reduction = none\\
    mean(L), & if\ reduction = mean\\
    sum(L), & if\ reduction = sum\\
    \end{cases}
    $$

  - Multi-label scenario:

    $$
    \ell_c(self, target) = L_c = \{l_{1,c},..., l_{n,c}\}^{T}
    $$

    $$
    \ell_{n,c} = -weight_{n,c}[pos\_weight_{n,c} \cdot target_{n,c} \cdot log(\sigma(self_{n,c})) + (1 - target_{n,c}) \cdot log(1 - \sigma(self_{n,c}))]
    $$

- **Parameters:**

   <table style="undefined;table-layout: fixed; width: 1290px"><colgroup>
    <col style="width: 183px">
    <col style="width: 120px">
    <col style="width: 263px">
    <col style="width: 222px">
    <col style="width: 152px">
    <col style="width: 101px">
    <col style="width: 104px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage instructions</th>
        <th>Data type</th>
        <th>Data format</th>
        <th>Dimensions (shape)</th>
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>The output of the connection layer.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target</td>
        <td>Input</td>
        <td>The label value.</td>
        <td>-</td>
        <td>Same as "self"</td>
        <td>ND</td>
        <td>Same as "self"</td>
        <td>√</td>
      </tr>
      <tr>
      <td>weightOptional</td>
      <td>Input</td>
      <td>The binary cross entropy weight.</td>
      <td>The shape must be broadcastable to "target"</td>
      <td>Same as "self"</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
      </tr>
      <tr>
        <td>posWeightOptional</td>
        <td>Input</td>
        <td>The positive class weight of each class.</td>
        <td>The shape must be broadcastable to "target"</td>
        <td>Same as "self"</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>Input</td>
        <td>The computation method of the output result.</td>
        <td>The direct-call operator currently supports only the none value: <ul><li>none means no operation is performed</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>The output error.</td>
        <td>If reduction = None, the shape is the same as self; in other cases, the shape is [1]</td>
        <td>Same as "target"</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
    </tbody></table>

  - <term>Atlas inference products</term>, <term>Atlas training products</term>: The data type does not support BFLOAT16.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [aclnnBinaryCrossEntropyWithLogits](./docs/aclnnBinaryCrossEntropyWithLogits_en.md) | Calls the SigmoidCrossEntropyWithLogitsV2 operator through the [aclnnBinaryCrossEntropyWithLogits](./docs/aclnnBinaryCrossEntropyWithLogits_en.md) interface. |
