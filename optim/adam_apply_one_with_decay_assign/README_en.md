# AdamApplyOneWithDecayAssign

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: Performs the single-step computation and update of the Adam optimization algorithm for one parameter (such as a weight) in the model.

- Formula:

$output_0 = input_0^2 \times mul_3\_x + input_1 \times mul_2\_x$

$output_1 = input_2 \times mul_0\_x + input_0 \times mul_1\_x$

$output_2 = input_3 - (\frac{output_1}{\sqrt(output_0) + add_2\_y}) \times input_4$

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
<col style="width: 100px">
<col style="width: 150px">
<col style="width: 280px">
<col style="width: 330px">
<col style="width: 120px">
</colgroup>
<thead>
    <tr>
    <th>Parameter name</th>
    <th>Input/Output/Attribute</th>
    <th>Description</th>
    <th>Data type</th>
    <th>Data format</th>
    </tr></thead>
<tbody>
    <tr>
    <td>input0</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, input0 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input1</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, input1 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input2</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, input2 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input3</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, input3 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>input4</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, input4 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul0_x</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, mul0_x in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul1_x</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, mul1_x in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul2_x</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, mul2_x in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>mul3_x</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, mul3_x in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>add2_y</td>
    <td>Input</td>
    <td>Input parameter for the adam_apply_one_with_decay_assign computation, add2_y in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output0</td>
    <td>Output</td>
    <td>Output parameter of the adam_apply_one_with_decay_assign computation, output0 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output1</td>
    <td>Output</td>
    <td>Output parameter of the adam_apply_one_with_decay_assign computation, output1 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>output2</td>
    <td>Output</td>
    <td>Output parameter of the adam_apply_one_with_decay_assign computation, output2 in the formula.</td>
    <td>BFLOAT16, FLOAT16, FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_adam_apply_one_with_decay_assign](./examples/test_geir_adam_apply_one_with_decay_assign.cpp)   | Calls the AdamApplyOne operator by building a graph with the [operator IR](./op_graph/adam_apply_one_with_decay_assign_proto.h). |
