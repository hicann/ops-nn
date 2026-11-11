# FusedMulApplyKerasMomentum

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

## 功能说明

- 算子功能：Keras风格Momentum SGD优化器的融合算子。先融合一次梯度乘法得到梯度，再按Momentum SGD公式以inplace方式更新权重var与动量累加器accum，支持标准Momentum与Nesterov Momentum两种模式。对标TensorFlow中`tf.raw_ops.ResourceApplyKerasMomentum`接口的计算语义。
- 计算公式：

  梯度与动量累加器更新（两种模式一致）：

  $$
  grad = x1 \times x2
  $$

  $$
  accum_{new} = accum \times momentum - grad \times lr
  $$

  - 标准模式（use_nesterov = false）：

  $$
  var_{new} = var + accum_{new}
  $$

  - Nesterov模式（use_nesterov = true）：

  $$
  var_{new} = var + accum_{new} \times momentum - grad \times lr
  $$

  其中`x1`、`x2`为梯度相关输入，`lr`为学习率，`momentum`为动量系数，`accum`为动量累加器，`var`为待更新的权重参数。
- 精度说明：FLOAT16输入在Kernel内部先升为FLOAT32，全程FLOAT32高精度累加，末尾还原为FLOAT16输出。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 150px">
<col style="width: 180px">
<col style="width: 400px">
<col style="width: 180px">
<col style="width: 120px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>var</td>
    <td>输入</td>
    <td>待更新的权重参数，对应公式中的var。与输出端口var同名（inplace别名），Kernel写输出即写回该输入存储。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入</td>
    <td>动量累加器，对应公式中的accum。与输出端口accum同名（inplace别名），Kernel写输出即写回该输入存储；shape与dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape为[1]的标量Tensor，dtype与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1</td>
    <td>输入</td>
    <td>梯度相关输入，对应公式中的x1。shape与dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数，对应公式中的momentum。shape为[1]的标量Tensor，dtype与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>梯度相关输入，对应公式中的x2。shape为[1]的标量Tensor，dtype与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的权重参数，对应公式中的var_new。与输入var共享Device内存（inplace别名），shape/dtype与输入var相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输出</td>
    <td>更新后的动量累加器，对应公式中的accum_new。与输入accum共享Device内存（inplace别名），shape/dtype与输入accum相同。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>可选属性</td>
    <td><ul><li>是否对更新操作加锁保护。</li><li>默认值为false。</li></ul></td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>可选属性</td>
    <td><ul><li>是否使用Nesterov动量。true为Nesterov模式，false为标准模式。</li><li>默认值为false。</li></ul></td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- var、accum、x1必须具有相同的数据类型和数据格式，且shape一致。
- var、accum的输出端口与输入端口同名（inplace别名），shape分别与对应输入相同，调用方必须把这两个输入视为可写。
- lr、momentum、x2为shape为[1]的标量Tensor，数据类型与var一致。
- var、accum、x1的维度数不超过8。
- 不支持非连续Tensor输入。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_fused_mul_apply_keras_momentum.cpp">调用样例</a></td>
    <td>本算子生产路径为GE IR图模式，通过<a href="./op_graph/fused_mul_apply_keras_momentum_proto.h">算子IR</a>构图方式调用FusedMulApplyKerasMomentum算子。TensorFlow训练经tf_plugin与融合PASS调度生成本算子。</td>
  </tr>
</tbody></table>
