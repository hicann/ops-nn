# SparseApplyFtrl

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                     |     √    |
| <term>Atlas A3系列产品</term>   |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    √     |
| <term>Atlas训练系列产品</term>                               |    √     |

> 说明：
> - Ascend 950PR/950DT 由本仓 Ascend C 实现提供（OpDef 注册 ascend950/ascend350，构建 ascend950/ascend350）；Atlas A3/A2、Atlas推理/训练由 canndev TBE 实现承接；Atlas 200I/500 A2 无对应二进制配置，不支持。
> - 当前各实现的数据类型均为 FLOAT（var/accum/linear/grad/lr/l1/l2/lr_power）与 INT32/INT64（indices），数据格式为 ND。

## 功能说明

- 算子功能：SparseApplyFtrl是FTRL-Proximal（Follow The Regularized Leader）优化器的稀疏更新算子。根据`indices`选择`var`、`accum`、`linear`的对应行，使用梯度`grad`进行FTRL-Proximal更新。该算子广泛用于推荐系统、CTR预估等场景中的稀疏特征权重更新。

- 计算公式：

$$
\text{accum\_new} = \text{accum} + \text{grad}^2
$$

$$
\sigma = \frac{\text{accum}^{-\text{lr\_power}} - \text{accum\_new}^{-\text{lr\_power}}}{\text{lr}}
$$

$$
\text{linear\_new} = \text{linear} + \text{grad} + \sigma \cdot \text{var}
$$

$$
x = \text{clamp}(\text{linear\_new}, -l1, l1) - \text{linear\_new}
$$

$$
y = \frac{\text{accum\_new}^{-\text{lr\_power}}}{\text{lr}} + 2 \cdot l2
$$

$$
\text{var\_new} = \frac{x}{y}
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>待更新的变量tensor，≥2维</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入</td>
      <td>梯度平方累加器，shape与var相同</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>输入</td>
      <td>线性累加器，shape与var相同</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>梯度tensor，shape[1:] == var.shape[1:]</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引向量，1-D，shape[0] == grad.shape[0]</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>输入</td>
      <td>L1正则化系数，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>输入</td>
      <td>L2正则化系数，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr_power</td>
      <td>输入</td>
      <td>学习率幂次，标量tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的变量，shape/dtype同输入var</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输出</td>
      <td>更新后的累加器，shape/dtype同输入accum</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>输出</td>
      <td>更新后的线性累加器，shape/dtype同输入linear</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>可选属性，默认false。是否使用锁保护更新操作；当前实现未使用该属性，不改变更新行为。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

1. var/accum/linear必须具有相同的shape和dtype（FLOAT）。
2. grad必须与var维数相同；grad.shape[0]必须等于indices.shape[0]，grad.shape[1:]必须等于var.shape[1:]。
3. indices必须为1-D tensor，dtype为INT32或INT64。
4. lr/l1/l2/lr\_power为标量tensor，形态为0-D或shape为[1]的1-D tensor；动态shape（-1/-2）在InferShape中按动态处理，`(1,1)`等更高维形态会被拒绝。
5. **本仓Ascend C实现要求var/accum/linear至少为2维，当前未显式校验rank上限（受框架/接口限制）**；Atlas A3/A2等由canndev TBE承接的实现对rank下限较宽（允许1-D var），跨产品使用时请按对应实现版本约束准备输入。
6. **lr为学习率，调用方必须保证lr > 0**；l1、l2应大于等于0，lr\_power应小于等于0。Host/Tiling不读取设备侧标量值，因此不做运行时值域校验，非法值会导致结果未定义。
7. **indices必须唯一**（不可有重复值）。TensorFlow和Ascend官方均声明：indices重复时行为未定义；当前实现未做运行时重复检查。
8. **indices取值范围必须满足0 <= indices[i] < var.shape[0]**。kernel不做运行时越界检查，越界将导致内存越界读写。
9. **输入tensor必须为连续内存布局**。kernel使用扁平寻址假设连续内存；`AutoContiguous()`只在aclnn通路生效，geir/tf通路需调用方保证输入连续。
10. **输出var/accum/linear为独立输出缓冲，不与输入原地共享存储**。调用方应使用输出tensor获取更新结果。
11. **数据格式仅支持ND**。本版本binary_config仅提供FLOAT+INT32/INT64、ND的组合；历史版本中的NCHW/NHWC等位置化组合不在本版本支持范围内。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|:---|:---|:---|
| GE图模式 | [test_geir_sparse_apply_ftrl.cpp](./examples/test_geir_sparse_apply_ftrl.cpp) | 通过[SparseApplyFtrl算子IR](./op_graph/sparse_apply_ftrl_proto.h)构图方式调用SparseApplyFtrl算子。 |

> TensorFlow通路当前仅注册`SparseApplyFtrl`；`ResourceSparseApplyFtrl`未在本仓插件中注册，由canndev实现承接。示例的编译与运行请参考CANN样例编译运行说明。
