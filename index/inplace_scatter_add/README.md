# InplaceScatterAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                          |    √   |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×   |

## 功能说明

- 算子功能：根据给定的indices，将updates中的值加到输入张量var的第一维度上。

- 计算公式：

  $$
  var[indices]  +=  updates
  $$

- 就地语义：本算子为就地算子，updates中的值直接累加到var的存储上，var在调用后将被原地改写；输出var与输入var的存储就地绑定，aclnn接口内部将结果拷贝至out交付。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 40px">
  <col style="width: 60px">
  <col style="width: 260px">
  <col style="width: 60px">
  <col style="width: 40px">
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
      <td>表示需要被更新的张量，公式中的var；计算结果就地写入var的存储，var会被原地改写（见约束说明）。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>表示对var的第一维的索引，shape只支持1维。期望取值范围为[0, var.dim[0])；取值范围不做校验，负值索引被kernel守卫静默跳过，越界索引行为未定义。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>表示要被加到var上的值。shape约束：第0维等于indices的元素个数，其余维度与var除第0维外的对应维度一致。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 就地语义：输出var与输入var共用存储（就地绑定），self在调用后被原地改写。
- updates形状约束：updates的第0维须等于indices的元素个数，其余维度与var除第0维外的对应维度一致；违反该约束的输入在tiling阶段被拒绝。
- indices取值范围：期望取值范围为[0, var.dim[0])；取值范围不做校验，负值索引被kernel守卫静默跳过（不参与累加），越界索引行为未定义。
- 空Tensor支持：updates为空Tensor时输出等于var（不进行累加）；var为空Tensor时输出为空Tensor。
- 张量规模：计算涉及的元素规模不超过uint32表示域（2^32-1）时走32位快速路径，超出时自动路由至64位变体，功能结果一致；实际容量上限取决于数据类型与Device内存。
- kernel直调边界：kernel直调（不经aclnn第一段校验）时的边界错误码不作承诺，输入域合法性由调用方或aclnn接口第一段校验保证。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------- | ---------------- | -------------------------------------------- |
| aclnn接口  | [test_aclnn_index_add.cpp](examples/test_aclnn_index_add.cpp) | 通过[aclnnIndexAdd](docs/aclnnIndexAdd.md)接口方式调用InplaceScatterAdd算子。实际分派行为：V1的InplaceScatterAdd快速路径经DAV_2201架构门禁仅在ascend910b/910_93系列放行；Ascend 950PR&950DT（arch35）上非确定性场景的float32/int32输入实际经InplaceIndexAdd执行；bool输出的字节语义不在本算子支持范围内（本算子kernel数据类型仅FLOAT/INT32）。 |
| aclnn接口  | [test_aclnn_index_add_v2.cpp](examples/test_aclnn_index_add_v2.cpp) | 通过[aclnnIndexAddV2](docs/aclnnIndexAddV2.md)接口方式调用InplaceScatterAdd算子，当前经PTA/确定性场景使用。 |
