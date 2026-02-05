# IndexPutImpl

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √   |


## 功能说明

- 接口功能：根据索引 indices 将输入 self 对应坐标的数据与输入 values 进行替换或累加。
- 计算公式：

  - accumulate = False:

    $$
    self[indices] = values
    $$

  - accumulate = True:
    
    $$
    self[indices]  = self[indices]  + values
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
        <th>输入/输出</th>
        <th>描述</th>
        <th>数据类型</th>
        <th>数据格式</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>selfRef</td>
        <td>输入</td>
        <td>公式中的 self。数据类型和values一致。</td>
        <td>FLOAT、FLOAT16、DOUBLE、INT32、INT64、INT16、INT8、UINT8、BOOL、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输入</td>
        <td>公式中的 indices。</td>
        <td>INT32、INT64、BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>values</td>
        <td>输入</td>
        <td>公式中的 values。</td>
        <td>和selfRef一致</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>accumulate</td>
        <td>输入</td>
        <td>累加或更新的操作类型标志位，Host侧的布尔值。<ul><li>accumulate为True时为累加；</li><li>accumulate为False时为更新。</li></ul></td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>unsafe</td>
        <td>输入</td>
        <td>检查索引是否在有效范围内标志位。<ul><li>unsafe为True时，索引越界会直接报错退出执行；</li><li>当unsafe为False时，如果出现了索引越界，就可能出现运行时异常。</li></ul></td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  - <term>Atlas 200I/500 A2 推理产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。
## 约束说明

无
## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_index_put_v2](./examples/test_aclnn_index_put_v2.cpp) | 通过[aclnnAbs](./docs/aclnnIndexPutImpl.md)接口方式调用IndexPutImpl算子。 |
