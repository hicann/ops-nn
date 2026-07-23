# ScatterDiv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |

## 功能说明

- 算子功能：实现兼容tf.compat.v1.scatter_div的功能，将tensor updates中的值按指定的索引tensor indices逐切片除到tensor var的切片上。若有多于一个updates值作用到var的同一个切片，则这些值依次在该切片上连除。属于原地（in-place）更新，输出复用输入var。规则如下：

  $$
  var[indices[i], ...] = var[indices[i], ...] / updates[i, ...]
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>支持空Tensor。表示待被更新的张量，Device侧的aclTensor，原地更新。shape支持1维及以上，数据类型需与updates一致。</td>
      <td>FLOAT16、FLOAT32、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>支持空Tensor。表示待更新的索引张量，Device侧的aclTensor。shape支持1维及以上，数据类型为INT32或INT64。支持索引越界，越界索引（小于0或大于等于var第一维）会被静默跳过，不做更新。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>支持空Tensor。表示需要除到var上的张量，Device侧的aclTensor。shape支持1维及以上，需满足updates.shape = indices.shape + var.shape[1:]，数据类型需与var一致。</td>
      <td>FLOAT16、FLOAT32、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>支持空Tensor。表示更新后的张量，Device侧的aclTensor，与输入var共享内存。shape和数据类型均与输入var一致。</td>
      <td>FLOAT16、FLOAT32、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>HOST侧的bool值，是否对更新加锁，默认false。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：不支持确定性计算。
- var的首维大小不超过INT32_MAX。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_scatter_div](examples/arch35/test_aclnn_scatter_div.cpp) | 通过[aclnnScatterDiv](docs/aclnnScatterDiv.md)接口方式调用ScatterDiv算子。 |
