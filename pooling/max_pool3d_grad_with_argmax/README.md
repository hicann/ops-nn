# MaxPool3DGradWithArgmax

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
 

## 功能说明

- 算子功能：正向最大池化的反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 500px">
  <col style="width: 300px">
  <col style="width: 200px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>待进行MaxPool3DGradWithArgmax计算的入参，表示当前节点的梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行MaxPool3DGradWithArgmax计算的入参，表示正向的输入Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>表示正向输入中最大元素的索引位置。数据格式需要与`self`一致，shape需要与`gradOutput`一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>表示最大池化的窗口大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>表示池化操作的步长。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>表示在输入的D、H、W方向上padding补0的层数。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入</td>
      <td>表示控制窗口中元素的步幅。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>输入</td>
      <td>表示正向平均池化过程中推导的输出的shape是否向上取整。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>待进行MaxPool3DGradWithArgmax计算的出参。shape、数据格式需要与`self`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明
- **未支持类型说明**
  * 不支持空进空出。
- **边界值场景说明**
  * 当输入是inf时，输出为inf。
  * 当输入是nan时，输出为nan。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_max_pool3d_with_argmax_backward.cpp](examples/test_aclnn_max_pool3d_with_argmax_backward.cpp) | 通过[aclnnMaxPool3dWithArgmaxBackward](docs/aclnnMaxPool3dWithArgmaxBackward.md)接口方式调用MaxPool3DGradWithArgmax算子。 |

