# UnsortedSegmentMax

## 产品支持情况
| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term> Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：分段计算输入tensor的最大值。
- 计算公式：
$$
output[i]={\max}_{j...}data[j...]
$$
*max* 返回元素 *j*... 中的最大值，其中 *segment_ids*[*j*...]==*i*。
- 用例：

  输入tensor $data = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [4&2&1] \end{bmatrix}$,
  分段索引tensor $segment\_ ids = [0, 1, 1]$,
  $num\_ segments= 2$，

  输出tensor $output = \begin{bmatrix} [1&2&3] \\ [4&5&6] \end{bmatrix}$
  - `segment_ids`不需要排序；分段索引，指示当前分段的值加到哪个段上
  - `segment_ids`不需要覆盖整个有效范围内所有值
  - `segment_ids`如果给定ID i为空，则输出data类型的最小值
  - `segment_ids`如果给定ID i为负值，则改值将被删除并且不会添加到段结果中
  - `num_segments`是scalar，代表输出的分段个数，应该大于等于实际分段的个数

## 参数说明

|类型| 名称 | 描述 |支持的数据类型 | 数据格式 |
| --- | --- | --- | --- | ---|
|输入|data|输入数据|FLOAT32, INT32, INT64, BFLOAT16, FLOAT16, UINT32, UINT64| ND |
|输入|segment_ids|分段索引|INT32, INT64| ND |
|输入|num_segments|分段个数 |INT32, INT64| ND |
|输出|value|输出值信息|FLOAT32, INT32, INT64, BFLOAT16, FLOAT16, UINT32, UINT64 | ND |

## 约束说明
**data**​：
  * 必须与 segment_ids 形状兼容（data.dim_size(0) == segment_ids.NumElements()）。
  * 维度至少 1（rank >=1）。

​**segment_ids**​：
  * 必须是 INT32 或 INT64 类型。
  * 形状必须是 data.shape 的前缀（即 segment_ids.shape 是data.shape[:segment_ids.rank] 的子集）。
  * 值必须 >= 0 且 < num_segments（负值被忽略）。

**num_segments**​：
  * 必须是 INT32 或 INT64 类型，且 > 0。
  * 应等于或大于最大 segment_id + 1。

  ## 调用说明

  | 调用方式   | 样例代码           | 说明                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | 图模式调用 | [test_geir_unsorted_segment_max](examples/test_geir_unsorted_segment_max.cpp) | 通过[算子IR](op_graph/unsorted_segment_max_proto.h)构图方式调用UnsortedSegmentMax算子。 |
