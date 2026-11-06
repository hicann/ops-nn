# InplaceAddRmsNormFusionPass

## 融合模式

场景一：将AddRmsNorm的y输出地址复用x1输入地址，x输出地址复用x2输入地址，转换为InplaceAddRmsNorm算子。

![](../../../docs/zh/figures/inplace_add_rms_norm_fusion_pass_1.png)

场景二：当识别到AddRmsNorm的y输出仅接了2个输出且第一个输出接Cast算子（将输出y从float16/bfloat16 Cast到float32），将AddRmsNorm和Cast融合为AddRmsNormCast算子。

![](../../../docs/zh/figures/inplace_add_rms_norm_fusion_pass_2.png)

## 使用约束

- 通用约束：仅限推理场景，不支持训练场景。
- 当AddRmsNorm的输入x1或x2的直接前驱节点类型为Data时，不进行原地融合。
- 融合成AddRmsNormCast约束：
  - 仅支持输入x1为float16/bfloat16。
  - 仅支持cast从float16/bfloat16转float32。
  - 融合前后的输出中都不能包含rstd。

- 融合成InplaceAddRmsNorm约束：非Ascend 950PR&950DT系列产品场景下，AddRmsNorm的第二个输出须无后继节点。

## 支持的型号

<!-- npu="910b" id1 -->
Atlas A2系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3系列产品
<!-- end id2 -->

<!-- npu="310p" id3 -->
Atlas推理系列产品
<!-- end id3 -->

<!-- npu="950" id4 -->
Ascend 950PR&950DT系列产品
<!-- end id4 -->
