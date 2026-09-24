# LayerNormONNXFusionPass

## 融合模式

该融合规则将ONNX导入的ReduceMean、Sub、Pow（或Square）、Add、Sqrt、Div等小算子组合识别并融合为LayerNorm算子。带缩放和偏移的场景同时融合末尾的Mul和Add。

- 场景一：带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_1.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_7.png)

- 场景二：不带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_2.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_8.png)

- 场景三：平方计算前带Cast，且带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_3.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_7.png)

- 场景四：平方计算前带Cast，不带gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_4.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_8.png)

- 场景五：开方前依次经过Max和Min，不带Cast、gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_5.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_8.png)

- 场景六：开方前依次经过Min和Max，不带Cast、gamma和beta。

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_6.png)

  **融合为**

  ![](../../../docs/zh/figures/layer_norm_onnx_fusion_pass_8.png)

上述各场景中，Pow的指数输入exp为2，也支持替换为不带exp输入的Square。场景三、四的除法节点仅支持Div；场景一、二、五、六的除法节点支持Div或RealDiv。Add\_0的两个输入可以交换，带gamma和beta时，Mul\_0、Add\_1的两个输入也可以分别交换。

融合后，x作为LayerNorm的x输入；带gamma和beta时复用原输入，不带时创建全1的gamma和全0的beta常量，shape为归一化轴对应的后缀shape。epsilon常量转为LayerNorm的epsilon属性，begin\_norm\_axis取归一化起始轴，begin\_params\_axis取beta维数的负值。只使用LayerNorm的y输出替换原子图出口，不额外连接mean和variance输出。

## 使用约束

- x的维数必须大于等于1。
- 两个ReduceMean按相同的归一化轴进行计算，keep\_dims为true。axes使用INT32或INT64常量，归一化轴必须为连续递增、以最后一维结束的非空轴序列。当前实现对ReduceMean\_0的axes进行转换和检查；涉及第0维时应使用对应的负轴表示，不使用值为0的轴。
- Sub的第一个输入必须是x，第二个输入必须是ReduceMean\_0的结果，不能调换位置。平方支路和除法分子必须复用同一个Sub结果。
- 场景三、四带Cast时，除法节点仅支持Div。Cast位于Sub与Pow（或Square）之间，目标数据类型为FLOAT32；Div的分子仍直接使用Sub结果。其他场景的除法节点支持Div或RealDiv。
- epsilon必须是FLOAT32或FLOAT16常量，shape为标量或[1]，取值必须接近0，且不大于0.1。Pow的指数输入exp也必须是这两种数据类型的标量或[1]常量，值为2。
- 带Max和Min时，Max的第二个输入为0，Min的第二个输入为inf（正无穷），均为FLOAT32或FLOAT16的标量或[1]常量。两个算子的常量输入不支持交换。
- 带gamma和beta时，在对应Mul或Add的两个输入均为静态shape的情况下，gamma或beta必须为常量，维数等于归一化轴的数量，且各维长度与x的对应归一化维度一致。对应节点存在动态shape输入时，当前实现跳过该参数的常量和shape检查。
- 不带gamma和beta时，不支持x为动态shape；归一化后缀shape的元素总数必须大于0，除法节点输出的数据类型仅支持FLOAT32、FLOAT16，用于创建gamma和beta常量。
- 被融合子图的内部节点不能被子图外的其他分支复用；子图出口可以有多个消费者。不带gamma和beta的模式以除法节点为出口，带gamma和beta的模式以Add\_1为出口。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
