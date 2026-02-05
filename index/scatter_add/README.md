# ScatterAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |


## 功能说明

- 算子功能：将src tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入self tensor中，若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加。
  对于一个3D tensor， self会按照如下的规则进行更新：

  ```
  self[index[i][j][k]][j][k] += src[i][j][k] # 如果 dim == 0
  self[i][index[i][j][k]][k] += src[i][j][k] # 如果 dim == 1
  self[i][j][index[i][j][k]] += src[i][j][k] # 如果 dim == 2
  ```

  在计算时需要满足以下要求：
  - self, index和src的维度数量必须相同
  - 对于每一个维度d, 有index.size(d) <= src.size(d)
  - 对于每一个维度d, 如果有d != dim, 有index.size(d) <= self.size(d)
  - dim取值范围为[-self.dim(), self.dim() - 1]
- 用例：
  
  输入tensor $self = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [7&8&9] \end{bmatrix}$,
  索引tensor $index = \begin{bmatrix} [0&2&1] \\ [0&0&1] \end{bmatrix}$, dim = 1,
  源tensor $src = \begin{bmatrix} [10&11&12] \\ [13&14&15] \end{bmatrix}$，
  输出tensor $output = \begin{bmatrix} [11&14&14] \\ [31&20&6] \\ [7&8&9] \end{bmatrix}$
  
  dim = 1 表示scatter_add根据$index$在tensor的列上进行累加。
  
  $output[0][0] = self[0][0] + src[0][0]$ = 1 + 10,
  
  $output[0][1] = self[0][1] + src[0][2]$ = 2 + 12,
  
  $output[0][2] = self[0][2] + src[0][1]$ = 3 + 11,
  
  $output[1][0] = self[1][0] + src[1][0] + src[1][1]$ = 4 + 13 + 14,
  
  $output[1][1] = self[1][1] + src[1][2]$ = 5 + 15,
  
  $output[1][2] = self[1][2]$ = 6,
  
  $output[2][0] = self[2][0]$ = 7,
  
  $output[2][1] = self[2][1]$ = 8,
  
  $output[2][2] = self[2][2]$ = 9.
  
  其中，$self$、$index$、$src$的维度数量均为2，$index$每个维度大小{2，3}都不大于$src$的对应维度大小{2，3}，在dim != 1的维度上（dim = 0），$index$的维度大小{2}不大于$self$的对应维度大小{3}，$index$中的最大值{2}，小于$self$在dim = 1维度的大小{3}。

  ## 参数说明

    - self（aclTensor*，计算输入）：公式中的输入`self`，Device侧的aclTensor。scatter的目标张量，shape支持0-8维，且维度数量需要与index和src相同。数据类型与src的数据类型一致。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128。
    - dim（int64_t, 计算输入）：计算公式中的输入`dim`，数据类型为INT64。
  
    - index（aclTensor*，计算输入）：公式中的输入`index`，Device侧的aclTensor。数据类型支持INT32、INT64。index维度数量需要与src相同。支持[非连续的Tensor](../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../docs/zh/context/数据格式.md)支持ND。
      - <term>Ascend 950PR/Ascend 950DT</term>：当dim轴上index值存在重复时，结果将是不确定的。若开启了确定性计算，可保证结果的确定性。
    - src（aclTensor*，计算输入）：公式中的输入`src`，Device侧的aclTensor。源张量，src维度数量需要与index相同。数据类型与self的数据类型一致。[数据格式](../../docs/zh/context/数据格式.md)支持ND。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128。
    - out（aclTensor*，计算输出）：公式中的`output`，Device侧的aclTensor。shape需要与self一致。数据类型与self的数据类型一致。[数据格式](../../docs/zh/context/数据格式.md)支持ND。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_scatter_add](examples/test_aclnn_scatter_add.cpp) | 通过[aclnnScatterAdd](docs/aclnnScatterAdd.md)接口方式调用ScatterAdd算子。 |