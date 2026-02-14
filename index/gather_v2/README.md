# GatherV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：从输入Tensor的指定维度dim，按index中的下标序号提取元素，保存到out Tensor中。
- 计算流程：
  例如输入张量 $x=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ 和索引张量 idx=[1, 0]。
  - dim=0的结果：$y=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$

  - dim=1的结果： $y=\begin{bmatrix}2 & 1\\ 5 & 4\\ 8 & 7\end{bmatrix}$

  具体计算过程如下：
  以三维张量为例，shape为(3,2,2)的张量 x =$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$   idx=[1, 0],   x张量dim=0,1,2对应的下标分别是$l, m, n$,  idx是一维（零维的情况：当成是size为1的一维）
  - dim为0：I=index[i];  &nbsp;&nbsp;   y$[i][m][n]$ = x$[I][m][n]$

  - dim为1：J=index[j];  &nbsp;&nbsp;&nbsp;    y$[l][j][n]$ = x$[l][J][n]$

  - dim为2：K=index[k]; &nbsp;  y$[l][m][k]$ = x$[l][m][K]$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
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
        <td>x</td>
        <td>输入</td>
        <td>输入的张量，公式中的x。</td>
        <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>indices</td>
        <td>输入</td>
        <td>输入的下标，用于指定索引位置，公式中的idx。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>axis</td>
        <td>输入</td>
        <td>一个标量，表示操作的轴（维度），公式中的dim。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>输入的结果，公式中的y。</td>
        <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>batch_dims</td>
        <td>可选属性</td>
        <td><ul><li>用于指定批处理的维度。</li><li>默认值为0.0。</ul></td>
        <td>INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>negative_index_support</td>
        <td>可选属性</td>
        <td><ul><li>用于指定是否支持负数索引，值为true表示支持索引为负数，值为false则不支持索引为负数。</li><li>默认值为false。</ul></td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_gather_v2](./examples/test_aclnn_gather_v2.cpp) | 通过[aclnnGatherV2](./docs/aclnnGatherV2.md)接口方式调用GatherV2算子。 |
| aclnn调用 | [test_aclnn_embedding_renorm](./examples/test_aclnn_embedding_renorm.cpp) | 通过[aclnnEmbeddingRenorm](./docs/aclnnEmbeddingRenorm.md)接口方式调用Renorm算子。 |