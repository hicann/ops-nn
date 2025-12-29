# FastGelu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT9</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √      |

## 功能说明

算子功能：返回fastgelu激活函数输出张量。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFastGeluGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFastGelu”接口执行计算。

## aclnnFastGeluGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
    <col style="width: 173px">
    <col style="width: 120px">
    <col style="width: 222px">
    <col style="width: 338px">
    <col style="width: 156px">
    <col style="width: 104px">
    <col style="width: 162px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>self</td>
      <td>输入</td>
      <td>输入张量。</td>
      <td>数据类型支持FLOAT16、FLOAT32、BFLOAT16</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>(N)</td>
      <td>√</td>
      </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>数据类型支持FLOAT16、FLOAT32、BFLOAT16</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
      <td>(N)</td>
      <td>√</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    </tbody></table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## aclnnFastGelu

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
      </colgroup>
      <thead>
      <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFastGeluGetWorkspaceSize获取。</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
      </tr>
      </tbody>
    </table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无