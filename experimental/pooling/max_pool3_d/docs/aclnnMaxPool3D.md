# aclnnMaxPool3D

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

- 算子功能：对五维输入张量的深度、高度和宽度维执行三维最大池化。
- 计算公式：

  $$
  y(n,c,o_d,o_h,o_w)=\max_{(i_d,i_h,i_w)\in R(o_d,o_h,o_w)}x(n,c,i_d,i_h,i_w)
  $$

  其中，$R(o_d,o_h,o_w)$ 是由 `ksize`、`strides`、`padding`、`pads` 和
  `dilation` 确定的有效输入区域。padding 区域不参与最大值计算。

## 函数原型

aclnnMaxPool3D采用两段式接口，必须先调用 `aclnnMaxPool3DGetWorkspaceSize` 获取
workspace大小和算子执行器，再调用 `aclnnMaxPool3D` 执行计算。

- `aclnnStatus aclnnMaxPool3DGetWorkspaceSize(const aclTensor *x, const aclIntArray *ksize, const aclIntArray *strides, const char *padding, const aclIntArray *padsOptional, const aclIntArray *dilationOptional, int64_t ceilMode, const char *dataFormatOptional, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnMaxPool3D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMaxPool3DGetWorkspaceSize

- **参数说明：**

  - x（aclTensor*，计算输入）：五维输入张量。数据类型支持FLOAT16、FLOAT、
    BFLOAT16，逻辑布局由 `dataFormatOptional` 指定，数据格式支持ND和NDC1HWC0。
    支持非连续Tensor，接口内部会将输入连续化。
  - ksize（aclIntArray*，计算输入）：池化窗口大小，支持1、3或5个元素。
    - 1个元素时，该值同时用于D、H、W维。
    - 3个元素时，顺序为D、H、W。
    - 5个元素时，NCDHW布局顺序为N、C、D、H、W，NDHWC布局顺序为
      N、D、H、W、C；N维和C维必须为1。
    - D、H、W维的值必须大于0。
  - strides（aclIntArray*，计算输入）：池化窗口步长，元素数量和顺序与
    `ksize` 相同。N维和C维必须为1，D、H、W维的值必须大于0。
  - padding（char*，计算输入）：padding模式，支持以下取值：
    - `VALID`：不补齐输入，忽略 `padsOptional` 和 `ceilMode`。
    - `SAME`：根据输出尺寸自动计算两侧padding，忽略 `padsOptional` 和
      `ceilMode`。
    - `CALCULATED`：使用 `padsOptional` 指定的padding，并由 `ceilMode`
      控制输出尺寸的取整方式。
  - padsOptional（aclIntArray*，可选计算输入）：显式padding，必须包含6个
    非负整数，顺序为front、back、top、bottom、left、right。传入空指针时使用
    `[0, 0, 0, 0, 0, 0]`。仅在 `CALCULATED` 模式下生效。
  - dilationOptional（aclIntArray*，可选计算输入）：池化窗口膨胀率，支持1、
    3或5个元素，空间维的解释方式与 `ksize` 相同，且必须大于0。传入空指针时
    使用 `[1, 1, 1, 1, 1]`。
  - ceilMode（int64_t，计算输入）：`CALCULATED` 模式下的输出尺寸取整方式。
    0表示向下取整，非0表示向上取整；其他padding模式不使用该参数。
  - dataFormatOptional（char*，可选计算输入）：输入输出的五维逻辑布局，支持
    `NCDHW` 和 `NDHWC`。传入空指针时使用 `NDHWC`。
  - out（aclTensor*，计算输出）：最大池化结果。数据类型必须与 `x` 相同，支持
    FLOAT16、FLOAT、BFLOAT16；逻辑布局必须与 `dataFormatOptional` 一致，数据
    格式支持ND和NDC1HWC0。
  - workspaceSize（uint64_t*，出参）：返回Device侧需要申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回包含算子计算流程的执行器。

- **返回值：**

  `aclnnStatus`：返回状态码。

```text
第一段接口完成基础入参校验，以下场景会返回错误：
161001（ACLNN_ERR_PARAM_NULLPTR）：x、ksize、strides、padding、out、workspaceSize或executor为空指针。
161002（ACLNN_ERR_PARAM_INVALID）：x的数据类型不受支持，或x与out的数据类型不同。

shape、数据格式、padding模式及各数组属性不满足上述约束时，后续算子编排或tiling校验失败。
```

## aclnnMaxPool3D

- **参数说明：**

  - workspace（void*，计算输入）：Device侧workspace地址。当
    `workspaceSize` 为0时允许传入空指针。
  - workspaceSize（uint64_t，计算输入）：第一段接口返回的workspace大小。
  - executor（aclOpExecutor*，计算输入）：第一段接口返回的算子执行器。
  - stream（aclrtStream，计算输入）：执行计算的Stream。

- **返回值：**

  `aclnnStatus`：返回状态码。

## 约束说明

- 输入和输出必须是五维逻辑张量，不支持广播。
- 输出的逻辑shape必须与 `ksize`、`strides`、`padding`、`padsOptional`、
  `dilationOptional`、`ceilMode` 和 `dataFormatOptional` 共同推导出的shape一致；
  NDC1HWC0格式的物理存储shape允许包含分形格式所需的补齐容量。
- 空输入或空输出直接返回，第一段接口返回的workspace大小为0。
- 算子采用确定性最大值归约，相同输入和属性产生相同输出。

## 调用示例

NDHWC布局的完整调用样例请参见
[test_aclnn_max_pool3_d.cpp](../examples/test_aclnn_max_pool3_d.cpp)，NCDHW布局样例请参见
[test_aclnn_max_pool3_d_ncdhw.cpp](../examples/test_aclnn_max_pool3_d_ncdhw.cpp)。两组样例均包含
ACL初始化、Tensor和属性创建、两段式接口调用、结果回拷与数值校验。

核心调用流程如下：

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclError ret = aclnnMaxPool3DGetWorkspaceSize(
    x, ksize, strides, "VALID", pads, dilation, 0, "NDHWC", out,
    &workspaceSize, &executor);
if (ret != ACL_SUCCESS) {
    return ret;
}

void* workspace = nullptr;
if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
}

ret = aclnnMaxPool3D(workspace, workspaceSize, executor, stream);
if (ret != ACL_SUCCESS) {
    return ret;
}
ret = aclrtSynchronizeStream(stream);
if (workspace != nullptr) {
    aclrtFree(workspace);
}
return ret;
```
