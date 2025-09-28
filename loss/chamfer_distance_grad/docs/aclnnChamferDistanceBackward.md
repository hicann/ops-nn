# aclnnChamferDistanceBackward

## 支持的产品型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。

## 功能说明

- 算子功能：ChamferDistance(倒角距离)的反向算子，根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。
- 计算公式：

  假设有两个点集：  xyz1=[B,N,2], xyz2=[B,M,2]

  - ChamferDistance(倒角距离)正向算子计算公式为：

    $dist1_i=Min((x_{1_i}−x_2)^2+(y_{1_i}−y_2)^2)，x_2, y_2∈xyz2$
    $dist2_i=Min((x_{2_i}-x_1)^2+(y_{2_i}-y_1)^2)，x_1,y_1∈xyz1$

  - 反向算子即为对该公式求导，计算公式为：
    - $dist1_i 对x_{1_i} $的导数$=2*grad\_dist1*(x_{1_i}-x_2)$

      其中：$x_{1_i}∈xyz1$，$x_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的横坐标，单点求导公式如上，因为单点梯度更新的位置是连续的，所以考虑多点并行计算。

    - $dist1_i 对y_{1_i} $的导数$=2*grad\_dist1*(y_{1_i}-y_2)$

      其中$y_{1_i}∈xyz1$，$y_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的纵坐标，单点求导公式如上，因为单点梯度更新的位置是连续的，所以也可以考虑多点并行计算。

    - $dist1_i 对x_2 $的导数$=-2*grad\_dist1*(x_{1_i}-x_2)$

      其中$x_{1_i}∈xyz1，x_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的横坐标，单点求导公式如上，因为单点梯度需要根据最小距离值对应的索引值去更新，所以这块无法并行只能单点计算。

    - $dist1_i 对y_2 $的导数$=-2*grad\_dist1*(y_{1_i}-y_2)$

      其中$y_{1_i}∈xyz1，y_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的纵坐标，单点求导公式如上，因为单点梯度需要根据最小值对应的索引值去更新，所以这块也无法并行只能单点计算。

  对应$dist2_i$对$x_{2_i} 、x_1、y_{2_i} 、y_1$的导数和上述过程类似，这里不再赘述。

  最终计算公式如下，i∈[0,n)：

  $grad_xyz1[2*i] = 2*grad\_dist_1*(x_{1_i}-x_2) - 2*grad\_dist_1*(x_{1_i}-x_2)$

  $grad_xyz1[2*i+1] = 2*grad\_dist1*(y_{1_i}-y_2) - 2*grad\_dist1*(y_{1_i}-y_2)$

  $grad_xyz2[2*i] = 2*grad\_dist2*(x_{1_i}-x_2) - 2*grad\_dist2*(x_{1_i}-x_2)$

  $grad_xyz2[2*i+1] = 2*grad\_dist2*(y_{1_i}-y_2) - 2*grad\_dist2*(y_{1_i}-y_2)$

## 函数原型

每个算子分为[两段式接口](./../../../docs/context/两段式接口.md)，必须先调用“aclnnChamferDistanceBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnChamferDistanceBackward”接口执行计算。

- `aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2, const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1, aclTensor* gradXyz2, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnChamferDistanceBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnChamferDistanceBackwardGetWorkspaceSize

- **参数说明**：
  - xyz1（aclTensor\*, 计算输入）：算子正向输入的点集1的坐标，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与xyz2、grad_dist1、grad_dist2、grad_xyz1、grad_xyz2一致，shape为（B,N,2），[数据格式](../../../docs/context/数据格式.md)支持ND
  - xyz2（aclTensor\*, 计算输入）：算子正向输入的点集2的坐标，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与xyz2、grad_dist1、grad_dist2、grad_xyz1、grad_xyz2一致，shape为（B,N,2），[数据格式](../../../docs/context/数据格式.md)支持ND
  - idx1（aclTensor\*, 计算输入）：算子正向输出的距离xyz1最小距离的xyz2中的点的索引tensor，Device侧的aclTensor，数据类型支持INT32，且数据类型与idx2一致，shape为（B,N），[数据格式](../../../docs/context/数据格式.md)支持ND
  - idx2（aclTensor\*, 计算输入）：算子正向输出的距离xyz2最小距离的xyz1中的点的索引tensor，Device侧的aclTensor，数据类型支持INT32，且数据类型与idx1一致，shape为（B,N），[数据格式](../../../docs/context/数据格式.md)支持ND
  - gradDist1（aclTensor\*, 计算输入）：正向输出dist1的反向梯度，也是反向算子的初始梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与xyz2、xyz1、grad_dist2、grad_xyz1、grad_xyz2一致，shape为（B,N），[数据格式](../../../docs/context/数据格式.md)支持ND
  - gradDist2（aclTensor\*, 计算输出）：正向输出dist2的反向梯度，也是反向算子的初始梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与xyz2、xyz1、grad_dist2、grad_xyz1、grad_xyz2一致，shape为（B,N），[数据格式](../../../docs/context/数据格式.md)支持ND
  - gradXyz1（aclTensor\*, 计算输出）：梯度更新之后正向算子输入xyz1对应的梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与xyz1、xyz2、grad_dist1、grad_dist2、grad_xyz2一致，shape为（B,N,2），[数据格式](../../../docs/context/数据格式.md)支持ND
  - gradXyz2（aclTensor\*, 计算输出）：正向输出dist2的反向梯度，也是反向算子的初始梯度，Device侧的aclTensor，数据类型支持FLOAT、FLOAT16，且数据类型与xyz1、xyz2、grad_dist1、grad_dist2、grad_xyz1一致，shape为（B,N,2），[数据格式](../../../docs/context/数据格式.md)支持ND
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\**, 出参）：返回op执行器，包含了算子计算流程。
- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）: 1. 传入的xyz1、xyz2等输入或grad_xyz1、grad_xyz2是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）: 1. 输入和输出的数据类型不在支持的范围之内。
                                        2. 输入无法做数据类型推导。
                                        3. 推导出的数据类型无法转换为指定输出out的类型。
  ```

## aclnnChamferDistanceBackward

- **参数说明**：

  - workspace（void\*, 入参）：在Device侧申请的workspace内存地址。
  - workspaceSize（uint64_t, 入参）：在Device侧申请的workspace大小，由第一段接口aclnnChamferDistanceBackwardGetWorkspaceSize获取。
  - executor（aclOpExecutor\*, 入参）：op执行器，包含了算子计算流程。
  - stream（aclrtStream, 入参）：指定执行任务的AscendCL Stream流。
- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](./../../../docs/context/aclnn返回码.md)。

## 约束说明
无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chamfer_distance_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                               \
        if (!(cond)) {                   \
        return_expr;                   \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                              \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
  }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
       strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.(固定写法)device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xyz1Shape = {2, 2, 2};
    std::vector<int64_t> xyz2Shape = {2, 2, 2};
    std::vector<int64_t> idx1Shape = {2, 2};
    std::vector<int64_t> idx2Shape = {2, 2};
    std::vector<int64_t> gradDist1Shape = {2, 2};
    std::vector<int64_t> gradDist2Shape = {2, 2};
    std::vector<int64_t> gradXyz1Shape = {2, 2, 2};
    std::vector<int64_t> gradXyz2Shape = {2, 2, 2};
    void* xyz1DeviceAddr = nullptr;
    void* xyz2DeviceAddr = nullptr;
    void* idx1DeviceAddr = nullptr;
    void* idx2DeviceAddr = nullptr;
    void* gradDist1DeviceAddr = nullptr;
    void* gradDist2DeviceAddr = nullptr;
    void* gradXyz1DeviceAddr = nullptr;
    void* gradXyz2DeviceAddr = nullptr;
    aclTensor* xyz1 = nullptr;
    aclTensor* xyz2 = nullptr;
    aclTensor* idx1 = nullptr;
    aclTensor* idx2 = nullptr;
    aclTensor* gradDist1 = nullptr;
    aclTensor* gradDist2 = nullptr;
    aclTensor* gradXyz1 = nullptr;
    aclTensor* gradXyz2 = nullptr;
    std::vector<float> xyz1HostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> xyz2HostData = {1, 1, 1, 2, 2, 2, 3, 3};
    std::vector<int32_t> idx1HostData = {0, 1, 2, 3};
    std::vector<int32_t> idx2HostData = {0, 1, 2, 3};
    std::vector<float> gradDist1HostData = {0, 1, 2, 3};
    std::vector<float> gradDist2HostData = {0, 1, 2, 3};
    std::vector<float> gradXyz1HostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradXyz2HostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // 创建xyz1 aclTensor
    ret = CreateAclTensor(xyz1HostData, xyz1Shape, &xyz1DeviceAddr, aclDataType::ACL_FLOAT, &xyz1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建xyz2 aclTensor
    ret = CreateAclTensor(xyz2HostData, xyz2Shape, &xyz2DeviceAddr, aclDataType::ACL_FLOAT, &xyz2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建idx1 aclTensor
    ret = CreateAclTensor(idx1HostData, idx1Shape, &idx1DeviceAddr, aclDataType::ACL_INT32, &idx1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建idx2 aclTensor
    ret = CreateAclTensor(idx2HostData, idx2Shape, &idx2DeviceAddr, aclDataType::ACL_INT32, &idx2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradDist1 aclTensor
    ret = CreateAclTensor(gradDist1HostData, gradDist1Shape, &gradDist1DeviceAddr, aclDataType::ACL_FLOAT, &gradDist1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradDist2 aclTensor
    ret = CreateAclTensor(gradDist2HostData, gradDist2Shape, &gradDist2DeviceAddr, aclDataType::ACL_FLOAT, &gradDist2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradXyz1 aclTensor
    ret = CreateAclTensor(gradXyz1HostData, gradXyz1Shape, &gradXyz1DeviceAddr, aclDataType::ACL_FLOAT, &gradXyz1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradXyz2 aclTensor
    ret = CreateAclTensor(gradXyz2HostData, gradXyz2Shape, &gradXyz2DeviceAddr, aclDataType::ACL_FLOAT, &gradXyz2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnChamferDistanceBackward第一段接口
    ret = aclnnChamferDistanceBackwardGetWorkspaceSize(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2,
                                                       &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChamferDistanceBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnChamferDistanceBackward第二段接口
    ret = aclnnChamferDistanceBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChamferDistanceBackward failed. ERROR: %d\n", ret); return ret);

    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradXyz1Shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXyz1DeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result1[%ld] is: %f\n", i, resultData[i]);
    }

    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradXyz2DeviceAddr,
                      size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result2[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(xyz1);
    aclDestroyTensor(xyz2);
    aclDestroyTensor(idx1);
    aclDestroyTensor(idx2);
    aclDestroyTensor(gradDist1);
    aclDestroyTensor(gradDist2);
    aclDestroyTensor(gradXyz1);
    aclDestroyTensor(gradXyz2);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xyz1DeviceAddr);
    aclrtFree(xyz2DeviceAddr);
    aclrtFree(idx1DeviceAddr);
    aclrtFree(idx2DeviceAddr);
    aclrtFree(gradDist1DeviceAddr);
    aclrtFree(gradDist2DeviceAddr);
    aclrtFree(gradXyz1DeviceAddr);
    aclrtFree(gradXyz2DeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```