#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_backward_mx_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        if (i != 0 && shapeSize > INT64_MAX / i) {
            return -1;
        }
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
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
                    aclDataType dataType, aclTensor** tensor)
{
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnSwigluBackwardMxQuantWithDualAxisTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 输入 x shape: [M, 2N] = [256, 1024]
    std::vector<int64_t> xShape = {256, 1024};
    // 输入 yGrad shape: [M, N] = [256, 512]
    std::vector<int64_t> yGradShape = {256, 512};
    // group_index shape: [G] = [1]，采用 cumsum 模式，值为 256 表示一个 group 包含所有 256 行
    std::vector<int64_t> groupIndexShape = {1};
    // 反向输出 shape: [M, 2N] = [256, 1024] (concat(gradA, gradB) 的量化结果)
    std::vector<int64_t> y1OutShape = {256, 1024};
    std::vector<int64_t> y2OutShape = {256, 1024};
    std::vector<int64_t> mxscale1OutShape = {256, 16, 2};
    std::vector<int64_t> mxscale2OutShape = {5, 1024, 2};

    void* xDeviceAddr = nullptr;
    void* yGradDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* mxscale1OutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* mxscale2OutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* yGrad = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* mxscale1Out = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* mxscale2Out = nullptr;

    // 输入数据初始化（BF16）
    std::vector<uint16_t> xHostData(256 * 1024, 0);
    for (int64_t i = 0; i < 256 * 1024; i++) {
        xHostData[i] = static_cast<uint16_t>(i % 100);
    }
    // yGrad 数据初始化（BF16）
    std::vector<uint16_t> yGradHostData(256 * 512, 0);
    for (int64_t i = 0; i < 256 * 512; i++) {
        yGradHostData[i] = static_cast<uint16_t>(i % 50);
    }
    // group_index 数据：cumsum 模式，值为 256 表示只有一个 group，包含所有行
    std::vector<int64_t> groupIndexHostData = {256};
    std::vector<uint8_t> y1OutHostData(256 * 1024, 0);
    std::vector<uint8_t> y2OutHostData(256 * 1024, 0);
    std::vector<uint8_t> mxscale1OutHostData(256 * 16 * 2, 0);
    std::vector<uint8_t> mxscale2OutHostData(5 * 1024 * 2, 0);

    // 参数设置
    bool activateLeft = true;
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstDtype = 36;  // FLOAT8_E4M3FN
    int64_t scaleAlg = 1;  // cuBLAS
    double maxDtypeValue = 0.0;

    // 创建 x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 yGrad aclTensor
    ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_BF16, &yGrad);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yGradTensorPtr(yGrad, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yGradDeviceAddrPtr(yGradDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 groupIndex aclTensor
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 y1Out aclTensor
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 mxscale1Out aclTensor
    ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 y2Out aclTensor
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 mxscale2Out aclTensor
    ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnSwigluBackwardMxQuantWithDualAxis第一段接口（带 groupIndex 输入）
    ret = aclnnSwigluBackwardMxQuantWithDualAxisGetWorkspaceSize(x, yGrad, groupIndex, activateLeft, roundModeOptional,
        scaleAlg, dstDtype, maxDtypeValue, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluBackwardMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnSwigluBackwardMxQuantWithDualAxis第二段接口
    ret = aclnnSwigluBackwardMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluBackwardMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size1 = GetShapeSize(y1OutShape);
    auto size2 = GetShapeSize(y2OutShape);
    std::vector<uint8_t> y1OutData(size1, 0);
    std::vector<uint8_t> y2OutData(size2, 0);

    ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(uint8_t), y1OutDeviceAddr,
                      size1 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(uint8_t), y2OutDeviceAddr,
                      size2 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out from device to host failed. ERROR: %d\n", ret);
              return ret);

    // 打印部分输出结果
    LOG_PRINT("y1Out first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size1; i++) {
        LOG_PRINT("y1Out[%ld] = %d\n", i, y1OutData[i]);
    }
    LOG_PRINT("y2Out first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size2; i++) {
        LOG_PRINT("y2Out[%ld] = %d\n", i, y2OutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnSwigluBackwardMxQuantWithDualAxisTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluBackwardMxQuantWithDualAxisTest failed. ERROR: %d\n", ret);
        return ret);

    Finalize(deviceId, stream);
    return 0;
}