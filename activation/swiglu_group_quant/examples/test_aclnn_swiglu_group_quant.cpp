#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return ACL_SUCCESS;
}

bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: SwigluGroupQuant only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
}

struct AclTensorResource {
    void* deviceAddr = nullptr;
    aclTensor* tensor = nullptr;
};

void DestroyAclTensorResource(AclTensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
        resource.tensor = nullptr;
    }
    if (resource.deviceAddr != nullptr) {
        aclrtFree(resource.deviceAddr);
        resource.deviceAddr = nullptr;
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                              shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

struct SwigluGroupQuantCase {
    const char* name;
    int64_t quantMode;
    int64_t blockSize;
    bool roundScale;
    std::vector<int64_t> yScaleShape;
    aclDataType yScaleDataType;
    bool yScaleIsE8M0;
};

int RunSwigluGroupQuantCase(const SwigluGroupQuantCase& testCase, aclrtStream stream)
{
    std::vector<int64_t> xShape = {2, 256};
    std::vector<int64_t> yShape = {2, 128};
    std::vector<int64_t> yOriginShape = {2, 128};

    std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
    for (size_t i = 0; i < xHostData.size(); ++i) {
        xHostData[i] = static_cast<uint16_t>(i % 23);
    }
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleE8M0HostData(GetShapeSize(testCase.yScaleShape), 0);
    std::vector<float> yScaleFp32HostData(GetShapeSize(testCase.yScaleShape), 0.0f);
    std::vector<uint16_t> yOriginHostData(GetShapeSize(yOriginShape), 0);

    AclTensorResource xResource;
    AclTensorResource yResource;
    AclTensorResource yScaleResource;
    AclTensorResource yOriginResource;

    auto ret = CreateAclTensor(xHostData, xShape, &xResource.deviceAddr, ACL_FLOAT16, &xResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yResource.deviceAddr, ACL_FLOAT8_E4M3FN, &yResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); return ret);
    if (testCase.yScaleIsE8M0) {
        ret = CreateAclTensor(yScaleE8M0HostData, testCase.yScaleShape, &yScaleResource.deviceAddr,
                              testCase.yScaleDataType, &yScaleResource.tensor);
    } else {
        ret = CreateAclTensor(yScaleFp32HostData, testCase.yScaleShape, &yScaleResource.deviceAddr,
                              testCase.yScaleDataType, &yScaleResource.tensor);
    }
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); return ret);
    ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginResource.deviceAddr, ACL_FLOAT16,
                          &yOriginResource.tensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource); return ret);

    int64_t dstType = 36;
    double clampLimit = -1.0;
    double dstTypeMax = 448.0;
    bool outputOrigin = false;

    LOG_PRINT("Run %s: quant_mode=%ld\n", testCase.name, testCase.quantMode);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnSwigluGroupQuantGetWorkspaceSize(xResource.tensor, nullptr, nullptr, nullptr, dstType,
                                                testCase.quantMode, testCase.blockSize, testCase.roundScale, clampLimit,
                                                dstTypeMax, outputOrigin, yResource.tensor, yScaleResource.tensor,
                                                yOriginResource.tensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
              DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  DestroyAclTensorResource(xResource); DestroyAclTensorResource(yResource);
                  DestroyAclTensorResource(yScaleResource); DestroyAclTensorResource(yOriginResource); return ret);
    }

    ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); return ret);

    std::vector<uint8_t> resultData(GetShapeSize(yShape), 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yResource.deviceAddr,
                      resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } DestroyAclTensorResource(xResource);
        DestroyAclTensorResource(yResource); DestroyAclTensorResource(yScaleResource);
        DestroyAclTensorResource(yOriginResource); return ret);
    LOG_PRINT("%s result[0] is: %d\n", testCase.name, resultData[0]);

    DestroyAclTensorResource(xResource);
    DestroyAclTensorResource(yResource);
    DestroyAclTensorResource(yScaleResource);
    DestroyAclTensorResource(yOriginResource);
    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return ACL_SUCCESS;
    }

    std::vector<SwigluGroupQuantCase> testCases = {
        {"block_fp8", 0, 0, false, {2, 1}, ACL_FLOAT, false},
        {"mx_fp8", 1, 0, true, {2, 2, 2}, ACL_FLOAT8_E8M0, true},
    };

    for (const auto& testCase : testCases) {
        ret = RunSwigluGroupQuantCase(testCase, stream);
        CHECK_RET(ret == ACL_SUCCESS, Finalize(deviceId, stream); return ret);
    }

    Finalize(deviceId, stream);
    return ACL_SUCCESS;
}
