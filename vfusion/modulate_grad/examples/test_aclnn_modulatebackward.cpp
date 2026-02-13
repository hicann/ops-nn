#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_modulate_grad.h"

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


constexpr size_t MEMORY_ALIGNMENT = 32;

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void* AlignedMalloc(size_t size) {
    size_t aligned_size = (size + MEMORY_ALIGNMENT - 1) / MEMORY_ALIGNMENT * MEMORY_ALIGNMENT;
    void* ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, aligned_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Aligned aclrtMalloc failed. ERROR: %d\n", ret); return nullptr);
    return ptr;
}

void PrintOutResult(const std::vector<int64_t>& shape, void* deviceAddr, const std::string& name) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float),
                            deviceAddr,size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR:%d\n", name.c_str(), ret); return );

    int print_count = std::min(static_cast<int64_t>(10),size);
    LOG_PRINT("%s (first %d elements):\n", name.c_str(), print_count);
    for (int64_t i = 0; i < print_count; i++) {
        LOG_PRINT("[%ld]: %f\n", i, resultData[i]);
    }
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
int CreateInputTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    *deviceAddr = AlignedMalloc(size);
    CHECK_RET(*deviceAddr != nullptr, LOG_PRINT("AlignedMalloc failed for input tensor\n"); return -1);

    auto ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                                aclFormat::ACL_FORMAT_ND,shape.data(), shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

template <typename T>
int CreateOutputTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    *deviceAddr = AlignedMalloc(size);
    CHECK_RET(*deviceAddr != nullptr, LOG_PRINT("AlignedMalloc failed for output tensor\n"); return -1);

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0,
                                aclFormat::ACL_FORMAT_ND, shape.data(),shape.size(), *deviceAddr);
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return -1);
    return 0;
}

std::vector<float> GenerateRandomData(int64_t size, float min = -1.0f, float max = 1.0f) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    for (int64_t i = 0; i < size; i++)
    {
        data[i] = dis(gen);
    }
    return data;
}

int main(){
    int32_t deviceId = 1;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> grad_outputShape = {2,3,4};
    std::vector<int64_t> inputShape = {2,3,4};
    std::vector<int64_t> scaleShape = {2,4};
    std::vector<int64_t> shiftShape = {2,4};
    std::vector<int64_t> grad_inputShape = {2,3,4};
    std::vector<int64_t> grad_scaleShape = {2,4};
    std::vector<int64_t> grad_shiftShape = {2,4};

    int64_t grad_outputSize = GetShapeSize(grad_outputShape);
    int64_t inputSize = GetShapeSize(inputShape);
    int64_t scaleSize = GetShapeSize(scaleShape);
    int64_t shiftSize = GetShapeSize(shiftShape);

    std::vector<float> grad_outputHostData(24,2);
    std::vector<float> inputHostData(24,2);
    std::vector<float> scaleHostData(8,2);
    std::vector<float> shiftHostData(8,2);

    void* grad_outputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* grad_inputDeviceAddr = nullptr;
    void* grad_scaleDeviceAddr = nullptr;
    void* grad_shiftDeviceAddr = nullptr;

    aclTensor* grad_output = nullptr;
    aclTensor* input = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* grad_input = nullptr;
    aclTensor* grad_scale = nullptr;
    aclTensor* grad_shift = nullptr;

    ret = CreateInputTensor(grad_outputHostData, grad_outputShape, &grad_outputDeviceAddr,
                            aclDataType::ACL_FLOAT, &grad_output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateInputTensor(inputHostData, inputShape, &inputDeviceAddr,
                            aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateInputTensor(scaleHostData, scaleShape, &scaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateInputTensor(shiftHostData, shiftShape, &shiftDeviceAddr,
                            aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateOutputTensor<float>(grad_inputShape, &grad_inputDeviceAddr,
                                    aclDataType::ACL_FLOAT, &grad_input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateOutputTensor<float>(grad_scaleShape, &grad_scaleDeviceAddr,
                                    aclDataType::ACL_FLOAT, &grad_scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateOutputTensor<float>(grad_shiftShape, &grad_shiftDeviceAddr,
                                    aclDataType::ACL_FLOAT, &grad_shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnModulateBackwardGetWorkspaceSize(grad_output, input, scale, shift,
                                                grad_input, grad_scale, grad_shift,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        workspaceAddr = AlignedMalloc(workspaceSize);
        CHECK_RET(workspaceAddr != nullptr, LOG_PRINT("allocate workspace failed\n"); return -1);
    }
    ret = aclnnModulateBackward(workspaceAddr,workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulateBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    PrintOutResult(grad_inputShape, grad_inputDeviceAddr, "grad_input");
    PrintOutResult(grad_scaleShape, grad_scaleDeviceAddr, "grad_scale");
    PrintOutResult(grad_shiftShape, grad_shiftDeviceAddr, "grad_shift");

    aclDestroyTensor(grad_output);
    aclDestroyTensor(input);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(grad_input);
    aclDestroyTensor(grad_scale);
    aclDestroyTensor(grad_shift);

    aclrtFree(grad_outputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(grad_inputDeviceAddr);
    aclrtFree(grad_scaleDeviceAddr);
    aclrtFree(grad_shiftDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}