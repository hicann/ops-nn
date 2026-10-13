/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_mask.h"

namespace {

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (const int64_t dim : shape) {
        size *= dim;
    }
    return size;
}

std::vector<int64_t> GetContiguousStrides(const std::vector<int64_t>& shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

template <typename T>
aclError CreateTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, aclDataType dataType,
                      aclFormat format, void** deviceAddress, aclTensor** tensor, bool strided = false)
{
    std::vector<T> storage = hostData;
    std::vector<int64_t> storageShape = shape;
    std::vector<int64_t> strides = GetContiguousStrides(shape);
    if (strided) {
        storage.assign(hostData.size() * 2, static_cast<T>(12345));
        for (size_t i = 0; i < hostData.size(); ++i) {
            storage[i * 2] = hostData[i];
        }
        storageShape.back() *= 2;
        for (auto& stride : strides) {
            stride *= 2;
        }
    }
    const size_t bytes = storage.size() * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMalloc failed: %d\n", ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddress, bytes, storage.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMemcpy host to device failed: %d\n", ret);
        return ret;
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format, storageShape.data(),
                              storageShape.size(), *deviceAddress);
    if (*tensor == nullptr) {
        std::printf("aclCreateTensor failed.\n");
        return ACL_ERROR_FAILURE;
    }
    return ACL_SUCCESS;
}

bool VerifyResults(const std::vector<float>& output, const std::vector<uint8_t>& indices,
                   const std::vector<float>& expectedOutput, const std::vector<int32_t>& expectedIndices)
{
    for (size_t i = 0; i < expectedOutput.size(); ++i) {
        if (std::memcmp(&output[i], &expectedOutput[i], sizeof(float)) != 0) {
            std::printf("output mismatch at %zu: actual=%f expected=%f\n", i, output[i], expectedOutput[i]);
            return false;
        }
        int32_t actualIndex = 0;
        std::memcpy(&actualIndex, indices.data() + i * sizeof(int32_t), sizeof(int32_t));
        if (actualIndex != expectedIndices[i]) {
            std::printf("index mismatch at %zu: actual=%d expected=%d\n", i, actualIndex, expectedIndices[i]);
            return false;
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    // Optional workload arguments let the official msopprof tool replay representative kernels.
    // This program checks results and does not implement a performance collector.
    int64_t channels = 1;
    int64_t height = 4;
    int64_t width = 4;
    int64_t window = 2;
    int64_t step = 2;
    int64_t pad = 0;
    if (argc != 1 && argc != 7 && argc != 8 && argc != 9) {
        std::fprintf(stderr,
                     "Usage: %s [channels height width kernel stride padding [normal|ties|zeros|adjacent "
                     "[strided|strided-input|strided-output]]]\n",
                     argv[0]);
        return 2;
    }
    if (argc >= 7) {
        int64_t* fields[] = {&channels, &height, &width, &window, &step, &pad};
        for (int i = 1; i <= 6; ++i) {
            char* end = nullptr;
            const long value = std::strtol(argv[i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || value < 0 || value > 4096) {
                std::fprintf(stderr, "Invalid workload argument: %s\n", argv[i]);
                return 2;
            }
            *fields[i - 1] = value;
        }
    }
    if (channels == 0 || height == 0 || width == 0 || window == 0 || step == 0 || pad > window / 2 ||
        height + 2 * pad < window || width + 2 * pad < window) {
        std::fprintf(stderr, "Invalid pooling geometry.\n");
        return 2;
    }
    const int64_t outH = (height + 2 * pad - window) / step + 1;
    const int64_t outW = (width + 2 * pad - window) / step + 1;
    const int64_t maskW = ((outH * outW + 15) / 16 + 1) * 32;
    if (window * window * maskW < outH * outW * 4 || channels * height * width > 64 * 1024 * 1024 ||
        channels * window * window * maskW > 512 * 1024 * 1024) {
        std::fprintf(stderr, "Workload exceeds example capacity.\n");
        return 2;
    }
    const char* pattern = argc >= 8 ? argv[7] : "normal";
    if (std::strcmp(pattern, "normal") && std::strcmp(pattern, "ties") && std::strcmp(pattern, "zeros") &&
        std::strcmp(pattern, "adjacent")) {
        std::fprintf(stderr, "Unknown input pattern: %s\n", pattern);
        return 2;
    }
    const bool strided = argc == 9;
    if (strided && std::strcmp(argv[8], "strided") != 0 && std::strcmp(argv[8], "strided-input") != 0 &&
        std::strcmp(argv[8], "strided-output") != 0) {
        std::fprintf(stderr, "Expected strided as the final argument.\n");
        return 2;
    }
    const bool stridedInput = strided && std::strcmp(argv[8], "strided-output") != 0;
    const bool stridedOutput = strided && std::strcmp(argv[8], "strided-input") != 0;
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclError ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        std::printf("aclInit failed: %d\n", ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSetDevice failed: %d\n", ret);
        return ret;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtCreateStream failed: %d\n", ret);
        return ret;
    }

    const std::vector<int64_t> inputShape = {1, channels, height, width};
    const std::vector<int64_t> outputShape = {1, channels, outH, outW};
    const std::vector<int64_t> indicesShape = {1, channels, window * window, maskW};
    std::vector<float> inputHost(GetShapeSize(inputShape));
    for (size_t i = 0; i < inputHost.size(); ++i) {
        inputHost[i] = argc == 1 ?
                           static_cast<float>(i + 1) :
                           static_cast<float>(static_cast<int32_t>((i * 65537 + 17) % 131071) - 65535) / 65536.0F;
    }
    for (size_t i = 0; i < inputHost.size(); ++i) {
        if (std::strcmp(pattern, "ties") == 0) {
            inputHost[i] = static_cast<float>((i * 13 + i / width) % 7);
        } else if (std::strcmp(pattern, "zeros") == 0) {
            inputHost[i] = i % 3 == 0 ? -0.0F : 0.0F;
        } else if (std::strcmp(pattern, "adjacent") == 0) {
            const uint32_t bits = 0x3F800000U + static_cast<uint32_t>((i * 13 + i / width) % 17);
            std::memcpy(&inputHost[i], &bits, sizeof(bits));
        }
    }
    std::vector<float> outputHost(GetShapeSize(outputShape), 0.0F);
    std::vector<uint8_t> indicesHost(GetShapeSize(indicesShape), 0x7fU);
    std::vector<float> expectedOutput(outputHost.size(), -std::numeric_limits<float>::infinity());
    std::vector<int32_t> expectedIndices(outputHost.size(), 0);
    for (int64_t c = 0; c < channels; ++c) {
        for (int64_t oh = 0; oh < outH; ++oh) {
            for (int64_t ow = 0; ow < outW; ++ow) {
                const int64_t outIndex = (c * outH + oh) * outW + ow;
                for (int64_t kh = 0; kh < window; ++kh) {
                    const int64_t ih = oh * step - pad + kh;
                    if (ih < 0 || ih >= height)
                        continue;
                    for (int64_t kw = 0; kw < window; ++kw) {
                        const int64_t iw = ow * step - pad + kw;
                        if (iw < 0 || iw >= width)
                            continue;
                        const float value = inputHost[(c * height + ih) * width + iw];
                        if (value > expectedOutput[outIndex]) {
                            expectedOutput[outIndex] = value;
                            expectedIndices[outIndex] = static_cast<int32_t>(ih * width + iw);
                        }
                    }
                }
            }
        }
    }

    void* inputDevice = nullptr;
    void* outputDevice = nullptr;
    void* indicesDevice = nullptr;
    void* workspaceDevice = nullptr;
    aclTensor* input = nullptr;
    aclTensor* output = nullptr;
    aclTensor* indices = nullptr;

    ret = CreateTensor(inputHost, inputShape, ACL_FLOAT, ACL_FORMAT_NCHW, &inputDevice, &input, stridedInput);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = CreateTensor(outputHost, outputShape, ACL_FLOAT, ACL_FORMAT_NCHW, &outputDevice, &output, stridedOutput);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = CreateTensor(indicesHost, indicesShape, ACL_INT8, ACL_FORMAT_NCHW, &indicesDevice, &indices);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    const std::vector<int64_t> kernelSizeValue = {window, window};
    const std::vector<int64_t> strideValue = {step, step};
    const std::vector<int64_t> paddingValue = {pad, pad};
    const std::vector<int64_t> dilationValue = {1, 1};
    aclIntArray* kernelSize = aclCreateIntArray(kernelSizeValue.data(), kernelSizeValue.size());
    aclIntArray* stride = aclCreateIntArray(strideValue.data(), strideValue.size());
    aclIntArray* padding = aclCreateIntArray(paddingValue.data(), paddingValue.size());
    aclIntArray* dilation = aclCreateIntArray(dilationValue.data(), dilationValue.size());
    if (kernelSize == nullptr || stride == nullptr || padding == nullptr || dilation == nullptr) {
        std::printf("aclCreateIntArray failed.\n");
        return ACL_ERROR_FAILURE;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMaxPool2dWithMaskGetWorkspaceSize(input, kernelSize, stride, padding, dilation, false, output, indices,
                                                 &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::printf("aclnnMaxPool2dWithMaskGetWorkspaceSize failed: %d\n", ret);
        return ret;
    }
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            std::printf("workspace allocation failed: %d\n", ret);
            return ret;
        }
    }
    ret = aclnnMaxPool2dWithMask(workspaceDevice, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclnnMaxPool2dWithMask failed: %d\n", ret);
        return ret;
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtSynchronizeStream failed: %d\n", ret);
        return ret;
    }

    std::vector<float> outputStorage(outputHost.size() * (stridedOutput ? 2 : 1));
    ret = aclrtMemcpy(outputStorage.data(), outputStorage.size() * sizeof(float), outputDevice,
                      outputStorage.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) {
        ret = aclrtMemcpy(indicesHost.data(), indicesHost.size(), indicesDevice, indicesHost.size(),
                          ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (ret != ACL_SUCCESS) {
        std::printf("aclrtMemcpy device to host failed: %d\n", ret);
        return ret;
    }

    bool gapsUnchanged = true;
    for (size_t i = 0; i < outputHost.size(); ++i) {
        outputHost[i] = outputStorage[i * (stridedOutput ? 2 : 1)];
        if (stridedOutput && outputStorage[i * 2 + 1] != 12345.0F) {
            gapsUnchanged = false;
        }
    }
    const bool passed = VerifyResults(outputHost, indicesHost, expectedOutput, expectedIndices) && gapsUnchanged;
    std::printf("MaxPool2dWithMask ACLNN NCHW FP32 C=%ld H=%ld W=%ld K=%ld S=%ld P=%ld pattern=%s "
                "stridedInput=%d stridedOutput=%d: %s\n",
                channels, height, width, window, step, pad, pattern, stridedInput, stridedOutput,
                passed ? "PASS" : "FAIL");

    aclDestroyTensor(input);
    aclDestroyTensor(output);
    aclDestroyTensor(indices);
    aclDestroyIntArray(kernelSize);
    aclDestroyIntArray(stride);
    aclDestroyIntArray(padding);
    aclDestroyIntArray(dilation);
    aclrtFree(inputDevice);
    aclrtFree(outputDevice);
    aclrtFree(indicesDevice);
    if (workspaceDevice != nullptr) {
        aclrtFree(workspaceDevice);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return passed ? 0 : 1;
}
