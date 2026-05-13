/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_relu_grad_v2.cpp
 * @brief ACLNN invocation example for experimental ThresholdBackward(ReLU grad) operator
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_threshold_backward.h"

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

namespace {
struct ReluGradConfig {
    aclDataType acl_dtype;
    size_t element_size;
    std::string name;
};

int ReportAclError(const char *stage, int ret)
{
    std::fprintf(stderr, "%s failed, ret=%d, msg=%s\n", stage, ret, aclGetRecentErrMsg());
    return ret;
}

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (int64_t dim : shape) {
        shape_size *= dim;
    }
    return shape_size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t> &shape)
{
    if (shape.empty()) {
        return {};
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = shape[static_cast<size_t>(i + 1)] * strides[static_cast<size_t>(i + 1)];
    }
    return strides;
}

bool ParseDtype(const std::string &dtype_name, ReluGradConfig *config)
{
    if (dtype_name == "fp16" || dtype_name == "float16") {
        *config = {ACL_FLOAT16, sizeof(uint16_t), "fp16"};
        return true;
    }
    if (dtype_name == "fp32" || dtype_name == "float32") {
        *config = {ACL_FLOAT, sizeof(float), "fp32"};
        return true;
    }
    if (dtype_name == "bf16" || dtype_name == "bfloat16") {
        *config = {ACL_BF16, sizeof(uint16_t), "bf16"};
        return true;
    }
    if (dtype_name == "int8") {
        *config = {ACL_INT8, sizeof(int8_t), "int8"};
        return true;
    }
    if (dtype_name == "uint8") {
        *config = {ACL_UINT8, sizeof(uint8_t), "uint8"};
        return true;
    }
    if (dtype_name == "int32") {
        *config = {ACL_INT32, sizeof(int32_t), "int32"};
        return true;
    }
    if (dtype_name == "int64") {
        *config = {ACL_INT64, sizeof(int64_t), "int64"};
        return true;
    }
    return false;
}

bool ParseShape(const std::string &shape_text, std::vector<int64_t> *shape)
{
    shape->clear();
    if (shape_text.empty() || shape_text == "scalar") {
        return true;
    }

    size_t start = 0;
    while (start < shape_text.size()) {
        size_t end = shape_text.find(',', start);
        std::string token = shape_text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (token.empty()) {
            return false;
        }
        shape->push_back(std::stoll(token));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return true;
}

bool ReadFile(const std::string &path, std::vector<char> *buffer)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }
    stream.seekg(0, std::ios::end);
    std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    if (size < 0) {
        return false;
    }
    buffer->resize(static_cast<size_t>(size));
    return size == 0 || stream.read(buffer->data(), size).good();
}

bool WriteFile(const std::string &path, const std::vector<char> &buffer)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }
    stream.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return stream.good();
}

aclError CreateAclTensor(
    const std::vector<int64_t> &shape, aclDataType dtype, void *device_addr, aclTensor **tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    const int64_t *shape_ptr = shape.empty() ? nullptr : shape.data();
    const int64_t *strides_ptr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(
        shape_ptr, shape.size(), dtype, strides_ptr, 0, ACL_FORMAT_ND, shape_ptr, shape.size(), device_addr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int RunReluGradV2(const std::vector<char> &gradients_host, const std::vector<char> &features_host,
                  const std::vector<int64_t> &shape, const ReluGradConfig &config, std::vector<char> *output_host,
                  int32_t device_id)
{
    int final_ret = ACL_SUCCESS;
    bool acl_initialized = false;
    bool device_set = false;
    aclrtStream stream = nullptr;
    void *gradients_device = nullptr;
    void *features_device = nullptr;
    void *output_device = nullptr;
    void *workspace = nullptr;
    aclTensor *gradients_tensor = nullptr;
    aclTensor *features_tensor = nullptr;
    aclTensor *output_tensor = nullptr;
    aclScalar *threshold_scalar = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspace_size = 0;
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * config.element_size;
    std::vector<char> zero_buffer(bytes, 0);

    auto cleanup = [&]() -> int {
        if (gradients_tensor != nullptr) {
            aclDestroyTensor(gradients_tensor);
        }
        if (features_tensor != nullptr) {
            aclDestroyTensor(features_tensor);
        }
        if (output_tensor != nullptr) {
            aclDestroyTensor(output_tensor);
        }
        if (threshold_scalar != nullptr) {
            aclDestroyScalar(threshold_scalar);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (gradients_device != nullptr) {
            aclrtFree(gradients_device);
        }
        if (features_device != nullptr) {
            aclrtFree(features_device);
        }
        if (output_device != nullptr) {
            aclrtFree(output_device);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (device_set) {
            aclrtResetDevice(device_id);
        }
        if (acl_initialized) {
            aclFinalize();
        }
        return final_ret;
    };

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclInit", ret);
        return cleanup();
    }
    acl_initialized = true;
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclrtSetDevice", ret);
        return cleanup();
    }
    device_set = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclrtCreateStream", ret);
        return cleanup();
    }

    if (bytes > 0) {
        ret = aclrtMalloc(&gradients_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMalloc(gradients)", ret);
            return cleanup();
        }
        ret = aclrtMalloc(&features_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMalloc(features)", ret);
            return cleanup();
        }
        ret = aclrtMalloc(&output_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMalloc(output)", ret);
            return cleanup();
        }

        ret = aclrtMemcpy(gradients_device, bytes, gradients_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMemcpy(gradients H2D)", ret);
            return cleanup();
        }
        ret = aclrtMemcpy(features_device, bytes, features_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMemcpy(features H2D)", ret);
            return cleanup();
        }
        ret = aclrtMemcpy(output_device, bytes, zero_buffer.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMemcpy(output H2D)", ret);
            return cleanup();
        }
    }

    ret = CreateAclTensor(shape, config.acl_dtype, gradients_device, &gradients_tensor);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("CreateAclTensor(gradients)", ret);
        return cleanup();
    }
    ret = CreateAclTensor(shape, config.acl_dtype, features_device, &features_tensor);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("CreateAclTensor(features)", ret);
        return cleanup();
    }
    ret = CreateAclTensor(shape, config.acl_dtype, output_device, &output_tensor);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("CreateAclTensor(output)", ret);
        return cleanup();
    }
    int32_t threshold_value = 0;
    threshold_scalar = aclCreateScalar(&threshold_value, ACL_INT32);
    if (threshold_scalar == nullptr) {
        final_ret = ReportAclError("aclCreateScalar(threshold)", ACL_ERROR_FAILURE);
        return cleanup();
    }

    ret = aclnnThresholdBackwardGetWorkspaceSize(
        gradients_tensor, features_tensor, threshold_scalar, output_tensor, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclnnThresholdBackwardGetWorkspaceSize", ret);
        return cleanup();
    }
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMalloc(workspace)", ret);
            return cleanup();
        }
    }

    ret = aclnnThresholdBackward(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclnnThresholdBackward", ret);
        return cleanup();
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclrtSynchronizeStream", ret);
        return cleanup();
    }

    output_host->resize(bytes);
    if (bytes > 0) {
        ret = aclrtMemcpy(output_host->data(), bytes, output_device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMemcpy(output D2H)", ret);
            return cleanup();
        }
    }

    final_ret = ret;
    return cleanup();
}

template <typename T>
std::vector<char> ToBytes(const std::vector<T> &values)
{
    std::vector<char> buffer(values.size() * sizeof(T));
    if (!buffer.empty()) {
        std::memcpy(buffer.data(), values.data(), buffer.size());
    }
    return buffer;
}

int RunDefaultExample()
{
    ReluGradConfig config{ACL_FLOAT, sizeof(float), "fp32"};
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> gradients = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> features = {-4.0f, -3.0f, -2.0f, 0.0f, 1.0f, 2.0f, 4.0f, 5.0f};
    std::vector<char> output;
    auto ret = RunReluGradV2(ToBytes(gradients), ToBytes(features), shape, config, &output, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const float *result = reinterpret_cast<const float *>(output.data());
    const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        if (result[i] != expected[i]) {
            std::fprintf(stderr, "default check failed at %zu: got %.8f expected %.8f\n", i, result[i], expected[i]);
            return 1;
        }
    }
    std::printf("default example passed\n");
    return 0;
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc == 1) {
        return RunDefaultExample();
    }

    if (argc != 7) {
        std::fprintf(stderr,
                     "Usage: %s <dtype> <shape|scalar> <gradients.bin> <features.bin> <output.bin> <device_id>\n",
                     argv[0]);
        return 2;
    }

    ReluGradConfig config{};
    CHECK_RET(ParseDtype(argv[1], &config), return 2);

    std::vector<int64_t> shape;
    CHECK_RET(ParseShape(argv[2], &shape), return 2);

    std::vector<char> gradients_host;
    CHECK_RET(ReadFile(argv[3], &gradients_host), return 3);
    std::vector<char> features_host;
    CHECK_RET(ReadFile(argv[4], &features_host), return 3);

    std::vector<char> output_host;
    int ret = RunReluGradV2(gradients_host, features_host, shape, config, &output_host, std::atoi(argv[6]));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(WriteFile(argv[5], output_host), return 4);
    return 0;
}
