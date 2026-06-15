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
 * @file test_aclnn_hard_sigmoid_v2.cpp
 * @brief ACLNN invocation example for experimental HardSigmoidV2 operator
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_hardsigmoid_v2.h"

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

namespace {
struct HardSigmoidConfig {
    aclDataType acl_dtype;
    size_t element_size;
    std::string name;
};

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

bool ParseDtype(const std::string &dtype_name, HardSigmoidConfig *config)
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
    if (dtype_name == "int32") {
        *config = {ACL_INT32, sizeof(int32_t), "int32"};
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

int RunHardSigmoidV2(
    const std::vector<char> &input_host, const std::vector<int64_t> &shape, const HardSigmoidConfig &config,
    std::vector<char> *output_host, int32_t device_id)
{
    int final_ret = ACL_SUCCESS;
    bool acl_initialized = false;
    bool device_set = false;
    aclrtStream stream = nullptr;
    void *input_device = nullptr;
    void *output_device = nullptr;
    void *workspace = nullptr;
    aclTensor *input_tensor = nullptr;
    aclTensor *output_tensor = nullptr;
    aclOpExecutor *executor = nullptr;
    uint64_t workspace_size = 0;
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * config.element_size;

    auto cleanup = [&]() -> int {
        if (input_tensor != nullptr) {
            aclDestroyTensor(input_tensor);
        }
        if (output_tensor != nullptr) {
            aclDestroyTensor(output_tensor);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (input_device != nullptr) {
            aclrtFree(input_device);
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
        final_ret = ret;
        return cleanup();
    }
    acl_initialized = true;
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }
    device_set = true;
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    if (bytes > 0) {
        ret = aclrtMalloc(&input_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
            return cleanup();
        }
        ret = aclrtMalloc(&output_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
            return cleanup();
        }

        ret = aclrtMemcpy(input_device, bytes, input_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
            return cleanup();
        }
        std::vector<char> zero_buffer(bytes, 0);
        ret = aclrtMemcpy(output_device, bytes, zero_buffer.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
            return cleanup();
        }
    }

    ret = CreateAclTensor(shape, config.acl_dtype, input_device, &input_tensor);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }
    ret = CreateAclTensor(shape, config.acl_dtype, output_device, &output_tensor);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    ret = aclnnHardsigmoidV2GetWorkspaceSize(input_tensor, output_tensor, &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
            return cleanup();
        }
    }

    ret = aclnnHardsigmoidV2(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ret;
        return cleanup();
    }

    output_host->resize(bytes);
    if (bytes > 0) {
        ret = aclrtMemcpy(output_host->data(), bytes, output_device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            final_ret = ret;
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
    HardSigmoidConfig config{ACL_FLOAT, sizeof(float), "fp32"};
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> input = {-4.0f, -3.0f, -2.0f, 0.0f, 1.0f, 2.0f, 4.0f, 5.0f};
    std::vector<char> output;
    auto ret = RunHardSigmoidV2(ToBytes(input), shape, config, &output, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const float *result = reinterpret_cast<const float *>(output.data());
    const std::vector<float> expected = {0.0f, 0.0f, 1.0f / 6.0f, 0.5f, 2.0f / 3.0f, 5.0f / 6.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(result[i] - expected[i]) > 1e-5f) {
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

    if (argc != 6) {
        std::fprintf(stderr, "Usage: %s <dtype> <shape|scalar> <input.bin> <output.bin> <device_id>\n", argv[0]);
        return 2;
    }

    HardSigmoidConfig config{};
    CHECK_RET(ParseDtype(argv[1], &config), return 2);

    std::vector<int64_t> shape;
    CHECK_RET(ParseShape(argv[2], &shape), return 2);

    std::vector<char> input;
    CHECK_RET(ReadFile(argv[3], &input), return 2);

    std::vector<char> output;
    auto ret = RunHardSigmoidV2(input, shape, config, &output, std::atoi(argv[5]));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(WriteFile(argv[4], output), return 2);
    return 0;
}
