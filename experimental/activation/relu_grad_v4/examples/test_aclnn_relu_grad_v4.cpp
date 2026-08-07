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
 * @file test_aclnn_relu_grad_v4.cpp
 * @brief ACLNN invocation example for experimental ReluGradV4(ReLU grad) operator
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_relu_grad_v4.h"

#define CHECK_RET(cond, expr) \
    do {                      \
        if (!(cond)) {        \
            expr;             \
        }                     \
    } while (0)

namespace {
struct ReluGradConfig {
    aclDataType acl_dtype;
    size_t element_size;
    std::string name;
};

int ReportAclError(const char* stage, int ret)
{
    std::fprintf(stderr, "%s error, ret=%d, msg=%s\n", stage, ret, aclGetRecentErrMsg());
    return ret;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (int64_t dim : shape) {
        shape_size *= dim;
    }
    return shape_size;
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape)
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

bool ParseDtype(const std::string& dtype_name, ReluGradConfig* config)
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
    return false;
}

bool ParseShape(const std::string& shape_text, std::vector<int64_t>* shape)
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

bool ReadFile(const std::string& path, std::vector<char>* buffer)
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

bool WriteFile(const std::string& path, const std::vector<char>& buffer)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream.is_open()) {
        return false;
    }
    stream.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    return stream.good();
}

aclError CreateAclTensor(const std::vector<int64_t>& shape, aclDataType dtype, void* device_addr, aclTensor** tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    const int64_t* shape_ptr = shape.empty() ? nullptr : shape.data();
    const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(shape_ptr, shape.size(), dtype, strides_ptr, 0, ACL_FORMAT_ND, shape_ptr, shape.size(),
                              device_addr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int RunReluGradV4(const std::vector<char>& gradients_host, const std::vector<char>& features_host,
                  const std::vector<int64_t>& shape, const ReluGradConfig& config, std::vector<char>* output_host,
                  int32_t device_id)
{
    int final_ret = ACL_SUCCESS;
    bool acl_initialized = false;
    bool device_set = false;
    aclrtStream stream = nullptr;
    void* gradients_device = nullptr;
    void* features_device = nullptr;
    void* output_device = nullptr;
    void* workspace = nullptr;
    aclTensor* gradients_tensor = nullptr;
    aclTensor* features_tensor = nullptr;
    aclTensor* output_tensor = nullptr;
    aclScalar* threshold_scalar = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * config.element_size;
    const size_t mask_bytes = static_cast<size_t>(GetShapeSize(shape)) * sizeof(uint8_t); // mask is uint8
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
        ret = aclrtMalloc(&features_device, mask_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
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
        ret = aclrtMemcpy(features_device, mask_bytes, features_host.data(), mask_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
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
    ret = CreateAclTensor(shape, ACL_UINT8, features_device, &features_tensor); // mask is uint8
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

    ret = aclnnReluGradV4GetWorkspaceSize(gradients_tensor, features_tensor, threshold_scalar, output_tensor,
                                          &workspace_size, &executor);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclnnReluGradV4GetWorkspaceSize", ret);
        return cleanup();
    }
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            final_ret = ReportAclError("aclrtMalloc(workspace)", ret);
            return cleanup();
        }
    }

    ret = aclnnReluGradV4(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) {
        final_ret = ReportAclError("aclnnReluGradV4", ret);
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
std::vector<char> ToBytes(const std::vector<T>& values)
{
    std::vector<char> buffer(values.size() * sizeof(T));
    if (!buffer.empty()) {
        std::memcpy(buffer.data(), values.data(), buffer.size());
    }
    return buffer;
}

// Encode an fp32 value into its fp16 (IEEE half) bit pattern. Sufficient for the small
// integer demo values used below (no subnormal/overflow handling needed).
static uint16_t FloatToFp16(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

// Encode an fp32 value into its bfloat16 bit pattern (truncate the low 16 mantissa bits).
static uint16_t FloatToBf16(float f)
{
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(x));
    return static_cast<uint16_t>(x >> 16);
}

int RunDefaultExample()
{
    ReluGradConfig config{ACL_FLOAT, sizeof(float), "fp32"};
    std::vector<int64_t> shape = {4, 2};
    std::vector<float> gradients = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    // mask is uint8 {0, 1}: 0 -> backprop 0, 1 -> backprop grad.
    std::vector<uint8_t> mask = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<char> output;
    auto ret = RunReluGradV4(ToBytes(gradients), ToBytes(mask), shape, config, &output, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const float* result = reinterpret_cast<const float*>(output.data());
    const std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    for (size_t i = 0; i < expected.size(); ++i) {
        if (result[i] != expected[i]) {
            std::fprintf(stderr, "default check mismatch at %zu: got %.8f expected %.8f\n", i, result[i], expected[i]);
            return 1;
        }
    }
    std::printf("default example passed\n");
    return 0;
}

// Run one call for each supported dtype using a fixed [4,8] shape to demonstrate multi-dtype aclnn invocation.
int RunAllExamples()
{
    std::vector<int64_t> shape = {4, 8};
    int pass = 0, err = 0;
    // Shared uint8 ReLU mask (values {0, 1}) reused by every dtype demo.
    std::vector<uint8_t> mask = {0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1};

    // float32
    {
        ReluGradConfig cfg{ACL_FLOAT, sizeof(float), "fp32"};
        std::vector<float> grad = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<char> out;
        int ret = RunReluGradV4(ToBytes(grad), ToBytes(mask), shape, cfg, &out, 0);
        std::printf("[%s] fp32  shape=[4,8] ret=%d\n", ret == 0 ? "PASS" : "NG", ret);
        ret == 0 ? ++pass : ++err;
    }
    // float16 (gradients encoded as IEEE half bit patterns)
    {
        ReluGradConfig cfg{ACL_FLOAT16, sizeof(uint16_t), "fp16"};
        std::vector<uint16_t> grad(16);
        for (int i = 0; i < 16; ++i) {
            grad[i] = FloatToFp16(static_cast<float>(i + 1));
        }
        std::vector<char> out;
        int ret = RunReluGradV4(ToBytes(grad), ToBytes(mask), shape, cfg, &out, 0);
        std::printf("[%s] fp16  shape=[4,8] ret=%d\n", ret == 0 ? "PASS" : "NG", ret);
        ret == 0 ? ++pass : ++err;
    }
    // bfloat16 (gradients encoded as bf16 bit patterns)
    {
        ReluGradConfig cfg{ACL_BF16, sizeof(uint16_t), "bf16"};
        std::vector<uint16_t> grad(16);
        for (int i = 0; i < 16; ++i) {
            grad[i] = FloatToBf16(static_cast<float>(i + 1));
        }
        std::vector<char> out;
        int ret = RunReluGradV4(ToBytes(grad), ToBytes(mask), shape, cfg, &out, 0);
        std::printf("[%s] bf16  shape=[4,8] ret=%d\n", ret == 0 ? "PASS" : "NG", ret);
        ret == 0 ? ++pass : ++err;
    }
    // int8
    {
        ReluGradConfig cfg{ACL_INT8, sizeof(int8_t), "int8"};
        std::vector<int8_t> grad = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<char> out;
        int ret = RunReluGradV4(ToBytes(grad), ToBytes(mask), shape, cfg, &out, 0);
        std::printf("[%s] int8  shape=[4,8] ret=%d\n", ret == 0 ? "PASS" : "NG", ret);
        ret == 0 ? ++pass : ++err;
    }
    // uint8
    {
        ReluGradConfig cfg{ACL_UINT8, sizeof(uint8_t), "uint8"};
        std::vector<uint8_t> grad = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<char> out;
        int ret = RunReluGradV4(ToBytes(grad), ToBytes(mask), shape, cfg, &out, 0);
        std::printf("[%s] uint8 shape=[4,8] ret=%d\n", ret == 0 ? "PASS" : "NG", ret);
        ret == 0 ? ++pass : ++err;
    }
    // int32
    {
        ReluGradConfig cfg{ACL_INT32, sizeof(int32_t), "int32"};
        std::vector<int32_t> grad = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        std::vector<char> out;
        int ret = RunReluGradV4(ToBytes(grad), ToBytes(mask), shape, cfg, &out, 0);
        std::printf("[%s] int32 shape=[4,8] ret=%d\n", ret == 0 ? "PASS" : "NG", ret);
        ret == 0 ? ++pass : ++err;
    }

    std::printf("RunAllExamples: pass=%d err=%d\n", pass, err);
    return err == 0 ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
    if (argc == 1) {
        int ret = RunAllExamples();
        if (ret != 0) {
            return ret;
        }
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
    int ret = RunReluGradV4(gradients_host, features_host, shape, config, &output_host, std::atoi(argv[6]));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    CHECK_RET(WriteFile(argv[5], output_host), return 4);
    return 0;
}
