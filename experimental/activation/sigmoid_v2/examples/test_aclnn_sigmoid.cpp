/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_aclnn_sigmoid.cpp
 * @brief ACLNN invocation example for experimental Sigmoid operator
 *
 * Demonstrates:
 *   1. Building input/output aclTensor objects
 *   2. Calling aclnnSigmoidGetWorkspaceSize to prepare executor/workspace
 *   3. Calling aclnnSigmoid to launch the custom Sigmoid operator
 *
 * Build and run:
 *   cd examples && bash run.sh
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
#include "aclnnop/aclnn_sigmoid.h"

#define CHECK_RET(cond, expr) \
    do { \
        if (!(cond)) { \
            expr; \
        } \
    } while (0)

namespace {
struct SigmoidConfig {
    aclDataType acl_dtype;
    size_t element_size;
    std::string name;
};

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

bool ParseDtype(const std::string& dtype_name, SigmoidConfig* config)
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

aclError CreateAclTensor(
    const std::vector<int64_t>& shape, aclDataType dtype, void* device_addr, aclTensor** tensor)
{
    std::vector<int64_t> strides = MakeStrides(shape);
    const int64_t* shape_ptr = shape.empty() ? nullptr : shape.data();
    const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();
    *tensor = aclCreateTensor(
        shape_ptr, shape.size(), dtype, strides_ptr, 0, ACL_FORMAT_ND, shape_ptr, shape.size(), device_addr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int RunSigmoid(
    const std::vector<char>& input_host, const std::vector<int64_t>& shape, const SigmoidConfig& config,
    std::vector<char>* output_host, int32_t device_id)
{
    aclrtStream stream = nullptr;
    void* input_device = nullptr;
    void* output_device = nullptr;
    void* workspace = nullptr;
    aclTensor* input_tensor = nullptr;
    aclTensor* output_tensor = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    const size_t bytes = static_cast<size_t>(GetShapeSize(shape)) * config.element_size;
    bool acl_initialized = false;
    bool device_set = false;

    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    acl_initialized = true;
    ret = aclrtSetDevice(device_id);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    device_set = true;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    if (bytes > 0) {
        ret = aclrtMalloc(&input_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
        ret = aclrtMalloc(&output_device, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

        ret = aclrtMemcpy(input_device, bytes, input_host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
        std::vector<char> zero_buffer(bytes, 0);
        ret = aclrtMemcpy(output_device, bytes, zero_buffer.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = CreateAclTensor(shape, config.acl_dtype, input_device, &input_tensor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = CreateAclTensor(shape, config.acl_dtype, output_device, &output_tensor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    ret = aclnnSigmoidGetWorkspaceSize(input_tensor, output_tensor, &workspace_size, &executor);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

    ret = aclnnSigmoid(workspace, workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, goto cleanup);

    output_host->resize(bytes);
    if (bytes > 0) {
        ret = aclrtMemcpy(output_host->data(), bytes, output_device, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, goto cleanup);
    }

cleanup:
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
    return ret;
}

int RunSmokeCase()
{
    const std::vector<float> input_values = {0.0f, 1.0f, 2.0f, 3.0f};
    std::vector<char> input_host(sizeof(float) * input_values.size());
    std::memcpy(input_host.data(), input_values.data(), input_host.size());

    std::vector<char> output_host;
    SigmoidConfig config {ACL_FLOAT, sizeof(float), "fp32"};
    int ret = RunSigmoid(input_host, {2, 2}, config, &output_host, 0);
    CHECK_RET(ret == ACL_SUCCESS, std::fprintf(stderr, "run sigmoid failed: %d\n", ret); return ret);

    const float* output_values = reinterpret_cast<const float*>(output_host.data());
    bool pass = true;
    for (size_t i = 0; i < input_values.size(); ++i) {
        float expected = 1.0f / (1.0f + std::exp(-input_values[i]));
        float diff = std::fabs(output_values[i] - expected);
        std::printf("result[%zu] = %.6f expected = %.6f\n", i, output_values[i], expected);
        if (diff > 1e-6f) {
            pass = false;
        }
    }
    std::printf("test %s\n", pass ? "pass" : "failed");
    return pass ? 0 : 1;
}

void PrintUsage(const char* program)
{
    std::fprintf(
        stderr,
        "Usage: %s [dtype shape_csv input_bin output_bin [device_id]]\n"
        "Example: %s fp16 2,3 /tmp/in.bin /tmp/out.bin 0\n",
        program,
        program);
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc == 1) {
        return RunSmokeCase();
    }
    if (argc < 5 || argc > 6) {
        PrintUsage(argv[0]);
        return 1;
    }

    SigmoidConfig config {};
    CHECK_RET(ParseDtype(argv[1], &config), PrintUsage(argv[0]); return 1);

    std::vector<int64_t> shape;
    CHECK_RET(ParseShape(argv[2], &shape), PrintUsage(argv[0]); return 1);

    std::vector<char> input_host;
    CHECK_RET(ReadFile(argv[3], &input_host), std::fprintf(stderr, "failed to read input file\n"); return 1);

    const size_t expected_bytes = static_cast<size_t>(GetShapeSize(shape)) * config.element_size;
    CHECK_RET(
        input_host.size() == expected_bytes,
        std::fprintf(stderr, "input size mismatch: got %zu expected %zu\n", input_host.size(), expected_bytes);
        return 1);

    int32_t device_id = argc == 6 ? std::atoi(argv[5]) : 0;
    std::vector<char> output_host;
    int ret = RunSigmoid(input_host, shape, config, &output_host, device_id);
    CHECK_RET(ret == ACL_SUCCESS, std::fprintf(stderr, "run sigmoid failed: %d\n", ret); return ret);

    CHECK_RET(WriteFile(argv[4], output_host), std::fprintf(stderr, "failed to write output file\n"); return 1);
    return 0;
}
