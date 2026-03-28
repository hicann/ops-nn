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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/**
 * @file test_aclnn_foreach_div_scalar.cpp
 * @brief ForeachDivScalar 算子调用示例
 *
 * 功能：对输入 TensorList 中的每个 tensor 执行除以标量操作
 * 公式：y_i = x_i / scalar  (i = 0, 1, ..., n-1)
 *
 * 编译：
 *   export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0
 *   g++ -std=c++17 -o example example.cpp \
 *       -I${ASCEND_HOME_PATH}/include \
 *       -I${ASCEND_HOME_PATH}/opp/vendors/custom_nn/op_api/include \
 *       -L${ASCEND_HOME_PATH}/lib64 \
 *       -L${ASCEND_HOME_PATH}/opp/vendors/custom_nn/op_api/lib \
 *       -lascendcl -lopapi -lnnopbase -lcust_opapi
 *
 * 运行：
 *   export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/opp/vendors/custom_nn/op_api/lib:$LD_LIBRARY_PATH
 *   ./example
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "aclnn_foreach_div_scalar.h"

#define CHECK_ACL(expr)                                                            \
    do {                                                                           \
        auto _ret = (expr);                                                        \
        if (_ret != 0) {                                                           \
            printf("[ERROR] %s failed, ret=%d, line=%d\n", #expr, _ret, __LINE__); \
            return -1;                                                             \
        }                                                                          \
    } while (0)

// Compute row-major strides from shape
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

int main() {
    // ===================== 1. 初始化 ACL =====================
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    printf("=== ForeachDivScalar Example ===\n");
    printf("Formula: y_i = x_i / scalar\n\n");

    // ===================== 2. 准备参数 =====================
    const int tensorNum = 3;
    const std::vector<int64_t> shape = {2, 4};  // 每个 tensor 的 shape
    const int64_t numElements = 8;               // 2 * 4
    const float scalarValue = -2.0f;

    printf("Parameters:\n");
    printf("  tensorNum = %d\n", tensorNum);
    printf("  shape     = [2, 4] (8 elements per tensor)\n");
    printf("  scalar    = %.1f\n", scalarValue);
    printf("  dtype     = FLOAT32\n\n");

    // ===================== 3. 生成输入数据 =====================
    std::vector<std::vector<float>> hostInputs(tensorNum);
    for (int t = 0; t < tensorNum; t++) {
        hostInputs[t].resize(numElements);
        std::mt19937 gen(42 + t);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        for (int j = 0; j < numElements; j++) {
            hostInputs[t][j] = dist(gen);
        }
    }

    printf("Input tensors:\n");
    for (int t = 0; t < tensorNum; t++) {
        printf("  x%d = [", t);
        for (int j = 0; j < numElements; j++) {
            printf("%.4f%s", hostInputs[t][j], j < numElements - 1 ? ", " : "");
        }
        printf("]\n");
    }
    printf("\n");

    // ===================== 4. 分配连续 Device 内存 =====================
    // 所有 tensor 共享一块连续内存（与算子内部预期一致）
    size_t bytesPerTensor = numElements * sizeof(float);
    size_t totalBytes = tensorNum * bytesPerTensor;

    void* xDev = nullptr;
    void* yDev = nullptr;
    CHECK_ACL(aclrtMalloc(&xDev, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&yDev, totalBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(yDev, totalBytes, 0, totalBytes));

    // 拷贝输入数据到 Device（按偏移写入连续内存）
    for (int t = 0; t < tensorNum; t++) {
        size_t offset = t * bytesPerTensor;
        CHECK_ACL(aclrtMemcpy(static_cast<char*>(xDev) + offset, bytesPerTensor,
                               hostInputs[t].data(), bytesPerTensor,
                               ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ===================== 5. 创建 Tensor 和 TensorList =====================
    auto strides = ComputeStrides(shape);

    // 输入 TensorList
    std::vector<aclTensor*> xTensors(tensorNum);
    for (int t = 0; t < tensorNum; t++) {
        void* ptr = static_cast<char*>(xDev) + t * bytesPerTensor;
        xTensors[t] = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                       strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), ptr);
    }
    aclTensorList* xList = aclCreateTensorList(xTensors.data(), tensorNum);

    // 输出 TensorList
    std::vector<aclTensor*> yTensors(tensorNum);
    for (int t = 0; t < tensorNum; t++) {
        void* ptr = static_cast<char*>(yDev) + t * bytesPerTensor;
        yTensors[t] = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT,
                                       strides.data(), 0, ACL_FORMAT_ND,
                                       shape.data(), shape.size(), ptr);
    }
    aclTensorList* yList = aclCreateTensorList(yTensors.data(), tensorNum);

    // 标量 Tensor（shape={1} 的一维 tensor）
    void* scalarDev = nullptr;
    CHECK_ACL(aclrtMalloc(&scalarDev, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(scalarDev, sizeof(float), &scalarValue, sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> scalarShape = {1};
    auto scalarStrides = ComputeStrides(scalarShape);
    aclTensor* scalarTensor = aclCreateTensor(scalarShape.data(), 1, ACL_FLOAT,
                                               scalarStrides.data(), 0, ACL_FORMAT_ND,
                                               scalarShape.data(), 1, scalarDev);

    // ===================== 6. 调用算子（两阶段接口） =====================
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 阶段一：GetWorkspaceSize - 计算所需 workspace 大小
    CHECK_ACL(aclnnForeachDivScalarGetWorkspaceSize(xList, scalarTensor, yList,
                                                     &workspaceSize, &executor));
    printf("Workspace size: %lu bytes\n", workspaceSize);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 阶段二：Execute - 执行算子计算
    CHECK_ACL(aclnnForeachDivScalar(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    printf("Operator execution completed.\n\n");

    // ===================== 7. 获取并展示输出结果 =====================
    printf("Output tensors:\n");
    std::vector<std::vector<float>> hostOutputs(tensorNum, std::vector<float>(numElements));
    for (int t = 0; t < tensorNum; t++) {
        size_t offset = t * bytesPerTensor;
        CHECK_ACL(aclrtMemcpy(hostOutputs[t].data(), bytesPerTensor,
                               static_cast<char*>(yDev) + offset, bytesPerTensor,
                               ACL_MEMCPY_DEVICE_TO_HOST));
        printf("  y%d = [", t);
        for (int j = 0; j < numElements; j++) {
            printf("%.4f%s", hostOutputs[t][j], j < numElements - 1 ? ", " : "");
        }
        printf("]\n");
    }

    // ===================== 8. 精度验证 =====================
    printf("\nVerification (rtol=1e-4, atol=1e-4):\n");
    bool allPass = true;
    int totalMismatch = 0;

    for (int t = 0; t < tensorNum; t++) {
        int mismatch = 0;
        for (int j = 0; j < numElements; j++) {
            float expected = hostInputs[t][j] / scalarValue;
            float actual = hostOutputs[t][j];
            float diff = std::abs(expected - actual);
            float tol = 1e-4f + 1e-4f * std::abs(expected);
            if (diff > tol) {
                if (mismatch < 3) {
                    printf("  y%d[%d]: expected=%.6f, actual=%.6f, diff=%.6e\n",
                           t, j, expected, actual, diff);
                }
                mismatch++;
            }
        }
        if (mismatch == 0) {
            printf("  y%d: PASS (all %ld elements match)\n", t, numElements);
        } else {
            printf("  y%d: FAIL (%d/%ld mismatches)\n", t, mismatch, numElements);
            allPass = false;
        }
        totalMismatch += mismatch;
    }
    printf("\nOverall: %s\n", allPass ? "PASS" : "FAIL");

    // ===================== 9. 释放资源 =====================
    for (int t = 0; t < tensorNum; t++) {
        aclDestroyTensor(xTensors[t]);
        aclDestroyTensor(yTensors[t]);
    }
    aclDestroyTensor(scalarTensor);
    aclDestroyTensorList(xList);
    aclDestroyTensorList(yList);
    aclrtFree(xDev);
    aclrtFree(yDev);
    aclrtFree(scalarDev);
    if (workspace) aclrtFree(workspace);

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    printf("\n=== Example Done ===\n");
    return allPass ? 0 : -1;
}
