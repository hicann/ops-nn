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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * @file test_aclnn_selu.cpp
 * @brief Selu 算子调用示例
 *
 * 演示如何通过 aclnn 接口在 NPU 上调用 Selu 算子。
 *
 * 公式: SELU(x) = scale * [max(0, x) + min(0, alpha * (exp(x) - 1))]
 *   alpha = 1.6732632423543772848170429916717
 *   scale = 1.0507009873554804934193349852946
 *
 * 编译方法:
 *   1. 先编译并安装算子包:
 *      cd ops/selu && bash build.sh --pkg --soc=ascend910b
 *      cd build_out && ./custom_opp_ubuntu_aarch64.run
 *      source $ASCEND_OPP_PATH/vendors/selu_custom/bin/set_env.bash
 *
 *   2. 编译本示例:
 *      VENDOR_PATH=$ASCEND_OPP_PATH/vendors/selu_custom
 *      g++ -o test_aclnn_selu test_aclnn_selu.cpp \
 *          -I${ASCEND_HOME_PATH}/include \
 *          -I${VENDOR_PATH}/op_api/include \
 *          -L${ASCEND_HOME_PATH}/lib64 \
 *          -L${VENDOR_PATH}/op_api/lib \
 *          -lascendcl -lnnopbase -lcust_opapi -lacl_op_compiler \
 *          -Wl,-rpath,${VENDOR_PATH}/op_api/lib
 *
 * 运行方法:
 *   ./test_aclnn_selu
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>

#include "acl/acl.h"
#include "aclnn_selu.h"

#define CHECK_ACL(expr)                                                     \
    do {                                                                    \
        auto _ret = (expr);                                                 \
        if (_ret != ACL_SUCCESS) {                                          \
            std::cerr << "ACL Error: " << #expr << " returned " << _ret    \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

// SELU 固定常数
constexpr float ALPHA = 1.6732632423543772848170429916717f;
constexpr float SCALE = 1.0507009873554804934193349852946f;

// CPU 参考实现
static float seluRef(float x)
{
    if (x > 0.0f) {
        return SCALE * x;
    }
    return SCALE * ALPHA * (expf(x) - 1.0f);
}

int main()
{
    // ========================================================================
    // 1. 参数设置（2D shape: 3x5 = 15 个元素）
    // ========================================================================
    constexpr int64_t ROWS = 3;
    constexpr int64_t COLS = 5;
    constexpr int64_t ELEM_COUNT = ROWS * COLS;
    const int64_t shape[] = {ROWS, COLS};
    const int64_t strides[] = {COLS, 1};
    constexpr int64_t ndim = 2;

    // 输入数据: 覆盖大负值、小负值、零、小正值、大正值
    float hostInput[ELEM_COUNT] = {
        -5.0f,  -2.5f,  -1.0f,  -0.1f,  -0.001f,   // 负值区域（指数分支）
         0.0f,   0.001f, 0.1f,   0.5f,    1.0f,      // 零及小正值（线性分支）
         2.5f,   5.0f,  10.0f, 100.0f,   -10.0f      // 大正值 + 极端负值
    };

    // ========================================================================
    // 2. ACL 初始化
    // ========================================================================
    int ret = 1;
    aclrtStream stream = nullptr;
    void *devInput = nullptr;
    void *devOutput = nullptr;
    void *workspace = nullptr;
    aclTensor *selfTensor = nullptr;
    aclTensor *outTensor = nullptr;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateStream(&stream));

    // ========================================================================
    // 3. 设备内存分配 & 数据拷贝
    // ========================================================================
    {
        size_t dataBytes = ELEM_COUNT * sizeof(float);

        CHECK_ACL(aclrtMalloc(&devInput, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(&devOutput, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMemset(devOutput, dataBytes, 0, dataBytes));
        CHECK_ACL(aclrtMemcpy(devInput, dataBytes, hostInput, dataBytes, ACL_MEMCPY_HOST_TO_DEVICE));

        // ====================================================================
        // 4. 创建 aclTensor
        // ====================================================================
        selfTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                     ACL_FORMAT_ND, shape, ndim, devInput);
        outTensor = aclCreateTensor(shape, ndim, ACL_FLOAT, strides, 0,
                                    ACL_FORMAT_ND, shape, ndim, devOutput);

        // ====================================================================
        // 5. 调用 aclnnSelu（无 alpha 属性参数，仅输入和输出）
        // ====================================================================
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;

        CHECK_ACL(aclnnSeluGetWorkspaceSize(selfTensor, outTensor,
                                              &workspaceSize, &executor));
        if (workspaceSize > 0) {
            CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
        }

        CHECK_ACL(aclnnSelu(workspace, workspaceSize, executor, stream));
        CHECK_ACL(aclrtSynchronizeStream(stream));

        // ====================================================================
        // 6. 读取输出 & 精度验证
        // ====================================================================
        float hostOutput[ELEM_COUNT] = {};
        CHECK_ACL(aclrtMemcpy(hostOutput, dataBytes, devOutput, dataBytes, ACL_MEMCPY_DEVICE_TO_HOST));

        std::cout << "Selu Example (shape: [3, 5], dtype: float32)" << std::endl;
        std::cout << "  SELU(x) = scale * [max(0,x) + min(0, alpha*(exp(x)-1))]" << std::endl;
        std::cout << "  alpha = 1.6732632..., scale = 1.0507009..." << std::endl;
        std::cout << "---------------------------------------------------------------" << std::endl;
        printf("  %4s | %11s | %11s | %11s | %9s\n", "Idx", "Input", "NPU Output", "Expected", "Diff");
        std::cout << "---------------------------------------------------------------" << std::endl;

        int passCount = 0;
        constexpr float rtol = 1e-4f;
        constexpr float atol = 1e-4f;

        for (int i = 0; i < ELEM_COUNT; ++i) {
            float x = hostInput[i];
            float expected = seluRef(x);
            float diff = std::fabs(hostOutput[i] - expected);
            float threshold = atol + rtol * std::fabs(expected);
            bool pass = (diff <= threshold);
            passCount += pass ? 1 : 0;

            printf("  [%d,%d] | %11.5f | %11.5f | %11.5f | %9.2e %s\n",
                   (int)(i / COLS), (int)(i % COLS), x, hostOutput[i], expected, diff,
                   pass ? "PASS" : "FAIL");
        }

        std::cout << "---------------------------------------------------------------" << std::endl;
        std::cout << "Result: " << passCount << "/" << ELEM_COUNT << " passed";
        if (passCount == ELEM_COUNT) {
            std::cout << " -- ALL PASS" << std::endl;
            ret = 0;
        } else {
            std::cout << " -- FAILED" << std::endl;
            ret = 1;
        }
    }

    // ========================================================================
    // 7. 资源释放
    // ========================================================================
cleanup:
    if (selfTensor) aclDestroyTensor(selfTensor);
    if (outTensor) aclDestroyTensor(outTensor);
    if (workspace) aclrtFree(workspace);
    if (devInput) aclrtFree(devInput);
    if (devOutput) aclrtFree(devOutput);
    if (stream) aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return ret;
}
