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
 * @file test_geir_softsign.cpp
 * @brief GE IR (single-op) invocation example for Softsign operator
 *
 * This example demonstrates the GE IR style invocation of a custom op
 * via the ACL aclopExecuteV2 single-op execution interface. This is the
 * standard GE-IR-based dispatch path for custom operators registered with
 * REG_OP(Softsign) in op_graph/softsign_proto.h.
 *
 * Flow:
 *   1. Construct aclTensorDesc inputs/outputs that match the IR definition
 *      (input "x", output "y" with FloatingDataType + DT_BF16 supported).
 *   2. Allocate device memory and aclDataBuffer wrappers.
 *   3. Call aclopExecuteV2 with the IR op type "Softsign".
 *   4. Synchronize and verify against CPU reference.
 *
 * Build and run:
 *   cd examples && bash run.sh
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_mdl.h"
#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"
#include "acl/acl_rt.h"

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        auto __ret = (expr);                                                                            \
        int32_t __code = static_cast<int32_t>(__ret);                                                   \
        if (__code != 0) {                                                                              \
            fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, __code); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

/**
 * @brief CPU reference: softsign(x) = x / (1 + |x|)
 */
static float SoftsignRef(float x)
{
    return x / (1.0f + std::fabs(x));
}

int32_t main(int32_t argc, char** argv)
{
    // =========================================================================
    // 1. Initialize ACL environment
    // =========================================================================
    const int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    // =========================================================================
    // 2. Prepare input data and tensor descriptors
    //    The IR signature in op_graph/softsign_proto.h is:
    //      REG_OP(Softsign)
    //          .INPUT(x, TensorType::ALL())
    //          .OUTPUT(y, TensorType::ALL())
    //          .OP_END_FACTORY_REG(Softsign)
    // =========================================================================
    const char* opType = "Softsign";

    const std::vector<float> inputData = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    const std::vector<int64_t> shape = {static_cast<int64_t>(inputData.size())};
    const int64_t elementCount = static_cast<int64_t>(inputData.size());
    const size_t bufferSize = elementCount * sizeof(float);

    const aclDataType dataType = ACL_FLOAT;
    const aclFormat format = ACL_FORMAT_ND;

    // Tensor descriptors describing the IR input "x" and output "y"
    auto* inputDesc = aclCreateTensorDesc(dataType, shape.size(), shape.data(), format);
    auto* outputDesc = aclCreateTensorDesc(dataType, shape.size(), shape.data(), format);

    std::vector<aclTensorDesc*> inputDescs = {inputDesc};
    std::vector<aclTensorDesc*> outputDescs = {outputDesc};

    // =========================================================================
    // 3. Allocate device memory and create data buffers
    // =========================================================================
    void* inputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&inputDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto* inputBuffer = aclCreateDataBuffer(inputDeviceMem, bufferSize);

    void* outputDeviceMem = nullptr;
    CHECK_ACL(aclrtMalloc(&outputDeviceMem, bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
    auto* outputBuffer = aclCreateDataBuffer(outputDeviceMem, bufferSize);

    std::vector<aclDataBuffer*> inputBuffers = {inputBuffer};
    std::vector<aclDataBuffer*> outputBuffers = {outputBuffer};

    // Softsign has no IR attributes
    aclopAttr* opAttr = aclopCreateAttr();

    // =========================================================================
    // 4. Copy input to device and execute via single-op IR dispatch
    // =========================================================================
    CHECK_ACL(aclrtMemcpy(inputDeviceMem, bufferSize, inputData.data(),
                          bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Compile + execute via OPP custom op package lookup.
    // ACL_ENGINE_SYS lets ACL choose the engine; ACL_COMPILE_SYS uses the
    // installed system OPP path (custom op package installed under
    // $ASCEND_OPP_PATH/vendors/softsign_custom is automatically discovered).
    CHECK_ACL(aclopCompileAndExecute(opType,
                                     inputDescs.size(), inputDescs.data(), inputBuffers.data(),
                                     outputDescs.size(), outputDescs.data(), outputBuffers.data(),
                                     opAttr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, nullptr,
                                     stream));

    CHECK_ACL(aclrtSynchronizeStream(stream));

    // =========================================================================
    // 5. Copy result back to host
    // =========================================================================
    std::vector<float> outputData(elementCount, 0.0f);
    CHECK_ACL(aclrtMemcpy(outputData.data(), bufferSize, outputDeviceMem,
                          bufferSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // =========================================================================
    // 6. Verify result against CPU reference
    // =========================================================================
    printf("GE IR Softsign([");
    for (int64_t i = 0; i < elementCount; i++) {
        printf("%.1f%s", inputData[i], (i < elementCount - 1) ? ", " : "");
    }
    printf("]) =\n               [");
    for (int64_t i = 0; i < elementCount; i++) {
        printf("%.4f%s", outputData[i], (i < elementCount - 1) ? ", " : "");
    }
    printf("]\n");

    bool pass = true;
    const float tolerance = 1e-3f;
    for (int64_t i = 0; i < elementCount; i++) {
        float expected = SoftsignRef(inputData[i]);
        float diff = std::fabs(outputData[i] - expected);
        if (diff > tolerance) {
            fprintf(stderr, "[FAIL] index=%ld, output=%.6f, expected=%.6f, diff=%.6f\n",
                    i, outputData[i], expected, diff);
            pass = false;
        }
    }
    printf("test %s\n", pass ? "pass" : "failed");

    // =========================================================================
    // 7. Cleanup
    // =========================================================================
    CHECK_ACL(aclDestroyDataBuffer(inputBuffer));
    CHECK_ACL(aclDestroyDataBuffer(outputBuffer));
    CHECK_ACL(aclrtFree(inputDeviceMem));
    CHECK_ACL(aclrtFree(outputDeviceMem));
    aclDestroyTensorDesc(inputDesc);
    aclDestroyTensorDesc(outputDesc);
    aclopDestroyAttr(opAttr);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());

    return pass ? 0 : 1;
}
