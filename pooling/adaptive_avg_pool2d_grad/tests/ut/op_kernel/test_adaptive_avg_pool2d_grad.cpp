/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_adaptive_avg_pool2d_grad.cpp
 * \brief kernel UT for adaptive_avg_pool2d_grad on ascend950
 *
 * Tiling selection logic (from op_host IsCapable):
 *   BIG_KERNEL: kernelH * kernelW >= 256 (high upsampling ratio)
 *   SIMT:       kernelH * kernelW <  256 (fallback, always capable)
 * where kernelH = ceil(outputShape.H / inputShape.H).
 *
 * The kernel UT directly instantiates TPL_BIG_KERNEL / TPL_SIMT_KERNEL templates,
 * bypassing the runtime tiling selection. Tiling data is pre-computed in gen_tiling.py.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <numeric>
#include <unistd.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "../../../op_kernel/arch35/adaptive_avg_pool2d_grad.cpp"
#include "data_utils.h"
#endif
#include "gtest/gtest.h"
#include "../../../op_kernel/arch35/adaptive_avg_pool2d_grad_struct.h"

using namespace std;
using namespace AdaptiveAvgPool2dGradOp;

inline int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    return accumulate(shape.begin(), shape.end(), 1LL, multiplies<int64_t>());
}

class AdaptiveAvgPool2dGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "AdaptiveAvgPool2dGradKernelTest SetUp\n"; }
    static void TearDownTestCase() { cout << "AdaptiveAvgPool2dGradKernelTest TearDown\n"; }
};

template <typename T, uint64_t TEMPLATE_MODE>
void ExcuteTestCase(const std::vector<int64_t>& inputShape, const std::vector<int64_t>& outputShape,
                    const std::string& inputType, const std::string& caseName, uint64_t tilingKey,
                    size_t tilingDataSize, uint32_t blockDim)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t elemSize = (inputType == "float32" ? 4 : 2);
    int64_t inputGradShapeSize = GetShapeSize(inputShape);
    int64_t outputGradShapeSize = GetShapeSize(outputShape);
    size_t inputByteSize = inputGradShapeSize * elemSize;
    size_t outputByteSize = outputGradShapeSize * elemSize;
    size_t workspaceSize = outputByteSize;
    size_t tilingSize = tilingDataSize;

    // BIG_KERNEL CopyIn uses MultiCopy with plane stride hInput*wInput which may
    // exceed go buffer. Allocate outputByteSize for both GM buffers for safety.
    size_t gmAlignSz = (outputByteSize + 31) / 32 * 32;
    uint8_t* inputGrad = (uint8_t*)AscendC::GmAlloc(gmAlignSz);
    uint8_t* outputGrad = (uint8_t*)AscendC::GmAlloc(gmAlignSz);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset_s(inputGrad, gmAlignSz, 0, gmAlignSz);
    memset_s(outputGrad, gmAlignSz, 0, gmAlignSz);
    memset_s(workspace, workspaceSize, 0, workspaceSize);

    std::string path = std::string(get_current_dir_name()) + "/adaptive_avg_pool2d_grad_data/";
    system(("rm -rf " + path + "*.bin").c_str());

    system("cp -r ../../../../pooling/adaptive_avg_pool2d_grad/tests/ut/op_kernel/adaptive_avg_pool2d_grad_data ./");
    system("chmod -R 755 ./adaptive_avg_pool2d_grad_data/");

    std::string cmd = "cd ./adaptive_avg_pool2d_grad_data/ && python3 gen_data.py " + caseName + " " + inputType;
    system(cmd.c_str());
    cmd = "cd ./adaptive_avg_pool2d_grad_data/ && python3 gen_tiling.py " + caseName;
    system(cmd.c_str());

    ReadFile(path + "input_grad.bin", inputByteSize, inputGrad, inputByteSize);
    ReadFile(path + "tiling.bin", tilingSize, tiling, tilingSize);

    ICPU_SET_TILING_KEY(tilingKey);
    auto func = static_cast<void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)>(
        adaptive_avg_pool2d_grad<TEMPLATE_MODE, TPL_INT64, 0>);
    ICPU_RUN_KF(func, blockDim, inputGrad, outputGrad, workspace, tiling);

    WriteFile(path + "cce_cpu_x_grad.bin", outputGrad, outputByteSize);
    cmd = "cmp " + path + "cce_cpu_x_grad.bin " + path + "x_grad_golden.bin";
    (void)system(cmd.c_str());

    AscendC::GmFree(inputGrad);
    AscendC::GmFree(outputGrad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// ============ SIMT tests (TPL_SIMT_KERNEL template) ============
// kernelH=2, kernelW=2, product=4 < 256 → SIMT fallback path

TEST_F(AdaptiveAvgPool2dGradKernelTest, test_simt_fp32_small)
{
    ExcuteTestCase<float, TPL_SIMT_KERNEL>({1, 2, 4, 4}, {1, 2, 8, 8}, "float32", "test_simt_fp32_small", 200,
                                           sizeof(AdaptiveAvgPool2dGradSimtTiling), 1);
}

TEST_F(AdaptiveAvgPool2dGradKernelTest, test_simt_fp32_medium)
{
    ExcuteTestCase<float, TPL_SIMT_KERNEL>({1, 2, 2, 2}, {1, 2, 8, 8}, "float32", "test_simt_fp32_medium", 210,
                                           sizeof(AdaptiveAvgPool2dGradSimtTiling), 1);
}
