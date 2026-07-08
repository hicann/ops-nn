/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>

#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/glu_grad_tiling.h"
#include "../../../op_kernel/glu_grad.cpp"

using namespace std;

class GluGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "GluGradKernelTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "GluGradKernelTest TearDown" << endl;
    }
};

namespace {
void RunKernel(size_t gradOutByteSize, size_t selfByteSize, size_t outByteSize, uint32_t blockDim,
    int64_t tilingKey, const optiling::GluGradTilingData& tilingData,
    void (*kernelFunc)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR))
{
    auto* gradOut = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradOutByteSize));
    auto* self = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(selfByteSize));
    auto* out = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outByteSize));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(optiling::GluGradTilingData)));

    auto* tilingDataBuffer = reinterpret_cast<optiling::GluGradTilingData*>(tiling);
    *tilingDataBuffer = tilingData;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernelFunc, blockDim, gradOut, self, out, workspace, reinterpret_cast<uint8_t*>(tilingDataBuffer));

    AscendC::GmFree(gradOut);
    AscendC::GmFree(self);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

optiling::GluGradTilingData MakeSmallShapeTiling()
{
    optiling::GluGradTilingData tilingData;
    tilingData.set_group( 6);
    tilingData.set_loopNum( 0);
    tilingData.set_tailLoopNum( 0);
    tilingData.set_nLastTailGroup( 2);
    tilingData.set_lastTailGroup( 0);
    tilingData.set_splitSize( 1280);
    tilingData.set_realCoreNum( 40);
    tilingData.set_numPerCore( 2);
    tilingData.set_blockSize( 8);
    tilingData.set_ny( 1280);
    return tilingData;
}

optiling::GluGradTilingData MakeBigShapeTiling()
{
    optiling::GluGradTilingData tilingData;
    tilingData.set_group( 1);
    tilingData.set_loopNum( 2);
    tilingData.set_tailLoopNum( 512);
    tilingData.set_nLastTailGroup( 0);
    tilingData.set_lastTailGroup( 0);
    tilingData.set_splitSize( 512);
    tilingData.set_realCoreNum( 40);
    tilingData.set_numPerCore( 0);
    tilingData.set_blockSize( 8);
    tilingData.set_ny( 1024);
    return tilingData;
}
} // namespace

TEST_F(GluGradKernelTest, run_small_shape_kernel)
{
    auto tilingData = MakeSmallShapeTiling();
    auto KernelSmallShape = [](GM_ADDR grad_out, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
        ::glu_grad<float, TPL_SMALL_SHAPE>(grad_out, self, out, workspace, tiling);
    };
    RunKernel(1 * 80 * 1280 * sizeof(float), 1 * 80 * 2560 * sizeof(float),
        1 * 80 * 2560 * sizeof(float), 40, 1, tilingData, KernelSmallShape);
}

TEST_F(GluGradKernelTest, run_big_shape_kernel)
{
    auto tilingData = MakeBigShapeTiling();
    auto KernelBigShape = [](GM_ADDR grad_out, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
        ::glu_grad<float, TPL_BIG_SHAPE>(grad_out, self, out, workspace, tiling);
    };
    RunKernel(1 * 1 * 1024 * sizeof(float), 1 * 2 * 1024 * sizeof(float),
        1 * 2 * 1024 * sizeof(float), 40, 2, tilingData, KernelBigShape);
}
