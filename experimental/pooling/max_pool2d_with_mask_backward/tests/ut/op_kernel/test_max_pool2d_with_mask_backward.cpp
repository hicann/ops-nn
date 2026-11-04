// clang-format off
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// clang-format on

#include <cstdint>
#include <cstring>
#include <iostream>

#include "gtest/gtest.h"

#include "../../../op_kernel/max_pool2d_with_mask_backward_tiling_key.h"
#include "kernel_operator.h"
#include "../../../op_kernel/max_pool2d_with_mask_backward_common.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void max_pool2d_with_mask_backward(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                                    GM_ADDR workspace, GM_ADDR tiling);

class MaxPool2dWithMaskBackwardKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "MaxPool2dWithMaskBackwardKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "MaxPool2dWithMaskBackwardKernelTest TearDown" << endl; }
};

namespace {
constexpr uint64_t kUbSize = 192UL * 1024UL;
// Field order of MaxPool2dWithMaskBackwardTilingData (42 x uint64_t):
// ncDim, diDim, hiDim, wiDim, doDim, hoDim, woDim, kd, kh, kw, sd, sh, sw,
// padDTop, padHTop, padWTop, padDBottom, padHBottom, padWBottom,
// baseNc, baseDo, baseHo, baseWo, ncTail, doTail, hoTail, woTail,
// ncCnt, doCnt, hoCnt, woCnt, totalCnt, usedCoreNum, totalUBSize,
// singleCoreNc, singleCoreDo, singleCoreHo, singleCoreWo,
// ncRound, ncRoundTail, totalRound, preCoreNum
constexpr size_t kTilingFields = 42;

// The Normal path uses DataCopyPad, which the CPU mock does not fully emulate. These cases
// therefore verify that the kernel can be launched for different tiling shapes and overlap
// modes without crashing; numeric precision is covered by the 140-case NPU self-test
// (tests/acceptance_910b) and the AscendOpTest runs.
void RunKernelCase(uint64_t tilingKey, uint64_t nc, uint64_t hi, uint64_t wi, uint64_t k, uint64_t s, uint64_t p,
                   bool emptyWorkspace = false)
{
    const uint64_t ho = (hi + 2 * p - k) / s + 1;
    const uint64_t wo = (wi + 2 * p - k) / s + 1;
    const uint64_t inElems = nc * hi * wi;
    const uint64_t outElems = nc * ho * wo;
    const uint64_t idxBytes = outElems * sizeof(int32_t) + 64;
    const uint64_t tilingBytes = kTilingFields * sizeof(uint64_t);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inElems * sizeof(float)));
    uint8_t* grad = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(outElems * sizeof(float)));
    uint8_t* argmax = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(idxBytes));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(inElems * sizeof(float)));
    uint8_t* workspace = emptyWorkspace ? nullptr : reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024 * 1024));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingBytes));

    ASSERT_NE(x, nullptr);
    ASSERT_NE(grad, nullptr);
    ASSERT_NE(argmax, nullptr);
    ASSERT_NE(y, nullptr);
    if (!emptyWorkspace) {
        ASSERT_NE(workspace, nullptr);
    }
    ASSERT_NE(tiling, nullptr);

    memset(x, 0, inElems * sizeof(float));
    memset(grad, 0, outElems * sizeof(float));
    memset(y, 0, inElems * sizeof(float));
    if (workspace != nullptr) {
        memset(workspace, 0, 1024 * 1024);
    }
    memset(tiling, 0, tilingBytes);
    memset(argmax, 0, idxBytes);

    auto xPtr = reinterpret_cast<float*>(x);
    auto gPtr = reinterpret_cast<float*>(grad);
    auto idxPtr = reinterpret_cast<int32_t*>(argmax);
    for (uint64_t i = 0; i < inElems; ++i) {
        xPtr[i] = static_cast<float>(i % 13) * 0.25f - 1.5f;
    }
    for (uint64_t c = 0; c < nc; ++c) {
        for (uint64_t oh = 0; oh < ho; ++oh) {
            for (uint64_t ow = 0; ow < wo; ++ow) {
                const uint64_t outIdx = (c * ho + oh) * wo + ow;
                gPtr[outIdx] = 1.0f + static_cast<float>(outIdx % 5);
                const uint64_t ih = oh * s;
                const uint64_t iw = ow * s;
                idxPtr[outIdx] = static_cast<int32_t>(ih * wi + iw);
            }
        }
    }

    auto td = reinterpret_cast<uint64_t*>(tiling);
    td[0] = nc;
    td[1] = 1;
    td[2] = hi;
    td[3] = wi;
    td[4] = 1;
    td[5] = ho;
    td[6] = wo;
    td[7] = 1;
    td[8] = k;
    td[9] = k;
    td[10] = 1;
    td[11] = s;
    td[12] = s;
    td[13] = 0;
    td[14] = p;
    td[15] = p;
    td[16] = 0;
    td[17] = p;
    td[18] = p;
    td[19] = nc;
    td[20] = 1;
    td[21] = ho;
    td[22] = wo;
    td[23] = nc;
    td[24] = 1;
    td[25] = ho;
    td[26] = wo;
    td[27] = 1;
    td[28] = 1;
    td[29] = 1;
    td[30] = 1;
    td[31] = 1;
    td[32] = 1;
    td[33] = kUbSize;
    td[34] = nc;
    td[35] = 1;
    td[36] = ho;
    td[37] = wo;
    td[38] = 0;
    td[39] = 0;
    td[40] = 0;
    td[41] = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(max_pool2d_with_mask_backward, 1, x, grad, argmax, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(grad);
    AscendC::GmFree(argmax);
    AscendC::GmFree(y);
    if (workspace != nullptr) {
        AscendC::GmFree(workspace);
    }
    AscendC::GmFree(tiling);
}
} // namespace

// Non-overlapping window (stride == kernel), Normal path.
TEST_F(MaxPool2dWithMaskBackwardKernelTest, test_fp32_non_overlap_k2s2)
{
    RunKernelCase(MaxPool2dWithMaskBackwardTilingKey::NORMAL, 1, 8, 8, 2, 2, 0);
}

// Overlapping window (stride < kernel), Normal overlap path.
TEST_F(MaxPool2dWithMaskBackwardKernelTest, test_fp32_overlap_k3s1)
{
    RunKernelCase(MaxPool2dWithMaskBackwardTilingKey::NORMAL_OVERLAP, 1, 8, 8, 3, 1, 1);
}

// Unaligned width to exercise tail handling.
TEST_F(MaxPool2dWithMaskBackwardKernelTest, test_fp32_unaligned_width)
{
    RunKernelCase(MaxPool2dWithMaskBackwardTilingKey::NORMAL, 1, 8, 7, 2, 2, 0);
}

// Complete coverage needs neither GM scratch nor an output-initialization barrier.
TEST_F(MaxPool2dWithMaskBackwardKernelTest, test_fp32_non_overlap_null_workspace)
{
    RunKernelCase(MaxPool2dWithMaskBackwardTilingKey::NORMAL, 1, 8, 8, 2, 2, 0, true);
}

TEST_F(MaxPool2dWithMaskBackwardKernelTest, test_workspace_coverage_policy)
{
    MaxPool2dWithMaskBackwardComm::TilingParams params{};
    params.diDim = params.doDim = params.kd = params.sd = 1;
    params.hiDim = params.wiDim = 8;
    params.hoDim = params.woDim = 4;
    params.kh = params.kw = params.sh = params.sw = 2;
    EXPECT_FALSE(MaxPool2dWithMaskBackwardComm::NeedsOutputInitialization(params, false));
    EXPECT_TRUE(MaxPool2dWithMaskBackwardComm::NeedsOutputInitialization(params, true));
    params.wiDim = 7;
    params.woDim = 3;
    EXPECT_TRUE(MaxPool2dWithMaskBackwardComm::NeedsOutputInitialization(params, false));
}
