/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <set>
#include <iostream>
#include <string>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#endif
#include "../../../op_kernel/log_softmax_grad.cpp"
#include "../../../op_kernel/log_softmax_grad_tiling_data.h"
#include <cstdint>

using namespace std;

// Mode scenarios (must match op_kernel/log_softmax_grad_tiling_key.h).
enum LogSoftmaxGradMode { kNoNeedReduce = 0, kReduceTail = 1, kReduceMid = 2 };

// Fill LogSoftmaxGradTilingData for a SINGLE-CORE (block_dim = 1) smoke run that
// drives each kernel branch (NO_NEED_REDUCE / REDUCE_TAIL / REDUCE_MID) over every
// element exactly once by processing the whole tensor in a single tile. This exercises
// the real kernel compute paths without re-implementing the host tiling math.
//
// Output correctness for all dtypes/shapes is covered by the op_api UT
// (tests/ut/op_api/test_aclnn_logsoftmax_backward.cpp, which runs a torch golden
// comparison); this kernel UT follows the repo's smoke-test convention, like
// examples/add_example/tests/ut/op_kernel/test_add_example.cpp.
void FillTiling(LogSoftmaxGradTilingData& td, const std::vector<int64_t>& shape, const std::vector<int64_t>& axis,
                LogSoftmaxGradMode& mode)
{
    int64_t dimNum = static_cast<int64_t>(shape.size());
    std::set<int64_t> axes;
    for (int64_t a : axis) {
        axes.insert(a >= 0 ? a : dimNum + a);
    }
    int64_t axisStart = *axes.begin();
    int64_t axisEnd = *axes.rbegin() + 1;

    uint64_t m0 = 1;
    uint64_t m1 = 1; // merged reduce-axis length
    uint64_t m2 = 1;
    for (int64_t i = 0; i < axisStart; ++i) {
        m0 *= static_cast<uint64_t>(shape[i]);
    }
    for (int64_t i = axisStart; i < axisEnd; ++i) {
        m1 *= static_cast<uint64_t>(shape[i]);
    }
    for (int64_t i = axisEnd; i < dimNum; ++i) {
        m2 *= static_cast<uint64_t>(shape[i]);
    }
    uint64_t total = m0 * m1 * m2;

    // Default all loop/tile fields to zero; only the relevant ones are set per mode.
    td = LogSoftmaxGradTilingData{0};
    td.mergedDim0 = 1;
    td.mergedDim1 = 1;
    td.mergedDim2 = 1;

    if (m1 == 1) {
        // NO_NEED_REDUCE: reduction is an identity (axis size 1).
        mode = kNoNeedReduce;
        td.singleBufElems = total;
        td.mergedDim0 = 1;
        td.mergedDim1 = 1;
        td.mergedDim2 = 1;
        td.totalElems = total;
    } else if (m2 == 1) {
        // REDUCE_TAIL: reduce axis is the last dimension. The kernel remaps
        // (pre-reduce -> mergedDim1, reduce length -> mergedDim2).
        mode = kReduceTail;
        td.mergedDim0 = 1;
        td.mergedDim1 = m0;
        td.mergedDim2 = m1;
        td.dim1Tile = m0;
        td.dim1LoopTime = 1;
        td.dim1Remained = 0;
        td.dim2Tile = m1;
        td.dim2LoopTime = 1;
        td.dim2Remained = 0;
        td.singleBufElems = m0 * m1; // == total, one full tile
    } else {
        // REDUCE_MID: reduce axis is in the middle (dim0Tile = 0 -> LoopDim0).
        mode = kReduceMid;
        td.mergedDim0 = m0;
        td.mergedDim1 = m1;
        td.mergedDim2 = m2;
        td.dim0Tile = 0;
        td.dim0LoopTime = 0;
        td.dim0Remained = 0;
        td.dim2Tile = m2;
        td.dim2LoopTime = 1;
        td.dim2Remained = 0;
        td.dim1Tile = m1;
        td.dim1LoopTime = 1;
        td.dim1Remained = 0;
        td.singleBufElems = m1 * m2; // one full tile
    }
}

class LogSoftmaxGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "LogSoftmaxGradKernelTest SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "LogSoftmaxGradKernelTest TearDown\n" << endl; }

    void RunCase(const std::vector<int64_t>& shape, const std::vector<int64_t>& axis)
    {
        LogSoftmaxGradTilingData td;
        LogSoftmaxGradMode mode = kNoNeedReduce;
        FillTiling(td, shape, axis, mode);

        uint64_t total = 1;
        for (int64_t s : shape) {
            total *= static_cast<uint64_t>(s);
        }
        ASSERT_TRUE(total > 0u);

        // Pad GM buffers to a block multiple so the kernel's block-aligned GM<->UB
        // DataCopy never reads/writes past the allocation for trailing chunks.
        constexpr uint64_t kBlockElems = 8; // FP32_ELEMS_PER_BLOCK (32B / 4B)
        uint64_t allocElems = ((total + kBlockElems - 1) / kBlockElems + 1) * kBlockElems;
        size_t byteSize = allocElems * sizeof(float);

        uint8_t* dy = (uint8_t*)AscendC::GmAlloc(byteSize); // input grad (dy)
        uint8_t* x = (uint8_t*)AscendC::GmAlloc(byteSize);  // forward log_softmax output (x)
        uint8_t* z = (uint8_t*)AscendC::GmAlloc(byteSize);  // output grad (z)
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(LogSoftmaxGradTilingData));

        // Deterministic, non-trivial inputs so the kernel actually exercises exp/reduce.
        float* dyF = reinterpret_cast<float*>(dy);
        float* xF = reinterpret_cast<float*>(x);
        for (uint64_t i = 0; i < total; ++i) {
            dyF[i] = static_cast<float>((static_cast<int64_t>(i) % 7) - 3) * 0.1f;
            xF[i] = static_cast<float>((static_cast<int64_t>(i) % 5)) * 0.2f;
        }
        std::memcpy(tiling, &td, sizeof(LogSoftmaxGradTilingData));

        auto LogSoftmaxGradKernel = [mode](GM_ADDR dy_, GM_ADDR x_, GM_ADDR z_, GM_ADDR workspace_, GM_ADDR tiling_) {
            if (mode == kNoNeedReduce) {
                ::log_softmax_grad<kNoNeedReduce, false, true>(dy_, x_, z_, workspace_, tiling_);
            } else if (mode == kReduceTail) {
                ::log_softmax_grad<kReduceTail, true, true>(dy_, x_, z_, workspace_, tiling_);
            } else { // kReduceMid
                ::log_softmax_grad<kReduceMid, true, true>(dy_, x_, z_, workspace_, tiling_);
            }
        };

        uint32_t blockDim = 1;
        ICPU_SET_TILING_KEY(static_cast<uint32_t>(mode));
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(LogSoftmaxGradKernel, blockDim, dy, x, z, workspace, tiling);

        AscendC::GmFree(dy);
        AscendC::GmFree(x);
        AscendC::GmFree(z);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

// NO_NEED_REDUCE: reduce axis has size 1 (reduction is an identity).
TEST_F(LogSoftmaxGradKernelTest, test_case_no_need_reduce_3d) { RunCase({3, 1, 7}, {-2}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_no_need_reduce_2d) { RunCase({5, 1}, {1}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_no_need_reduce_4d) { RunCase({2, 1, 1, 8}, {-3}); }

// REDUCE_TAIL: reduce axis is the last dimension.
TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_tail_small) { RunCase({4, 32}, {-1}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_tail_large) { RunCase({3, 1000}, {-1}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_tail_3d) { RunCase({8, 16, 1}, {-2}); }

// REDUCE_MID: reduce axis is in the middle.
TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_mid) { RunCase({2, 16, 8}, {-2}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_mid_small) { RunCase({3, 4, 8}, {1}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_mid_4d) { RunCase({2, 3, 4, 8}, {1}); }

TEST_F(LogSoftmaxGradKernelTest, test_case_reduce_mid_1d_mid) { RunCase({1, 8, 16}, {1}); }
