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
#include <string>
#include "gtest/gtest.h"
#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#endif
#include "../../../op_kernel/hinge_loss.cpp"
#include "../../../op_kernel/hinge_loss_tiling_data.h"

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : data_(static_cast<uint8_t*>(AscendC::GmAlloc(bytes))) {}

    ~GmBuffer()
    {
        if (data_ != nullptr) {
            AscendC::GmFree(data_);
        }
    }

    uint8_t* Get() const { return data_; }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

private:
    uint8_t* data_ = nullptr;
};

class HingeLossTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment("experimental/loss/hinge_loss/tests/ut/op_kernel/hinge_loss_data",
                                        "hinge_loss_data");
    }

    template <uint32_t schMode>
    static void RunKernelCase(const std::string& shape, const std::string& dtype, uint32_t count, size_t elementSize,
                              uint32_t blockDim, const HingeLossTilingData& tilingData)
    {
        ASSERT_TRUE(kernel_ut::RunGenData("./hinge_loss_data", {shape, dtype}));
        const size_t bytes = count * elementSize;
        GmBuffer predict(bytes);
        GmBuffer target(bytes);
        GmBuffer loss(bytes);
        GmBuffer workspace(32);
        GmBuffer tiling(sizeof(HingeLossTilingData));
        ASSERT_NE(predict.Get(), nullptr);
        ASSERT_NE(target.Get(), nullptr);
        ASSERT_NE(loss.Get(), nullptr);
        ASSERT_NE(workspace.Get(), nullptr);
        ASSERT_NE(tiling.Get(), nullptr);

        size_t predictBytes = bytes;
        size_t targetBytes = bytes;
        const std::string prefix = "./hinge_loss_data/" + dtype;
        ASSERT_TRUE(ReadFile(prefix + "_predict_t_hinge_loss.bin", predictBytes, predict.Get(), bytes));
        ASSERT_TRUE(ReadFile(prefix + "_target_t_hinge_loss.bin", targetBytes, target.Get(), bytes));
        ASSERT_EQ(predictBytes, bytes);
        ASSERT_EQ(targetBytes, bytes);
        *reinterpret_cast<HingeLossTilingData*>(tiling.Get()) = tilingData;

        ICPU_SET_TILING_KEY(0);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        auto kernel = [](GM_ADDR p, GM_ADDR t, GM_ADDR l, GM_ADDR w, GM_ADDR td) {
            ::hinge_loss<schMode>(p, t, l, w, td);
        };
        ICPU_RUN_KF(kernel, blockDim, predict.Get(), target.Get(), loss.Get(), workspace.Get(), tiling.Get());
        ASSERT_TRUE(WriteFile(prefix + "_output_loss_t_hinge_loss.bin", loss.Get(), bytes));
        ASSERT_TRUE(kernel_ut::RunCompareData("./hinge_loss_data", {dtype}));
    }
};

TEST_F(HingeLossTest, float32_margin_positive_zero_and_negative)
{
    constexpr uint32_t count = 256 * 32;
    HingeLossTilingData tilingData = {};
    tilingData.smallCoreDataNum = count;
    tilingData.finalSmallTileNum = 4;
    tilingData.tileDataNum = 2048;
    tilingData.smallTailDataNum = 2048;
    RunKernelCase<0>("'(256, 32)'", "float32", count, sizeof(float), 1, tilingData);
}

TEST_F(HingeLossTest, float32_uneven_multicore_with_tail_tile)
{
    constexpr uint32_t count = 8193;
    HingeLossTilingData tilingData = {};
    tilingData.smallCoreDataNum = 2048;
    tilingData.bigCoreDataNum = 2049;
    tilingData.finalBigTileNum = 2;
    tilingData.finalSmallTileNum = 1;
    tilingData.tileDataNum = 2048;
    tilingData.smallTailDataNum = 2048;
    tilingData.bigTailDataNum = 1;
    tilingData.tailBlockNum = 1;
    RunKernelCase<0>("'(1, 8193)'", "float32", count, sizeof(float), 4, tilingData);
}

TEST_F(HingeLossTest, float16_margin_positive_zero_and_negative)
{
    constexpr uint32_t count = 256 * 32;
    HingeLossTilingData tilingData = {};
    tilingData.smallCoreDataNum = count;
    tilingData.finalSmallTileNum = 2;
    tilingData.tileDataNum = 4096;
    tilingData.smallTailDataNum = 4096;
    RunKernelCase<1>("'(256, 32)'", "float16", count, sizeof(uint16_t), 1, tilingData);
}

TEST_F(HingeLossTest, bfloat16_margin_positive_zero_and_negative)
{
    constexpr uint32_t count = 256 * 32;
    HingeLossTilingData tilingData = {};
    tilingData.smallCoreDataNum = count;
    tilingData.finalSmallTileNum = 2;
    tilingData.tileDataNum = 4096;
    tilingData.smallTailDataNum = 4096;
    RunKernelCase<2>("'(256, 32)'", "bfloat16", count, sizeof(uint16_t), 1, tilingData);
}
