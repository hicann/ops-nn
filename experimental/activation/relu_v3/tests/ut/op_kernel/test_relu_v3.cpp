/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#ifdef __CCE_KT_TEST__
#include "data_utils.h"
#include "tikicpulib.h"
#endif
#include "../../../op_kernel/relu_v3.cpp"

void *GmAllocAlign(size_t size)
{
    return GmAlloc(size + 31 >> 5 << 5);
}

template<typename T>
constexpr const char *GetTorchType()
{
    if constexpr (std::is_same_v<T, half>)
        return "float16";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_same_v<T, int>)
        return "int32";
    else if constexpr (std::is_same_v<T, int8_t>)
        return "int8";
    else if constexpr (std::is_same_v<T, uint8_t>)
        return "uint8";
    else if constexpr (std::is_same_v<T, bfloat16_t>)
        return "bfloat16";
    else
        static_assert(!std::is_same_v<T, T>);
}

template<typename... Args>
int ExecuteCommand(const char *path, Args... args)
{
    std::string command = path;
    ((command = command + " '" + args + "'"), ...);
    return std::system(command.c_str());
}

class relu_v3_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "relu_v3_test SetUp\n" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "relu_v3_test TearDown\n" << std::endl;
    }
};

TEST_F(relu_v3_test, test_case_0)
{
    size_t x_size = 448000 * sizeof(DTYPE_X);
    size_t y_size = 448000 * sizeof(DTYPE_Y);
    size_t mask_size = 56000 * sizeof(DTYPE_MASK);
    size_t tiling_size = sizeof(ReluV3TilingData);
    uint32_t block_dim = 20;

    auto x = static_cast<GM_ADDR>(GmAllocAlign(x_size));
    auto y = static_cast<GM_ADDR>(GmAllocAlign(y_size));
    auto mask = static_cast<GM_ADDR>(GmAllocAlign(mask_size));
    auto workspace = static_cast<GM_ADDR>(nullptr);
    auto tiling = static_cast<GM_ADDR>(GmAllocAlign(tiling_size));

    auto tiling_data = reinterpret_cast<ReluV3TilingData *>(tiling);
    tiling_data->size = 448000;

    ICPU_SET_TILING_KEY(0);
    SetKernelMode(KernelMode::AIV_MODE);

    auto x_type = GetTorchType<DTYPE_X>();
    auto x_shape = "(25, 7, 20, 8, 16)";
    auto x_range = "(-2.0, 2.0)";
    ExecuteCommand("python", "gen_data.py", x_type, x_shape, x_range);

    ReadFile("x.bin", x_size, x, x_size);

    ICPU_RUN_KF(relu_v3<0>, block_dim, x, y, mask, workspace, tiling);

    WriteFile("y.bin", y, y_size);
    WriteFile("mask.bin", mask, mask_size);

    auto y_type = GetTorchType<DTYPE_Y>();
    auto y_loss = "0";
    auto mask_type = GetTorchType<DTYPE_MASK>();
    auto mask_loss = "0";
    EXPECT_EQ(ExecuteCommand("python", "compare_data.py", y_type, y_loss, mask_type, mask_loss), 0);

    GmFree(x);
    GmFree(y);
    GmFree(mask);
    GmFree(tiling);
}
