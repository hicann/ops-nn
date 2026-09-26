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
 * \file test_npu_scatter_add.cpp
 * \brief NpuScatterAdd kernel UT（x86 CPU 仿真）
 */
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "npu_scatter_add_tiling_def.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void npu_scatter_add(GM_ADDR x, GM_ADDR y, GM_ADDR s, GM_ADDR indices,
                                                      GM_ADDR sort_idx, GM_ADDR valid_token_num, GM_ADDR y_ref,
                                                      GM_ADDR workspace, GM_ADDR tiling);

namespace {
// IEEE 754 float -> fp16 位模式转换（宿主侧构造数据用，round to nearest even）
uint16_t FloatToFp16Bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFF;
    if (exp <= 0) {
        return sign; // 下溢为0
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00); // inf
    }
    uint16_t half = static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
    uint32_t roundBits = mantissa & 0x1FFF;
    if (roundBits > 0x1000 || (roundBits == 0x1000 && (half & 1) != 0)) {
        half++;
    }
    return half;
}

float Fp16BitsToFloat(uint16_t bits)
{
    uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    uint32_t out = 0;
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7F800000 | (mantissa << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mantissa << 13);
    }
    float f = 0.0f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

// IEEE 754 float -> bf16 位模式转换（截断高16位，测试数据为精确整数，无精度损失）
uint16_t FloatToBf16Bits(float value)
{
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

float Bf16BitsToFloat(uint16_t bits)
{
    uint32_t out = static_cast<uint32_t>(bits) << 16;
    float f = 0.0f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

// 结构化测试参数，仅包含变化项
struct NpuScatterAddTestParams {
    int64_t S;   // 源行数
    int64_t D;   // 目标行数
    int64_t H;   // 隐藏维度
    bool isBf16; // 数据类型：true-bf16, false-fp16
    bool withValid;
    int64_t validTokenNum;
};
} // namespace

class npu_scatter_add_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "npu_scatter_add_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "npu_scatter_add_test TearDown\n" << endl; }

    // 数据全部为小整数值，在fp16/bf16下均可精确表示，golden可按整数累加精确比对。
    // x第i行所有元素为i+1，scale全1，y初始为0，indices[i] = (i * 7 + 3) % D。
    static void RunTest(const NpuScatterAddTestParams& params, int tilingKey, int coreNum)
    {
        const size_t kSysWorkspaceSize = 16 * 1024 * 1024 + 4096;
        const size_t kElemBytes = 2; // fp16/bf16

        const int64_t S = params.S;
        const int64_t D = params.D;
        const int64_t H = params.H;
        const uint32_t alignH = static_cast<uint32_t>((H * kElemBytes + 31) / 32 * 32 / kElemBytes);

        const size_t xSize = static_cast<size_t>(S * H) * kElemBytes;
        const size_t ySize = static_cast<size_t>(D * H) * kElemBytes;
        const size_t sSize = static_cast<size_t>(S) * kElemBytes;
        const size_t idxSize = static_cast<size_t>(S) * sizeof(int32_t);
        const size_t validSize = sizeof(int32_t);
        const size_t workspaceSize = kSysWorkspaceSize + coreNum * alignH * kElemBytes + coreNum * sizeof(int32_t);

        // 设备内存分配
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(NpuScatterAddTilingData));
        uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(xSize);
        uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(ySize);
        uint8_t* sGM = (uint8_t*)AscendC::GmAlloc(sSize);
        uint8_t* indicesGM = (uint8_t*)AscendC::GmAlloc(idxSize);
        uint8_t* sortIdxGM = (uint8_t*)AscendC::GmAlloc(idxSize);
        uint8_t* validGM = params.withValid ? (uint8_t*)AscendC::GmAlloc(validSize) : nullptr;

        // 构造host数据
        vector<uint16_t> xData(static_cast<size_t>(S * H));
        vector<uint16_t> yData(static_cast<size_t>(D * H), 0);
        vector<uint16_t> sData(static_cast<size_t>(S));
        vector<int32_t> indicesData(static_cast<size_t>(S));
        vector<int32_t> sortIdxData(static_cast<size_t>(S));
        for (int64_t i = 0; i < S; i++) {
            float rowValue = static_cast<float>(i + 1);
            for (int64_t j = 0; j < H; j++) {
                xData[static_cast<size_t>(i * H + j)] = params.isBf16 ? FloatToBf16Bits(rowValue) :
                                                                        FloatToFp16Bits(rowValue);
            }
            sData[static_cast<size_t>(i)] = params.isBf16 ? FloatToBf16Bits(1.0f) : FloatToFp16Bits(1.0f);
            indicesData[static_cast<size_t>(i)] = static_cast<int32_t>((i * 7 + 3) % D);
        }
        // sort_idx = argsort(indices)，稳定排序
        iota(sortIdxData.begin(), sortIdxData.end(), 0);
        stable_sort(sortIdxData.begin(), sortIdxData.end(),
                    [&indicesData](int32_t a, int32_t b) { return indicesData[a] < indicesData[b]; });

        // 拷贝到GM
        memcpy(xGM, xData.data(), xSize);
        memcpy(yGM, yData.data(), ySize);
        memcpy(sGM, sData.data(), sSize);
        memcpy(indicesGM, indicesData.data(), idxSize);
        memcpy(sortIdxGM, sortIdxData.data(), idxSize);
        if (params.withValid) {
            int32_t validNum = static_cast<int32_t>(params.validTokenNum);
            memcpy(validGM, &validNum, validSize);
        }

        // 配置Tiling参数
        NpuScatterAddTilingData* tilingData = reinterpret_cast<NpuScatterAddTilingData*>(tiling);
        tilingData->totalRows = static_cast<uint32_t>(params.withValid ? params.validTokenNum : S);
        tilingData->hiddenState = static_cast<uint32_t>(H);
        tilingData->alignHiddenState = alignH;
        tilingData->usedCoreNum = static_cast<uint32_t>(coreNum);
        tilingData->withValid = params.withValid ? 1 : 0;

        // 执行Kernel
        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(npu_scatter_add, coreNum, xGM, yGM, sGM, indicesGM, sortIdxGM, validGM, yGM, workspace, tiling);

        // 计算golden：整数累加，与kernel逐位一致
        const int64_t validRows = params.withValid ? params.validTokenNum : S;
        vector<uint16_t> golden(static_cast<size_t>(D * H), 0);
        for (int64_t i = 0; i < validRows; i++) {
            int64_t dst = indicesData[static_cast<size_t>(i)];
            for (int64_t j = 0; j < H; j++) {
                size_t idx = static_cast<size_t>(dst * H + j);
                float acc = params.isBf16 ? Bf16BitsToFloat(golden[idx]) : Fp16BitsToFloat(golden[idx]);
                acc += static_cast<float>(i + 1); // scale全1
                golden[idx] = params.isBf16 ? FloatToBf16Bits(acc) : FloatToFp16Bits(acc);
            }
        }

        // 比对结果
        vector<uint16_t> result(static_cast<size_t>(D * H), 0);
        memcpy(result.data(), yGM, ySize);
        for (size_t i = 0; i < result.size(); i++) {
            EXPECT_EQ(result[i], golden[i]) << "index: " << i;
        }

        // 内存释放（顺序与分配相反）
        if (validGM != nullptr) {
            AscendC::GmFree(validGM);
        }
        AscendC::GmFree(sortIdxGM);
        AscendC::GmFree(indicesGM);
        AscendC::GmFree(sGM);
        AscendC::GmFree(yGM);
        AscendC::GmFree(xGM);
        AscendC::GmFree(tiling);
        AscendC::GmFree(workspace);
    }
};

// tilingKey 1: fp16 + scale
TEST_F(npu_scatter_add_test, npu_scatter_add_kernel_test_fp16_with_scale)
{
    NpuScatterAddTestParams params = {
        .S = 16, .D = 8, .H = 32, .isBf16 = false, .withValid = false, .validTokenNum = 0};
    RunTest(params, 1, 8);
}

// tilingKey 3: fp16 无scale
TEST_F(npu_scatter_add_test, npu_scatter_add_kernel_test_fp16_no_scale)
{
    NpuScatterAddTestParams params = {
        .S = 16, .D = 8, .H = 32, .isBf16 = false, .withValid = false, .validTokenNum = 0};
    RunTest(params, 3, 8);
}

// tilingKey 5: fp16 + scale + 高精度模式（整数数据下与普通模式结果一致）
TEST_F(npu_scatter_add_test, npu_scatter_add_kernel_test_fp16_with_scale_high_precision)
{
    NpuScatterAddTestParams params = {
        .S = 16, .D = 8, .H = 32, .isBf16 = false, .withValid = false, .validTokenNum = 0};
    RunTest(params, 5, 8);
}

// tilingKey 1: fp16 + scale + valid_token_num（仅处理前10行）
TEST_F(npu_scatter_add_test, npu_scatter_add_kernel_test_fp16_valid_token_num)
{
    NpuScatterAddTestParams params = {
        .S = 16, .D = 8, .H = 32, .isBf16 = false, .withValid = true, .validTokenNum = 10};
    RunTest(params, 1, 8);
}

// tilingKey 0: bf16 + scale（小规模数据，bf16整数精确表示范围）
TEST_F(npu_scatter_add_test, npu_scatter_add_kernel_test_bf16_with_scale)
{
    NpuScatterAddTestParams params = {.S = 8, .D = 4, .H = 32, .isBf16 = true, .withValid = false, .validTokenNum = 0};
    RunTest(params, 0, 4);
}

// tilingKey 1: fp16 + scale，单核场景
TEST_F(npu_scatter_add_test, npu_scatter_add_kernel_test_fp16_single_core)
{
    NpuScatterAddTestParams params = {
        .S = 16, .D = 8, .H = 32, .isBf16 = false, .withValid = false, .validTokenNum = 0};
    RunTest(params, 1, 1);
}
