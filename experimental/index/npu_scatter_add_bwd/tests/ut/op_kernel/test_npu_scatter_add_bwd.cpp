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
 * \file test_npu_scatter_add_bwd.cpp
 * \brief NpuScatterAddBwd kernel UT（x86 CPU 仿真）
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
#include "npu_scatter_add_bwd_tiling_def.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void npu_scatter_add_bwd(GM_ADDR y_grad, GM_ADDR x, GM_ADDR s, GM_ADDR indices,
                                                          GM_ADDR x_grad, GM_ADDR s_grad, GM_ADDR workspace,
                                                          GM_ADDR tiling);

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
struct NpuScatterAddBwdTestParams {
    int64_t N;   // 源行数
    int64_t D;   // 目标行数
    int64_t H;   // 隐藏维度
    bool isBf16; // 数据类型：true-bf16, false-fp16
};
} // namespace

class npu_scatter_add_bwd_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "npu_scatter_add_bwd_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "npu_scatter_add_bwd_test TearDown\n" << endl; }

    // 数据全部为小整数值，在fp16/bf16下均可精确表示，golden可按整数精确比对。
    // y_grad第i行所有元素为i+1，x第i行所有元素为i+1，scale全1，indices[i] = (i * 7 + 3) % D。
    static void RunTest(const NpuScatterAddBwdTestParams& params, int tilingKey, int coreNum)
    {
        const size_t kSysWorkspaceSize = 16 * 1024 * 1024 + 4096;
        const size_t kElemBytes = 2; // fp16/bf16

        const int64_t N = params.N;
        const int64_t D = params.D;
        const int64_t H = params.H;
        const uint32_t alignH = static_cast<uint32_t>((H * kElemBytes + 31) / 32 * 32 / kElemBytes);

        const size_t yGradSize = static_cast<size_t>(D * H) * kElemBytes;
        const size_t xSize = static_cast<size_t>(N * H) * kElemBytes;
        const size_t sSize = static_cast<size_t>(N) * kElemBytes;
        const size_t idxSize = static_cast<size_t>(N) * sizeof(int32_t);
        const size_t workspaceSize = kSysWorkspaceSize;

        // 设备内存分配
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(NpuScatterAddBwdTilingData));
        uint8_t* yGradGM = (uint8_t*)AscendC::GmAlloc(yGradSize);
        uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(xSize);
        uint8_t* sGM = (uint8_t*)AscendC::GmAlloc(sSize);
        uint8_t* indicesGM = (uint8_t*)AscendC::GmAlloc(idxSize);
        uint8_t* xGradGM = (uint8_t*)AscendC::GmAlloc(xSize);
        uint8_t* sGradGM = (uint8_t*)AscendC::GmAlloc(sSize);

        // 构造host数据
        vector<uint16_t> yGradData(static_cast<size_t>(D * H));
        vector<uint16_t> xData(static_cast<size_t>(N * H));
        vector<uint16_t> sData(static_cast<size_t>(N));
        vector<int32_t> indicesData(static_cast<size_t>(N));
        for (int64_t i = 0; i < D; i++) {
            for (int64_t j = 0; j < H; j++) {
                yGradData[static_cast<size_t>(i * H + j)] = params.isBf16 ? FloatToBf16Bits(static_cast<float>(i + 1)) :
                                                                            FloatToFp16Bits(static_cast<float>(i + 1));
            }
        }
        for (int64_t i = 0; i < N; i++) {
            float rowValue = static_cast<float>(i + 1);
            for (int64_t j = 0; j < H; j++) {
                xData[static_cast<size_t>(i * H + j)] = params.isBf16 ? FloatToBf16Bits(rowValue) :
                                                                        FloatToFp16Bits(rowValue);
            }
            sData[static_cast<size_t>(i)] = params.isBf16 ? FloatToBf16Bits(1.0f) : FloatToFp16Bits(1.0f);
            indicesData[static_cast<size_t>(i)] = static_cast<int32_t>((i * 7 + 3) % D);
        }

        // 拷贝到GM
        memcpy(yGradGM, yGradData.data(), yGradSize);
        memcpy(xGM, xData.data(), xSize);
        memcpy(sGM, sData.data(), sSize);
        memcpy(indicesGM, indicesData.data(), idxSize);
        memset(xGradGM, 0, xSize);
        memset(sGradGM, 0, sSize);

        // 配置Tiling参数
        NpuScatterAddBwdTilingData* tilingData = reinterpret_cast<NpuScatterAddBwdTilingData*>(tiling);
        tilingData->rowsPerCore = static_cast<uint32_t>((N + coreNum - 1) / coreNum);
        tilingData->totalRows = static_cast<uint32_t>(N);
        tilingData->hiddenState = static_cast<uint32_t>(H);
        tilingData->alignHiddenState = alignH;
        tilingData->usedCoreNum = static_cast<uint32_t>(coreNum);

        // 执行Kernel
        ICPU_SET_TILING_KEY(tilingKey);
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_RUN_KF(npu_scatter_add_bwd, coreNum, yGradGM, xGM, sGM, indicesGM, xGradGM, sGradGM, workspace, tiling);

        // 计算golden：整数精确累加，与kernel逐位一致
        vector<uint16_t> xGradGolden(static_cast<size_t>(N * H), 0);
        vector<uint16_t> sGradGolden(static_cast<size_t>(N), 0);
        for (int64_t i = 0; i < N; i++) {
            int64_t dst = indicesData[static_cast<size_t>(i)];
            float xGradVal = static_cast<float>(dst + 1); // y_grad[dst] * s[i] = (dst+1)*1
            for (int64_t j = 0; j < H; j++) {
                xGradGolden[static_cast<size_t>(i * H + j)] = params.isBf16 ? FloatToBf16Bits(xGradVal) :
                                                                              FloatToFp16Bits(xGradVal);
            }
            float sGradVal = static_cast<float>(H) * static_cast<float>(i + 1) * static_cast<float>(dst + 1);
            sGradGolden[static_cast<size_t>(i)] = params.isBf16 ? FloatToBf16Bits(sGradVal) : FloatToFp16Bits(sGradVal);
        }

        // 比对结果
        vector<uint16_t> xGradResult(static_cast<size_t>(N * H), 0);
        vector<uint16_t> sGradResult(static_cast<size_t>(N), 0);
        memcpy(xGradResult.data(), xGradGM, xSize);
        memcpy(sGradResult.data(), sGradGM, sSize);
        for (size_t i = 0; i < xGradResult.size(); i++) {
            EXPECT_EQ(xGradResult[i], xGradGolden[i]) << "x_grad index: " << i;
        }
        for (size_t i = 0; i < sGradResult.size(); i++) {
            EXPECT_EQ(sGradResult[i], sGradGolden[i]) << "s_grad index: " << i;
        }

        // 内存释放（顺序与分配相反）
        AscendC::GmFree(sGradGM);
        AscendC::GmFree(xGradGM);
        AscendC::GmFree(indicesGM);
        AscendC::GmFree(sGM);
        AscendC::GmFree(xGM);
        AscendC::GmFree(yGradGM);
        AscendC::GmFree(tiling);
        AscendC::GmFree(workspace);
    }
};

// tilingKey 0: bf16，多核
TEST_F(npu_scatter_add_bwd_test, npu_scatter_add_bwd_kernel_test_bf16)
{
    NpuScatterAddBwdTestParams params = {.N = 16, .D = 8, .H = 32, .isBf16 = true};
    RunTest(params, 0, 8);
}

// tilingKey 1: fp16，多核
TEST_F(npu_scatter_add_bwd_test, npu_scatter_add_bwd_kernel_test_fp16)
{
    NpuScatterAddBwdTestParams params = {.N = 16, .D = 8, .H = 32, .isBf16 = false};
    RunTest(params, 1, 8);
}

// tilingKey 1: fp16，单核
TEST_F(npu_scatter_add_bwd_test, npu_scatter_add_bwd_kernel_test_fp16_single_core)
{
    NpuScatterAddBwdTestParams params = {.N = 16, .D = 8, .H = 32, .isBf16 = false};
    RunTest(params, 1, 1);
}

// tilingKey 0: bf16，隐藏维度非32B对齐（H=30 -> alignH=32）
TEST_F(npu_scatter_add_bwd_test, npu_scatter_add_bwd_kernel_test_bf16_unaligned)
{
    NpuScatterAddBwdTestParams params = {.N = 8, .D = 4, .H = 30, .isBf16 = true};
    RunTest(params, 0, 4);
}
