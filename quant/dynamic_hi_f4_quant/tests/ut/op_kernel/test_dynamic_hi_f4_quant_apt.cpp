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
#include <iostream>
#include <string>
#include <cstring>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "dynamic_hi_f4_quant_data_utils.h"
#include "../../../op_kernel/arch35/dynamic_hi_f4_quant_tilingdata.h"

using namespace std;

extern "C" __global__ __aicore__ void dynamic_hi_f4_quant(GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
                                                          GM_ADDR tiling);

class DynamicHiF4QuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "DynamicHiF4QuantKernelTest SetUp" << endl; }
    static void TearDownTestCase() { cout << "DynamicHiF4QuantKernelTest TearDown" << endl; }
};

static void RunKernelTestAxisNeg1(int64_t dim0, int64_t dim1, int64_t totalCoreNum, uint64_t tilingKey)
{
    // CCE 移植后的 -1 轴 tiling 语义: 定长 tile + 按 tile 分核, 循环数由 kernel 派生;
    // tile 粒度取 256 group(kernel 700B/group 记账共 175KB, 兼容 ICPU 仿真 UB)
    int64_t groupNum = dim0 * dim1 / 64;
    int64_t alignGroupNum = 256;
    int64_t totalTiles = (groupNum + alignGroupNum - 1) / alignGroupNum;
    int64_t usedCoreNum = std::min(totalCoreNum, totalTiles);
    int64_t blockPerHeadCore = (totalTiles + usedCoreNum - 1) / usedCoreNum;
    int64_t blockPerTailCore = totalTiles / usedCoreNum;
    int64_t headCoreNum = totalTiles % usedCoreNum;

    size_t inputSize = dim0 * dim1 * sizeof(uint16_t);
    size_t outputSize = dim0 * dim1 / 2;
    size_t scaleSize = dim0 * (dim1 / 64) * sizeof(float);
    size_t tilingDataSize = sizeof(DynamicHiF4QuantTilingData);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    DynamicHiF4QuantTilingData* tilingData = reinterpret_cast<DynamicHiF4QuantTilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->groupNum = groupNum;
    tilingData->alignGroupNum = alignGroupNum;
    tilingData->blockPerHeadCore = blockPerHeadCore;
    tilingData->blockPerTailCore = blockPerTailCore;
    tilingData->headCoreNum = headCoreNum;
    tilingData->scaleAlg = 0;

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dynamic_hi_f4_quant, blockDim, x, y, scale, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// -2 轴手搓 tiling: blockW/splitBlockH 由用例指定(横竖自适应各档位), 其余字段与 host SplitCore 公式一致
static void FillNeg2TilingData(DynamicHiF4QuantTilingData* tilingData, int64_t dim0, int64_t dimNeg2, int64_t dimNeg1,
                               int64_t totalCoreNum, uint64_t tilingKey, int64_t blockW, int64_t splitBlockH)
{
    int64_t dimNeg2SplitBlockNum = (dimNeg2 + splitBlockH - 1) / splitBlockH;
    int64_t dimNeg1BlockNum = (dimNeg1 + blockW - 1) / blockW;
    int64_t totalTaskNum = dim0 * dimNeg2SplitBlockNum * dimNeg1BlockNum;

    tilingData->totalCoreNum = totalCoreNum;
    tilingData->usedCoreNum = std::min(totalCoreNum, totalTaskNum);
    tilingData->dim0 = dim0;
    tilingData->dimNeg2 = dimNeg2;
    tilingData->dimNeg1 = dimNeg1;
    tilingData->blockW = blockW;
    tilingData->tilingKey = tilingKey;
    tilingData->splitBlockH = splitBlockH;
    tilingData->dimNeg2Tail = dimNeg2 % splitBlockH == 0 ? splitBlockH : dimNeg2 % splitBlockH;
    tilingData->dimNeg1Tail = dimNeg1 % blockW == 0 ? blockW : dimNeg1 % blockW;
    tilingData->dimNeg2SplitBlockNum = dimNeg2SplitBlockNum;
    tilingData->dimNeg1BlockNum = dimNeg1BlockNum;
    tilingData->blockPerHeadCore = (totalTaskNum + totalCoreNum - 1) / totalCoreNum;
    tilingData->blockPerTailCore = totalTaskNum / totalCoreNum;
    tilingData->headCoreNum = totalTaskNum % totalCoreNum;
    tilingData->dimNeg2IsOdd = (dimNeg2 / 64) % 2;
    tilingData->dimNeg1IsOdd = (dimNeg1 / 64) % 2;
    tilingData->dimNeg1IsPad = dimNeg1 % 64 == 0 ? 0 : 1;
    tilingData->blockCountPerBatch = dimNeg2SplitBlockNum * dimNeg1BlockNum;
    tilingData->scale1ColCountPerBatch = (dimNeg1 + 63) / 64;
    tilingData->scale2RowCountPerBatch = (dimNeg2 + 63) / 64;
    tilingData->scaleAlg = 0;
}

static void RunKernelTestAxisNeg2(int64_t dim0, int64_t dimNeg2, int64_t dimNeg1, int64_t totalCoreNum,
                                  uint64_t tilingKey, int64_t blockW = 128, int64_t splitBlockH = 64)
{
    size_t inputSize = dim0 * dimNeg2 * dimNeg1 * sizeof(uint16_t);
    size_t outputSize = dim0 * dimNeg2 * dimNeg1 / 2;
    size_t scaleSize = dim0 * (dimNeg2 / 64) * dimNeg1 * 2 * sizeof(uint16_t);
    size_t tilingDataSize = sizeof(DynamicHiF4QuantTilingData);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* scaleBuf = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* tilingData = reinterpret_cast<DynamicHiF4QuantTilingData*>(tiling);
    FillNeg2TilingData(tilingData, dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, blockW, splitBlockH);

    uint32_t blockDim = static_cast<uint32_t>(tilingData->usedCoreNum);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dynamic_hi_f4_quant, blockDim, x, y, scaleBuf, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(scaleBuf);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_bf16_basic)
{
    int64_t dim0 = 2;
    int64_t dim1 = 128;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_bf16_large)
{
    int64_t dim0 = 16;
    int64_t dim1 = 1024;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_small)
{
    int64_t dim0 = 1;
    int64_t dim1 = 64;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_bf16_multi_tile)
{
    // 多 tile/核(不均分): groups=27648 -> tiles=108, 头部 12 核各 5 tile(触发 y 双缓冲环回转、
    // 输入 4 深环回转、scale 满批 4 tile 与核末部分批), 其余 12 核各 4 tile
    int64_t dim0 = 216;
    int64_t dim1 = 8192;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg1_bf16_tail_tile)
{
    // 全局尾 tile 非整块: groups=258 -> tiles=2, 尾 tile 有效 2 group
    int64_t dim0 = 2;
    int64_t dim1 = 8256;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 0;
    RunKernelTestAxisNeg1(dim0, dim1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_bf16_basic)
{
    int64_t dim0 = 1;
    int64_t dimNeg2 = 128;
    int64_t dimNeg1 = 128;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_bf16_large)
{
    int64_t dim0 = 4;
    int64_t dimNeg2 = 512;
    int64_t dimNeg1 = 256;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_narrow_g4_multitask)
{
    // 窄形竖切档: W=128, G=4(H=256); 7168=256×28 精确整除 → 28 任务, GM srcStride=0 连续流
    int64_t dim0 = 1;
    int64_t dimNeg2 = 7168;
    int64_t dimNeg1 = 128;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 128, 256);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_rowtail_multichunk)
{
    // 行尾×多 chunk(中间缓冲布局专项): W=256(chunkLoop=2), G=2(H=128); 320=2×128+64 → 3 行带尾 64,
    // 尾带 "chunk 槽位整 H 分配 + 块内前缀填充" 的生产/消费重定基路径在此触发
    int64_t dim0 = 1;
    int64_t dimNeg2 = 320;
    int64_t dimNeg1 = 256;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 256, 128);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_coltail_wide_multiband)
{
    // 宽形列尾 + 多行带: W=512(G=1, H=64); 8256=16×512+64 → 17 列块尾 64; dimNeg2=192 → 3 行带
    int64_t dim0 = 1;
    int64_t dimNeg2 = 192;
    int64_t dimNeg1 = 8256;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 512, 64);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_ceil_stale_multibatch)
{
    // ceil 档 stale 列槽 + 多 batch: W=256(有效 192), G=2(H=128); stale 列的垃圾输出由 CopyOut 截断
    int64_t dim0 = 2;
    int64_t dimNeg2 = 128;
    int64_t dimNeg1 = 192;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 256, 128);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_rowtail_g4_multitask)
{
    // 窄形行尾: W=128, G=4(H=256); 7232=28×256+64 → 29 行带尾 64
    int64_t dim0 = 1;
    int64_t dimNeg2 = 7232;
    int64_t dimNeg1 = 128;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 128, 256);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_min_shape)
{
    // 最小 shape: W=128, G=4(H=256), 单行带 H'=64(行向循环数按尾切)
    int64_t dim0 = 1;
    int64_t dimNeg2 = 64;
    int64_t dimNeg1 = 64;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    RunKernelTestAxisNeg2(dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 128, 256);
}

TEST_F(DynamicHiF4QuantKernelTest, test_axis_neg2_dual_tiling_bitexact)
{
    // 同输入双 tiling 对拍: 量化链逐 lane 独立(无跨列/跨组归约) ⇒ y/scale 与 tiling 逐位一致。
    // tiling A = W=256/G=2(chunkLoop=2 × 行尾 64, 触发 chunk 重定基 + scale 组主序路径),
    // tiling B = W=128/G=4(纯竖切, chunkLoop=1); 任何中间缓冲错位都会使 A/B 输出出现字节差异
    int64_t dim0 = 1;
    int64_t dimNeg2 = 320;
    int64_t dimNeg1 = 256;
    int64_t totalCoreNum = 24;
    uint64_t tilingKey = 1;
    size_t inputSize = dim0 * dimNeg2 * dimNeg1 * sizeof(uint16_t);
    size_t outputSize = dim0 * dimNeg2 * dimNeg1 / 2;
    size_t scaleSize = dim0 * (dimNeg2 / 64) * dimNeg1 * 2 * sizeof(uint16_t);
    size_t workspaceSize = 32;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize);
    // 确定性填充: 有界幅值 bf16 模式(避免 NaN/Inf)
    uint16_t* xU16 = reinterpret_cast<uint16_t*>(x);
    for (size_t i = 0; i < inputSize / sizeof(uint16_t); i++) {
        float v = static_cast<float>((i * 37 % 33) - 16) * 0.25f; // [-4, 4]
        uint32_t bits = *reinterpret_cast<uint32_t*>(&v);
        xU16[i] = static_cast<uint16_t>(bits >> 16);
    }
    uint8_t* yA = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* sA = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* yB = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* sB = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(DynamicHiF4QuantTilingData));

    auto* tilingData = reinterpret_cast<DynamicHiF4QuantTilingData*>(tiling);
    FillNeg2TilingData(tilingData, dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 256, 128);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(dynamic_hi_f4_quant, static_cast<uint32_t>(tilingData->usedCoreNum), x, yA, sA, workspace, tiling);

    FillNeg2TilingData(tilingData, dim0, dimNeg2, dimNeg1, totalCoreNum, tilingKey, 128, 256);
    ICPU_RUN_KF(dynamic_hi_f4_quant, static_cast<uint32_t>(tilingData->usedCoreNum), x, yB, sB, workspace, tiling);

    EXPECT_EQ(std::memcmp(yA, yB, outputSize), 0);
    EXPECT_EQ(std::memcmp(sA, sB, scaleSize), 0);

    AscendC::GmFree(x);
    AscendC::GmFree(yA);
    AscendC::GmFree(sA);
    AscendC::GmFree(yB);
    AscendC::GmFree(sB);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
