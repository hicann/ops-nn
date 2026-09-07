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
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
#include <cmath>
#include <limits>

#include "../../../op_kernel/anti_mx_quant.cpp"

using namespace std;

namespace {
constexpr size_t kUbSize = 253952;
constexpr uint32_t kDstTypeFp32 = 0;
constexpr uint32_t kTotalCoreNum = 64;
constexpr uint32_t kTilingKeyTailAxis = 0; // TPL_AXIS_TAIL
constexpr int64_t kBlockSize = 32;         // mx group 大小：每 32 个元素共享 1 字节 E8M0 scale
constexpr int64_t kSplitM = 1;             // 宿侧 tiling：行方向块大小
constexpr int64_t kSplitN = 512;           // 宿侧 tiling：列方向块大小
constexpr int64_t kBlocksPer512 = 16;      // 512 / 32
// FP8 -> FP32 每个 512 元素块占用：DB_BUFFER * (512 + 2048 + 16) + 64 = 5216 字节
constexpr int64_t kBytesPerUbBlock = 2 * (512 + 2048 + 16) + 64;

static inline int64_t CeilDivInt(int64_t a, int64_t b) { return (a + b - 1) / b; }

// FP8_E4M3FN 解码：S|EEEE|MMM，bias 7，无 inf，S.1111.111 = NaN
static inline float Fp8E4m3fnToFloat(uint8_t v)
{
    const uint32_t sign = (v >> 7) & 0x1U;
    const uint32_t exp = (v >> 3) & 0xFU;
    const uint32_t man = v & 0x7U;
    float val = 0.0f;
    if (exp == 0) {
        val = std::ldexp(static_cast<float>(man), -9); // 次正规数：(m / 8) * 2^-6
    } else if (exp == 0xFU && man == 0x7U) {
        return std::numeric_limits<float>::quiet_NaN(); // 0x7F / 0xFF
    } else {
        val = std::ldexp(static_cast<float>(8 + man), static_cast<int>(exp) - 10); // (1 + m/8) * 2^(e-7)
    }
    return sign ? -val : val;
}

// FP8_E8M0 解码：value = 2^(code - 127)。测试取值范围保证 E8M0 -> BF16 -> FP32 链路精确可表示
static inline float E8m0ScaleToFloat(uint8_t v) { return std::ldexp(1.0f, static_cast<int>(v) - 127); }
} // namespace

class anti_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "anti_mx_quant_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "anti_mx_quant_test TearDown\n" << endl; }
};

// 公共执行函数：构造已知值数据 -> 调 kernel -> 逐元素位级校验
static void RunAntiMxQuantCase(int64_t rowNum, int64_t colNum, int64_t rowTileNum, int64_t colTileNum)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // mxScale 每行布局：ceil2(ceil(colNum / 32)) 字节 E8M0
    const int64_t scaleColNum = CeilDivInt(CeilDivInt(colNum, kBlockSize), 2) * 2;
    size_t shape_x = rowNum * colNum * sizeof(uint8_t);
    size_t shape_mxScale = rowNum * scaleColNum * sizeof(uint8_t);
    size_t shape_y = rowNum * colNum * sizeof(float);

    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(AntiMxQuantTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t* xGM = (uint8_t*)AscendC::GmAlloc(shape_x);
    uint8_t* mxScaleGM = (uint8_t*)AscendC::GmAlloc(shape_mxScale);
    uint8_t* yGM = (uint8_t*)AscendC::GmAlloc(shape_y);

    memset(xGM, 0, shape_x);
    memset(mxScaleGM, 0, shape_mxScale);
    memset(yGM, 0, shape_y);

    // 已知值输入：x[i] = (i * 67 + 29) & 0xFF 覆盖全部 256 个 E4M3FN 编码（正常数/次正规数/±0/NaN）；
    // mxScale[r][g] = 127 + ((r + g) % 5) - 2，指数 125~129（即 2^-2 ~ 2^2），相邻 group 的 scale
    {
        for (int64_t i = 0; i < rowNum * colNum; i++) {
            xGM[i] = static_cast<uint8_t>((i * 67 + 29) & 0xFF);
        }
        for (int64_t r = 0; r < rowNum; r++) {
            for (int64_t g = 0; g < scaleColNum; g++) {
                mxScaleGM[r * scaleColNum + g] = static_cast<uint8_t>(127 + ((r + g) % 5) - 2);
            }
        }
    }

    const int64_t rowBlockLoopNum = CeilDivInt(rowNum, kSplitM);
    const int64_t colBlockLoopNum = CeilDivInt(colNum, kSplitN);
    const int64_t usedCoreNum = rowTileNum * colTileNum;
    const int64_t rowNormalBlockNum = CeilDivInt(rowBlockLoopNum, rowTileNum);
    const int64_t colNormalBlockNum = CeilDivInt(colBlockLoopNum, colTileNum);
    const int64_t rowTailLen = rowNum - rowNormalBlockNum * (rowTileNum - 1);
    const int64_t colTailLen = colNum - colNormalBlockNum * kSplitN * (colTileNum - 1);
    ASSERT_GT(rowNormalBlockNum, 0);
    ASSERT_GT(colNormalBlockNum, 0);
    ASSERT_GT(rowTailLen, 0);
    ASSERT_GT(colTailLen, 0);
    ASSERT_EQ(rowNormalBlockNum * (rowTileNum - 1) + rowTailLen, rowNum);
    ASSERT_EQ(colNormalBlockNum * kSplitN * (colTileNum - 1) + colTailLen, colNum);

    AntiMxQuantTilingData* tilingData = reinterpret_cast<AntiMxQuantTilingData*>(tiling);
    tilingData->ubSize = kUbSize;
    tilingData->dstType = kDstTypeFp32;
    tilingData->totalCoreNum = kTotalCoreNum;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->rowTileNum = rowTileNum;
    tilingData->colTileNum = colTileNum;
    tilingData->rowNum = rowNum;
    tilingData->colNum = colNum;
    tilingData->colNormalBlockNum = colNormalBlockNum;
    tilingData->colTailLen = colTailLen;
    tilingData->rowNormalBlockNum = rowNormalBlockNum;
    tilingData->rowTailLen = rowTailLen;
    tilingData->maxUbBlockNum = (static_cast<int64_t>(kUbSize) / kBytesPerUbBlock) * kBlocksPer512;

    ICPU_SET_TILING_KEY(kTilingKeyTailAxis);
    ICPU_RUN_KF(::anti_mx_quant<kTilingKeyTailAxis>, static_cast<uint32_t>(usedCoreNum), xGM, mxScaleGM, yGM, workspace,
                (uint8_t*)(tilingData));

    // 逐元素校验：y[r][c] = fp32(e4m3fn(x[r][c])) * 2^(E8M0(mxScale[r][c/32]) - 127)，位级一致
    {
        const float* yF = reinterpret_cast<const float*>(yGM);
        size_t n = shape_y / sizeof(float);
        for (size_t i = 0; i < n; i++) {
            int64_t r = static_cast<int64_t>(i) / colNum;
            int64_t c = static_cast<int64_t>(i) % colNum;
            float expected = Fp8E4m3fnToFloat(xGM[i]) * E8m0ScaleToFloat(mxScaleGM[r * scaleColNum + c / kBlockSize]);
            // NaN 传播语义：双方均为 NaN 视为一致
            bool nanMatch = std::isnan(expected) && std::isnan(yF[i]);
            if (!nanMatch && memcmp(&yF[i], &expected, sizeof(float)) != 0) {
                uint32_t expectedBits = 0;
                uint32_t actualBits = 0;
                (void)memcpy(&expectedBits, &expected, sizeof(expectedBits));
                (void)memcpy(&actualBits, &yF[i], sizeof(actualBits));
                printf("anti_mx_quant mismatch at [%ld, %ld]: xCode=0x%02X, scaleCode=0x%02X, "
                       "expected %f(0x%08X), got %f(0x%08X)\n",
                       static_cast<long>(r), static_cast<long>(c), static_cast<unsigned int>(xGM[i]),
                       static_cast<unsigned int>(mxScaleGM[r * scaleColNum + c / kBlockSize]), expected, expectedBits,
                       yF[i], actualBits);
                ASSERT_EQ(yF[i], expected);
                break;
            }
        }
    }
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    AscendC::GmFree((void*)xGM);
    AscendC::GmFree((void*)mxScaleGM);
    AscendC::GmFree((void*)yGM);
}

// 单核场景：shape [2, 512]，单个 1x512 列块，全部行落在核 0 上
TEST_F(anti_mx_quant_test, fp8_e4m3_to_fp32_basic) { RunAntiMxQuantCase(2, 512, 1, 1); }

// 列切分尾块场景：shape [3, 600] 双核，colTileNum=2 -> 核 0 处理 512 正常列，
// 核 1 处理 88 尾列；rowTailLen=3 保证行循环非退化
TEST_F(anti_mx_quant_test, fp8_e4m3_to_fp32_col_tail_multicore) { RunAntiMxQuantCase(3, 600, 1, 2); }

// 行切分多核场景：shape [4, 512] 双核，rowTileNum=2 -> 每核 2 行（2 + 2）；
// colTailLen=512（整块）保证列循环非退化
TEST_F(anti_mx_quant_test, fp8_e4m3_to_fp32_row_split_multicore) { RunAntiMxQuantCase(4, 512, 2, 1); }
