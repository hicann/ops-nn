/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file test_bn_infer_grad_tiling.cpp
 * \brief BnInferGrad Tiling UT - Iteration 1, 2 & 3
 *
 * Iteration 1: NCHW format, fp32, CONTIGUOUS branch (TilingKey=0).
 * Verifies tileLen, totalElements, channelSize, spatialSize, numTiles, etc.
 *
 * Iteration 2: NHWC/NC1HWC0 format Tiling + multi-core split logic.
 * Verifies formatMode, NHWC tileLen alignment to channelSize,
 * NC1HWC0 totalTasks/tasksPerCore/tileHW/numTilesHW fields,
 * and multi-core usedCoreNum/blockNum distribution.
 *
 * Iteration 3: Full dtype coverage (fp16/bf16) + boundary cases.
 * Verifies fp16 and bf16 tiling for NCHW/NHWC/NC1HWC0 formats,
 * empty tensor handling (totalElements=0), and channel=1 boundary.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include "tiling_case_executor.h"
#include "bn_infer_grad_tiling_data.h"

using namespace std;
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// CompileInfo placeholder required by TilingContextBuilder::Build()
struct BnInferGradCompileInfo {};
static BnInferGradCompileInfo g_compileInfo;

// Constants matching the tiling implementation
constexpr uint32_t BLOCK_SIZE = 32U;
constexpr uint32_t FLOAT_SIZE = 4U;
constexpr uint32_t ALIGN_ELEM_FP32 = BLOCK_SIZE / FLOAT_SIZE;  // 8
constexpr uint32_t BYTES_PER_ELEM = 20U;

static inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

// Helper: Create a TilingContextPara for BnInferGrad (NCHW fp32)
// Inputs: grads (NCHW), scale (1D ND), batch_variance (1D ND)
// Output: x_backprop (NCHW, same shape as grads)
static gert::TilingContextPara MakeTilingPara(
    const gert::StorageShape& gradsShape,
    int64_t channelSize,
    ge::DataType dtype = ge::DT_FLOAT,
    ge::Format format = ge::FORMAT_NCHW,
    float epsilon = 0.0001f)
{
    // Attribute: epsilon (index=0)
    std::vector<OpAttr> attrs = {
        OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
    };

    // scale: (C,)
    gert::StorageShape scaleShape({channelSize}, {channelSize});
    // batch_variance: (C,)
    gert::StorageShape varianceShape({channelSize}, {channelSize});

    std::vector<TensorDesc> inputs = {
        TensorDesc(gradsShape, dtype, format),
        TensorDesc(scaleShape, ge::DT_FLOAT, ge::FORMAT_ND),
        TensorDesc(varianceShape, ge::DT_FLOAT, ge::FORMAT_ND),
    };

    // Output: same shape and format as grads
    std::vector<TensorDesc> outputs = {
        TensorDesc(gradsShape, dtype, format),
    };

    return gert::TilingContextPara("BnInferGrad", inputs, outputs, attrs, &g_compileInfo);
}

// Helper: Run tiling and extract BnInferGradTilingData
static bool RunTilingAndGetData(const gert::TilingContextPara& para,
                                TilingInfo& info,
                                BnInferGradTilingData& td)
{
    bool ok = ExecuteTiling(para, info);
    if (!ok) return false;
    if (info.tilingDataSize < sizeof(BnInferGradTilingData)) return false;
    memcpy(&td, info.tilingData.get(), sizeof(BnInferGradTilingData));
    return true;
}

// Helper: Compute expected tileLen for given channelSize and ubSize
static int64_t ComputeExpectedTileLen(int64_t channelSize, int64_t ubSize = 262144)
{
    int64_t alignedC = AlignUp(channelSize, static_cast<int64_t>(ALIGN_ELEM_FP32));
    int64_t overhead = alignedC * 3 * static_cast<int64_t>(FLOAT_SIZE);
    int64_t tileLen = (ubSize - overhead) / static_cast<int64_t>(BYTES_PER_ELEM);
    tileLen = (tileLen / static_cast<int64_t>(ALIGN_ELEM_FP32)) * static_cast<int64_t>(ALIGN_ELEM_FP32);
    return tileLen;
}

// ---------- Iteration 1: NCHW fp32 CONTIGUOUS Core Path ----------

// TC-T-001: Basic NCHW fp32 tiling, shape=(2,3,4,4)
// totalElements=96, C=3, spatial=16, formatMode=0, single tile
TEST(BnInferGradTiling, NCHW_FP32_Basic)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    auto para = MakeTilingPara(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);       // 2*3*4*4
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);         // 4*4
    EXPECT_EQ(td.formatMode, 0);           // NCHW
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.usedCoreNum, 1);          // Iteration 1: single core
    EXPECT_EQ(td.elemsPerCore, 96);
    EXPECT_EQ(td.tailCoreElems, 96);

    // Verify tileLen computation
    int64_t expectedTileLen = ComputeExpectedTileLen(3);
    EXPECT_EQ(td.tileLen, expectedTileLen);
    EXPECT_GT(td.tileLen, 0);

    // totalElements(96) < tileLen -> single tile
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 96);

    // alignedC = AlignUp(3, 8) = 8
    EXPECT_EQ(td.alignedC, 8);

    // Verify epsilon stored as bits
    float epsilon = 0.0001f;
    int64_t expectedBits = 0;
    memcpy(&expectedBits, &epsilon, sizeof(float));
    EXPECT_EQ(td.epsilonBits, expectedBits);

    // Block dim should be 1
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-002: Single element, shape=(1,1,1,1)
// elemsPerCore = AlignUp(1, 8) = 8 after iteration 2 alignment
TEST(BnInferGradTiling, NCHW_FP32_SingleElement)
{
    gert::StorageShape gradsShape({1, 1, 1, 1}, {1, 1, 1, 1});
    auto para = MakeTilingPara(gradsShape, 1);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 1);
    EXPECT_EQ(td.channelSize, 1);
    EXPECT_EQ(td.spatialSize, 1);
    EXPECT_EQ(td.formatMode, 0);
    EXPECT_EQ(td.N, 1);
    EXPECT_EQ(td.usedCoreNum, 1);
    EXPECT_EQ(td.numTiles, 1);
    // elemsPerCore = AlignUp(1, 8) = 8; lastTileLen based on elemsPerCore
    EXPECT_EQ(td.elemsPerCore, 8);
    EXPECT_EQ(td.lastTileLen, 8);
    EXPECT_EQ(td.alignedC, 8);   // AlignUp(1, 8) = 8
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-003: Large spatial, shape=(1,3,32,32)
// totalElements=3072, C=3, spatial=1024
TEST(BnInferGradTiling, NCHW_FP32_LargeSpatial)
{
    gert::StorageShape gradsShape({1, 3, 32, 32}, {1, 3, 32, 32});
    auto para = MakeTilingPara(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 3072);     // 1*3*32*32
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 1024);       // 32*32
    EXPECT_EQ(td.formatMode, 0);
    EXPECT_EQ(td.N, 1);
    EXPECT_EQ(td.usedCoreNum, 1);

    // 3072 < tileLen -> single tile
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 3072);
}

// TC-T-004: Large channel count, shape=(2,64,8,8)
// totalElements=8192, C=64, spatial=64
TEST(BnInferGradTiling, NCHW_FP32_LargeChannel)
{
    gert::StorageShape gradsShape({2, 64, 8, 8}, {2, 64, 8, 8});
    auto para = MakeTilingPara(gradsShape, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8192);     // 2*64*8*8
    EXPECT_EQ(td.channelSize, 64);
    EXPECT_EQ(td.spatialSize, 64);         // 8*8
    EXPECT_EQ(td.formatMode, 0);
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.usedCoreNum, 1);

    // alignedC = AlignUp(64, 8) = 64 (already aligned)
    EXPECT_EQ(td.alignedC, 64);

    // Verify tileLen
    int64_t expectedTileLen = ComputeExpectedTileLen(64);
    EXPECT_EQ(td.tileLen, expectedTileLen);
    EXPECT_GT(td.tileLen, 0);
}

// TC-T-005: Large batch, shape=(8,3,4,4)
// totalElements=384, C=3, spatial=16
TEST(BnInferGradTiling, NCHW_FP32_LargeBatch)
{
    gert::StorageShape gradsShape({8, 3, 4, 4}, {8, 3, 4, 4});
    auto para = MakeTilingPara(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 384);      // 8*3*4*4
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);
    EXPECT_EQ(td.N, 8);
    EXPECT_EQ(td.usedCoreNum, 1);
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 384);
}

// TC-T-006: Very large tensor triggering multi-tile & multi-core, shape=(8,256,64,64)
// totalElements=8388608, should require multiple tiles and multiple cores
TEST(BnInferGradTiling, NCHW_FP32_MultiTile)
{
    gert::StorageShape gradsShape({8, 256, 64, 64}, {8, 256, 64, 64});
    auto para = MakeTilingPara(gradsShape, 256);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8388608);  // 8*256*64*64
    EXPECT_EQ(td.channelSize, 256);
    EXPECT_EQ(td.spatialSize, 4096);       // 64*64
    EXPECT_EQ(td.formatMode, 0);
    EXPECT_EQ(td.N, 8);

    // Iteration 2: multi-core is now enabled
    EXPECT_GT(td.usedCoreNum, 0);

    // Should require multiple tiles per core
    EXPECT_GT(td.numTiles, 1);
    EXPECT_GT(td.tileLen, 0);
    EXPECT_GT(td.lastTileLen, 0);
    EXPECT_LE(td.lastTileLen, td.tileLen);

    // Verify: (numTiles - 1) * tileLen + lastTileLen == elemsPerCore (per-core coverage)
    EXPECT_EQ((td.numTiles - 1) * td.tileLen + td.lastTileLen, td.elemsPerCore);

    // Overall coverage: all cores cover all elements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);
}

// TC-T-007: Non-square spatial, shape=(2,3,4,5)
// totalElements=120, C=3, spatial=20
TEST(BnInferGradTiling, NCHW_FP32_NonSquareSpatial)
{
    gert::StorageShape gradsShape({2, 3, 4, 5}, {2, 3, 4, 5});
    auto para = MakeTilingPara(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 120);      // 2*3*4*5
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 20);         // 4*5
    EXPECT_EQ(td.formatMode, 0);
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 120);
}

// TC-T-008: Spatial = 1x1, shape=(4,128,1,1)
// totalElements=512, C=128, spatial=1
TEST(BnInferGradTiling, NCHW_FP32_Spatial1x1)
{
    gert::StorageShape gradsShape({4, 128, 1, 1}, {4, 128, 1, 1});
    auto para = MakeTilingPara(gradsShape, 128);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 512);      // 4*128*1*1
    EXPECT_EQ(td.channelSize, 128);
    EXPECT_EQ(td.spatialSize, 1);
    EXPECT_EQ(td.formatMode, 0);
    EXPECT_EQ(td.N, 4);
    EXPECT_EQ(td.alignedC, 128);  // AlignUp(128, 8) = 128
}

// TC-T-009: Custom epsilon value
TEST(BnInferGradTiling, NCHW_FP32_CustomEpsilon)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    float customEpsilon = 1e-5f;
    auto para = MakeTilingPara(gradsShape, 3, ge::DT_FLOAT, ge::FORMAT_NCHW, customEpsilon);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);

    // Verify epsilon bits match custom value
    int64_t expectedBits = 0;
    memcpy(&expectedBits, &customEpsilon, sizeof(float));
    EXPECT_EQ(td.epsilonBits, expectedBits);
}

// TC-T-010: Channel not aligned to 8 elements, shape=(2,5,4,4)
// C=5, alignedC = AlignUp(5, 8) = 8
TEST(BnInferGradTiling, NCHW_FP32_UnalignedChannel)
{
    gert::StorageShape gradsShape({2, 5, 4, 4}, {2, 5, 4, 4});
    auto para = MakeTilingPara(gradsShape, 5);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 160);      // 2*5*4*4
    EXPECT_EQ(td.channelSize, 5);
    EXPECT_EQ(td.alignedC, 8);             // AlignUp(5, 8) = 8
    EXPECT_EQ(td.numTiles, 1);
}

// TC-T-011: TileLen and numTiles consistency check for medium tensor
// shape=(4,64,16,16), totalElements=65536
// With iteration 2 multi-core: tiles cover elemsPerCore, not totalElements
TEST(BnInferGradTiling, NCHW_FP32_TileConsistency)
{
    gert::StorageShape gradsShape({4, 64, 16, 16}, {4, 64, 16, 16});
    auto para = MakeTilingPara(gradsShape, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 65536);    // 4*64*16*16

    // Verify tile consistency per core:
    // (numTiles - 1) * tileLen + lastTileLen == elemsPerCore
    EXPECT_EQ((td.numTiles - 1) * td.tileLen + td.lastTileLen, td.elemsPerCore);

    // Verify overall core distribution:
    // (usedCoreNum - 1) * elemsPerCore + tailCoreElems == totalElements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);

    // tileLen should be aligned to ALIGN_ELEM_FP32 (8)
    EXPECT_EQ(td.tileLen % 8, 0);

    // lastTileLen should be > 0 and <= tileLen
    EXPECT_GT(td.lastTileLen, 0);
    EXPECT_LE(td.lastTileLen, td.tileLen);
}

// TC-T-012: NC1HWC0 and NHWC should be rejected or handled differently
// (This test is for coverage - NC1HWC0 should get schMode=1)
// Iteration 2: with multi-core, usedCoreNum may be > 1
TEST(BnInferGradTiling, NCHW_FP32_BatchOne)
{
    gert::StorageShape gradsShape({1, 256, 8, 8}, {1, 256, 8, 8});
    auto para = MakeTilingPara(gradsShape, 256);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 16384);    // 1*256*8*8
    EXPECT_EQ(td.channelSize, 256);
    EXPECT_EQ(td.spatialSize, 64);         // 8*8
    EXPECT_EQ(td.N, 1);
    EXPECT_EQ(td.formatMode, 0);           // NCHW -> CONTIGUOUS
    EXPECT_GT(td.usedCoreNum, 0);
}

// ==========================================================================
// Iteration 2: NHWC format Tiling (FORMAT_NHWC -> CONTIGUOUS, formatMode=1)
// ==========================================================================

// Helper: Create a TilingContextPara for NHWC format
// NHWC shape: (N, H, W, C). scale/variance: (C,)
static gert::TilingContextPara MakeTilingParaNHWC(
    const gert::StorageShape& gradsShape,
    int64_t channelSize,
    ge::DataType dtype = ge::DT_FLOAT,
    float epsilon = 0.0001f,
    uint64_t coreNum = 64,
    uint64_t ubSize = 262144)
{
    std::vector<OpAttr> attrs = {
        OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
    };

    gert::StorageShape scaleShape({channelSize}, {channelSize});
    gert::StorageShape varianceShape({channelSize}, {channelSize});

    std::vector<TensorDesc> inputs = {
        TensorDesc(gradsShape, dtype, ge::FORMAT_NHWC),
        TensorDesc(scaleShape, ge::DT_FLOAT, ge::FORMAT_ND),
        TensorDesc(varianceShape, ge::DT_FLOAT, ge::FORMAT_ND),
    };

    std::vector<TensorDesc> outputs = {
        TensorDesc(gradsShape, dtype, ge::FORMAT_NHWC),
    };

    return gert::TilingContextPara("BnInferGrad", inputs, outputs, attrs, &g_compileInfo,
                                    coreNum, ubSize);
}

// Helper: Create a TilingContextPara for NC1HWC0 format
// NC1HWC0 shape: (N, C1, H, W, C0). scale/variance shape: (C1*C0,)
static gert::TilingContextPara MakeTilingParaNC1HWC0(
    const gert::StorageShape& gradsShape,
    int64_t channelSize,
    ge::DataType dtype = ge::DT_FLOAT,
    float epsilon = 0.0001f,
    uint64_t coreNum = 64,
    uint64_t ubSize = 262144)
{
    std::vector<OpAttr> attrs = {
        OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
    };

    gert::StorageShape scaleShape({channelSize}, {channelSize});
    gert::StorageShape varianceShape({channelSize}, {channelSize});

    std::vector<TensorDesc> inputs = {
        TensorDesc(gradsShape, dtype, ge::FORMAT_NC1HWC0),
        TensorDesc(scaleShape, ge::DT_FLOAT, ge::FORMAT_ND),
        TensorDesc(varianceShape, ge::DT_FLOAT, ge::FORMAT_ND),
    };

    std::vector<TensorDesc> outputs = {
        TensorDesc(gradsShape, dtype, ge::FORMAT_NC1HWC0),
    };

    return gert::TilingContextPara("BnInferGrad", inputs, outputs, attrs, &g_compileInfo,
                                    coreNum, ubSize);
}

// Helper: Create NCHW TilingContextPara with custom coreNum/ubSize
static gert::TilingContextPara MakeTilingParaEx(
    const gert::StorageShape& gradsShape,
    int64_t channelSize,
    ge::DataType dtype = ge::DT_FLOAT,
    ge::Format format = ge::FORMAT_NCHW,
    float epsilon = 0.0001f,
    uint64_t coreNum = 64,
    uint64_t ubSize = 262144)
{
    std::vector<OpAttr> attrs = {
        OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(epsilon)),
    };

    gert::StorageShape scaleShape({channelSize}, {channelSize});
    gert::StorageShape varianceShape({channelSize}, {channelSize});

    std::vector<TensorDesc> inputs = {
        TensorDesc(gradsShape, dtype, format),
        TensorDesc(scaleShape, ge::DT_FLOAT, ge::FORMAT_ND),
        TensorDesc(varianceShape, ge::DT_FLOAT, ge::FORMAT_ND),
    };

    std::vector<TensorDesc> outputs = {
        TensorDesc(gradsShape, dtype, format),
    };

    return gert::TilingContextPara("BnInferGrad", inputs, outputs, attrs, &g_compileInfo,
                                    coreNum, ubSize);
}

// TC-T-013: NHWC basic, shape=(2,4,4,3), C=3, formatMode=1, CONTIGUOUS branch
TEST(BnInferGradTiling, NHWC_FP32_Basic)
{
    gert::StorageShape gradsShape({2, 4, 4, 3}, {2, 4, 4, 3});
    auto para = MakeTilingParaNHWC(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);       // 2*4*4*3
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);         // 4*4
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.alignedC, 8);            // AlignUp(3, 8) = 8

    // NHWC tileLen should be aligned to channelSize(3) when possible
    EXPECT_GT(td.tileLen, 0);

    // 96 elements fits in single tile
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 96);

    // Epsilon bits should match
    float epsilon = 0.0001f;
    int64_t expectedBits = 0;
    memcpy(&expectedBits, &epsilon, sizeof(float));
    EXPECT_EQ(td.epsilonBits, expectedBits);
}

// TC-T-014: NHWC with larger channel, shape=(2,8,8,64), C=64
TEST(BnInferGradTiling, NHWC_FP32_LargeChannel)
{
    gert::StorageShape gradsShape({2, 8, 8, 64}, {2, 8, 8, 64});
    auto para = MakeTilingParaNHWC(gradsShape, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8192);     // 2*8*8*64
    EXPECT_EQ(td.channelSize, 64);
    EXPECT_EQ(td.spatialSize, 64);         // 8*8
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.alignedC, 64);           // AlignUp(64, 8) = 64

    // Verify tileLen is aligned to channelSize (64)
    EXPECT_EQ(td.tileLen % 64, 0);
    EXPECT_GT(td.tileLen, 0);
}

// TC-T-015: NHWC with unaligned channel, shape=(1,4,4,5), C=5
// tileLen aligned to C=5 may fail; should fallback to ALIGN_ELEM_FP32
TEST(BnInferGradTiling, NHWC_FP32_UnalignedChannel)
{
    gert::StorageShape gradsShape({1, 4, 4, 5}, {1, 4, 4, 5});
    auto para = MakeTilingParaNHWC(gradsShape, 5);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 80);       // 1*4*4*5
    EXPECT_EQ(td.channelSize, 5);
    EXPECT_EQ(td.spatialSize, 16);         // 4*4
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 1);

    // tileLen should be > 0 and aligned to channelSize(5) if possible
    EXPECT_GT(td.tileLen, 0);

    // Single tile for small tensor
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 80);
}

// TC-T-016: NHWC single element, shape=(1,1,1,1), C=1
// elemsPerCore = AlignUp(1, 8) = 8 after iteration 2 alignment
TEST(BnInferGradTiling, NHWC_FP32_SingleElement)
{
    gert::StorageShape gradsShape({1, 1, 1, 1}, {1, 1, 1, 1});
    auto para = MakeTilingParaNHWC(gradsShape, 1);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 1);
    EXPECT_EQ(td.channelSize, 1);
    EXPECT_EQ(td.spatialSize, 1);
    EXPECT_EQ(td.formatMode, 1);
    EXPECT_EQ(td.N, 1);
    EXPECT_EQ(td.numTiles, 1);
    // elemsPerCore = AlignUp(1, 8) = 8; lastTileLen based on elemsPerCore
    EXPECT_EQ(td.elemsPerCore, 8);
    EXPECT_EQ(td.lastTileLen, 8);
}

// TC-T-017: NHWC large tensor multi-tile, shape=(8,64,64,128), C=128
// totalElements=4194304, should require multiple tiles and multiple cores
TEST(BnInferGradTiling, NHWC_FP32_MultiTile)
{
    gert::StorageShape gradsShape({8, 64, 64, 128}, {8, 64, 64, 128});
    auto para = MakeTilingParaNHWC(gradsShape, 128);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 4194304);  // 8*64*64*128
    EXPECT_EQ(td.channelSize, 128);
    EXPECT_EQ(td.spatialSize, 4096);       // 64*64
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 8);

    // Should use multiple tiles per core
    EXPECT_GT(td.numTiles, 0);
    EXPECT_GT(td.tileLen, 0);
    EXPECT_GT(td.lastTileLen, 0);
    EXPECT_LE(td.lastTileLen, td.tileLen);

    // tileLen should be aligned to channelSize(128)
    EXPECT_EQ(td.tileLen % 128, 0);

    // Tile consistency: covers elemsPerCore
    EXPECT_EQ((td.numTiles - 1) * td.tileLen + td.lastTileLen, td.elemsPerCore);

    // Core distribution covers totalElements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);
}

// ==========================================================================
// Iteration 2: NC1HWC0 format Tiling (FORMAT_NC1HWC0, formatMode=2)
// ==========================================================================

// TC-T-018: NC1HWC0 basic, shape=(2,4,8,8,16), C1=4, C0=16, C=64
// schMode=SCH_NC1HWC0(1), totalTasks=N*C1=8
TEST(BnInferGradTiling, NC1HWC0_FP32_Basic)
{
    // NC1HWC0: (N, C1, H, W, C0)
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    int64_t C = 4 * 16;  // C1*C0 = 64
    auto para = MakeTilingParaNC1HWC0(gradsShape, C);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 2 * 4 * 8 * 8 * 16);  // 8192
    EXPECT_EQ(td.channelSize, 64);          // C1*C0
    EXPECT_EQ(td.spatialSize, 64);          // H*W = 8*8
    EXPECT_EQ(td.formatMode, 2);            // NC1HWC0
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.C1, 4);
    EXPECT_EQ(td.C0, 16);

    // NC1HWC0-specific fields
    EXPECT_EQ(td.totalTasks, 8);            // N*C1 = 2*4
    EXPECT_GT(td.tasksPerCore, 0);
    EXPECT_GT(td.tailCoreTasks, 0);
    EXPECT_LE(td.tailCoreTasks, td.tasksPerCore);

    // tileHW and numTilesHW
    EXPECT_GT(td.tileHW, 0);
    EXPECT_LE(td.tileHW, 64);              // tileHW <= spatialSize
    EXPECT_GT(td.numTilesHW, 0);
    EXPECT_GT(td.lastTileHW, 0);
    EXPECT_LE(td.lastTileHW, td.tileHW);

    // Verify numTilesHW consistency
    // (numTilesHW - 1) * tileHW + lastTileHW == spatialSize
    EXPECT_EQ((td.numTilesHW - 1) * td.tileHW + td.lastTileHW, td.spatialSize);

    // alignedC0 = AlignUp(16, 8) = 16
    EXPECT_EQ(td.alignedC0, 16);

    // alignedC = AlignUp(64, 8) = 64
    EXPECT_EQ(td.alignedC, 64);

    // tileLen should be tileHW * C0
    EXPECT_EQ(td.tileLen, td.tileHW * td.C0);

    // usedCoreNum <= totalTasks
    EXPECT_LE(td.usedCoreNum, td.totalTasks);
    EXPECT_GT(td.usedCoreNum, 0);

    // blockDim should equal usedCoreNum
    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-019: NC1HWC0 single task, shape=(1,1,4,4,16), C1=1, C0=16
// totalTasks = N*C1 = 1
TEST(BnInferGradTiling, NC1HWC0_FP32_SingleTask)
{
    gert::StorageShape gradsShape({1, 1, 4, 4, 16}, {1, 1, 4, 4, 16});
    int64_t C = 1 * 16;  // 16
    auto para = MakeTilingParaNC1HWC0(gradsShape, C);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 256);       // 1*1*4*4*16
    EXPECT_EQ(td.channelSize, 16);
    EXPECT_EQ(td.spatialSize, 16);          // 4*4
    EXPECT_EQ(td.formatMode, 2);
    EXPECT_EQ(td.N, 1);
    EXPECT_EQ(td.C1, 1);
    EXPECT_EQ(td.C0, 16);
    EXPECT_EQ(td.totalTasks, 1);            // N*C1 = 1
    EXPECT_EQ(td.usedCoreNum, 1);
    EXPECT_EQ(td.tasksPerCore, 1);
    EXPECT_EQ(td.tailCoreTasks, 1);

    // tileHW should cover all spatial positions since small
    EXPECT_EQ(td.tileHW, 16);              // spatialSize fits in UB
    EXPECT_EQ(td.numTilesHW, 1);
    EXPECT_EQ(td.lastTileHW, 16);

    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-020: NC1HWC0 large spatial, shape=(1,2,64,64,16), C1=2, C0=16
// spatialSize = 4096, may require multiple HW tiles
TEST(BnInferGradTiling, NC1HWC0_FP32_LargeSpatial)
{
    gert::StorageShape gradsShape({1, 2, 64, 64, 16}, {1, 2, 64, 64, 16});
    int64_t C = 2 * 16;  // 32
    auto para = MakeTilingParaNC1HWC0(gradsShape, C);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 1 * 2 * 64 * 64 * 16);  // 131072
    EXPECT_EQ(td.channelSize, 32);
    EXPECT_EQ(td.spatialSize, 4096);        // 64*64
    EXPECT_EQ(td.formatMode, 2);
    EXPECT_EQ(td.C1, 2);
    EXPECT_EQ(td.C0, 16);
    EXPECT_EQ(td.totalTasks, 2);            // N*C1 = 1*2

    // May have multiple HW tiles
    EXPECT_GT(td.tileHW, 0);
    EXPECT_GT(td.numTilesHW, 0);
    EXPECT_GT(td.lastTileHW, 0);

    // Verify HW tile consistency
    EXPECT_EQ((td.numTilesHW - 1) * td.tileHW + td.lastTileHW, td.spatialSize);

    // tileLen = tileHW * C0
    EXPECT_EQ(td.tileLen, td.tileHW * td.C0);

    // alignedC0 = AlignUp(16, 8) = 16
    EXPECT_EQ(td.alignedC0, 16);
}

// TC-T-021: NC1HWC0 many tasks, shape=(4,8,4,4,16), N=4, C1=8
// totalTasks = 32 (distributed across multiple cores)
TEST(BnInferGradTiling, NC1HWC0_FP32_ManyTasks)
{
    gert::StorageShape gradsShape({4, 8, 4, 4, 16}, {4, 8, 4, 4, 16});
    int64_t C = 8 * 16;  // 128
    auto para = MakeTilingParaNC1HWC0(gradsShape, C);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 4 * 8 * 4 * 4 * 16);  // 32768
    EXPECT_EQ(td.channelSize, 128);
    EXPECT_EQ(td.spatialSize, 16);          // 4*4
    EXPECT_EQ(td.formatMode, 2);
    EXPECT_EQ(td.N, 4);
    EXPECT_EQ(td.C1, 8);
    EXPECT_EQ(td.C0, 16);
    EXPECT_EQ(td.totalTasks, 32);           // N*C1 = 4*8

    // With coreNum=64 default, usedCoreNum <= totalTasks=32
    EXPECT_LE(td.usedCoreNum, 32);
    EXPECT_GT(td.usedCoreNum, 0);

    // Verify task distribution
    // (usedCoreNum - 1) * tasksPerCore + tailCoreTasks == totalTasks
    EXPECT_EQ((td.usedCoreNum - 1) * td.tasksPerCore + td.tailCoreTasks, td.totalTasks);

    // blockDim == usedCoreNum
    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-022: NC1HWC0 with C0=32, shape=(1,2,4,4,32), C1=2, C0=32
TEST(BnInferGradTiling, NC1HWC0_FP32_C0_32)
{
    gert::StorageShape gradsShape({1, 2, 4, 4, 32}, {1, 2, 4, 4, 32});
    int64_t C = 2 * 32;  // 64
    auto para = MakeTilingParaNC1HWC0(gradsShape, C);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 1 * 2 * 4 * 4 * 32);  // 1024
    EXPECT_EQ(td.channelSize, 64);
    EXPECT_EQ(td.spatialSize, 16);          // 4*4
    EXPECT_EQ(td.formatMode, 2);
    EXPECT_EQ(td.C1, 2);
    EXPECT_EQ(td.C0, 32);
    EXPECT_EQ(td.totalTasks, 2);            // N*C1 = 1*2

    // alignedC0 = AlignUp(32, 8) = 32
    EXPECT_EQ(td.alignedC0, 32);

    // tileLen = tileHW * C0
    EXPECT_EQ(td.tileLen, td.tileHW * td.C0);
}

// ==========================================================================
// Iteration 2: Multi-core split logic
// ==========================================================================

// TC-T-023: NCHW multi-core - large tensor triggers multi-core
// shape=(16,256,64,64), totalElements=16777216, should use multiple cores
TEST(BnInferGradTiling, NCHW_FP32_MultiCore)
{
    gert::StorageShape gradsShape({16, 256, 64, 64}, {16, 256, 64, 64});
    auto para = MakeTilingParaEx(gradsShape, 256, ge::DT_FLOAT, ge::FORMAT_NCHW, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 16777216);  // 16*256*64*64
    EXPECT_EQ(td.formatMode, 0);            // NCHW

    // Should use multiple cores (totalElements >> tileLen)
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 64);

    // Verify core distribution
    // (usedCoreNum - 1) * elemsPerCore + tailCoreElems == totalElements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);

    // elemsPerCore should be aligned to 8
    EXPECT_EQ(td.elemsPerCore % 8, 0);

    // tailCoreElems <= elemsPerCore
    EXPECT_GT(td.tailCoreElems, 0);
    EXPECT_LE(td.tailCoreElems, td.elemsPerCore);

    // blockDim should match usedCoreNum
    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-024: NCHW multi-core with limited core count (coreNum=4)
// shape=(8,256,64,64), totalElements=8388608
TEST(BnInferGradTiling, NCHW_FP32_MultiCoreLimitedCores)
{
    gert::StorageShape gradsShape({8, 256, 64, 64}, {8, 256, 64, 64});
    auto para = MakeTilingParaEx(gradsShape, 256, ge::DT_FLOAT, ge::FORMAT_NCHW, 0.0001f, 4);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8388608);
    EXPECT_EQ(td.formatMode, 0);

    // Should use at most 4 cores
    EXPECT_LE(td.usedCoreNum, 4);
    EXPECT_GT(td.usedCoreNum, 0);

    // Verify distribution
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);
    EXPECT_EQ(td.elemsPerCore % 8, 0);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-025: NCHW single core - small tensor stays on one core
// shape=(1,3,4,4), totalElements=48, only 1 core needed
TEST(BnInferGradTiling, NCHW_FP32_SingleCoreSmalltensor)
{
    gert::StorageShape gradsShape({1, 3, 4, 4}, {1, 3, 4, 4});
    auto para = MakeTilingParaEx(gradsShape, 3, ge::DT_FLOAT, ge::FORMAT_NCHW, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 48);
    EXPECT_EQ(td.usedCoreNum, 1);
    EXPECT_EQ(td.elemsPerCore, 48);
    EXPECT_EQ(td.tailCoreElems, 48);
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-026: NHWC multi-core - large tensor triggers multi-core
// shape=(8,64,64,256), totalElements=8388608
TEST(BnInferGradTiling, NHWC_FP32_MultiCore)
{
    gert::StorageShape gradsShape({8, 64, 64, 256}, {8, 64, 64, 256});
    auto para = MakeTilingParaNHWC(gradsShape, 256, ge::DT_FLOAT, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8388608);   // 8*64*64*256
    EXPECT_EQ(td.channelSize, 256);
    EXPECT_EQ(td.formatMode, 1);            // NHWC

    // Should use multiple cores
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 64);

    // Verify core distribution
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);
    EXPECT_EQ(td.elemsPerCore % 8, 0);
    EXPECT_GT(td.tailCoreElems, 0);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-027: NC1HWC0 multi-core - many tasks distributed across cores
// shape=(8,16,8,8,16), N=8, C1=16, totalTasks=128, coreNum=64
TEST(BnInferGradTiling, NC1HWC0_FP32_MultiCore)
{
    gert::StorageShape gradsShape({8, 16, 8, 8, 16}, {8, 16, 8, 8, 16});
    int64_t C = 16 * 16;  // 256
    auto para = MakeTilingParaNC1HWC0(gradsShape, C, ge::DT_FLOAT, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8 * 16 * 8 * 8 * 16);  // 131072
    EXPECT_EQ(td.formatMode, 2);
    EXPECT_EQ(td.totalTasks, 128);          // N*C1 = 8*16

    // With 128 tasks and coreNum=64, should use multiple cores
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 64);

    // Verify task distribution
    EXPECT_EQ((td.usedCoreNum - 1) * td.tasksPerCore + td.tailCoreTasks, td.totalTasks);
    EXPECT_GT(td.tasksPerCore, 0);
    EXPECT_GT(td.tailCoreTasks, 0);
    EXPECT_LE(td.tailCoreTasks, td.tasksPerCore);

    // blockDim == usedCoreNum
    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-028: NC1HWC0 multi-core with fewer tasks than cores
// shape=(2,3,4,4,16), N=2, C1=3, totalTasks=6, coreNum=64
// usedCoreNum should be limited to totalTasks
TEST(BnInferGradTiling, NC1HWC0_FP32_FewerTasksThanCores)
{
    gert::StorageShape gradsShape({2, 3, 4, 4, 16}, {2, 3, 4, 4, 16});
    int64_t C = 3 * 16;  // 48
    auto para = MakeTilingParaNC1HWC0(gradsShape, C, ge::DT_FLOAT, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalTasks, 6);            // N*C1 = 2*3
    EXPECT_LE(td.usedCoreNum, 6);           // Cannot exceed totalTasks
    EXPECT_GT(td.usedCoreNum, 0);

    // Verify task distribution
    EXPECT_EQ((td.usedCoreNum - 1) * td.tasksPerCore + td.tailCoreTasks, td.totalTasks);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-029: NC1HWC0 elemsPerCore and tailCoreElems consistency
// These should match tasksPerCore * spatialSize * C0
TEST(BnInferGradTiling, NC1HWC0_FP32_ElemsPerCoreConsistency)
{
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    int64_t C = 4 * 16;  // 64
    auto para = MakeTilingParaNC1HWC0(gradsShape, C, ge::DT_FLOAT, 0.0001f, 4);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalTasks, 8);            // N*C1 = 2*4
    EXPECT_EQ(td.formatMode, 2);

    // elemsPerCore = tasksPerCore * spatialSize * C0
    EXPECT_EQ(td.elemsPerCore, td.tasksPerCore * td.spatialSize * td.C0);

    // tailCoreElems = tailCoreTasks * spatialSize * C0
    EXPECT_EQ(td.tailCoreElems, td.tailCoreTasks * td.spatialSize * td.C0);

    // numTiles == numTilesHW (compatibility)
    EXPECT_EQ(td.numTiles, td.numTilesHW);

    // lastTileLen = lastTileHW * C0
    EXPECT_EQ(td.lastTileLen, td.lastTileHW * td.C0);
}

// TC-T-030: NCHW multi-core tile consistency
// Verify that (numTiles - 1) * tileLen + lastTileLen == elemsPerCore (per-core coverage)
TEST(BnInferGradTiling, NCHW_FP32_MultiCoreTileConsistency)
{
    gert::StorageShape gradsShape({8, 128, 32, 32}, {8, 128, 32, 32});
    auto para = MakeTilingParaEx(gradsShape, 128, ge::DT_FLOAT, ge::FORMAT_NCHW, 0.0001f, 32);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8 * 128 * 32 * 32);  // 4194304

    // numTiles is based on elemsPerCore, not totalElements
    // (numTiles - 1) * tileLen + lastTileLen should cover elemsPerCore
    EXPECT_EQ((td.numTiles - 1) * td.tileLen + td.lastTileLen, td.elemsPerCore);

    // tileLen should be aligned to 8
    EXPECT_EQ(td.tileLen % 8, 0);

    // Overall: all cores cover all elements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);
}

// ==========================================================================
// Iteration 3: fp16 data type Tiling
// The tiling logic uses BYTES_PER_ELEM=20 for all dtypes, so tiling results
// should be identical to fp32 for the same shape. The dtype only affects the
// TilingKey template parameter (dTypeX).
// ==========================================================================

// TC-T-031: NCHW fp16 basic, shape=(2,3,4,4)
// Verify fp16 tiling produces same tiling parameters as fp32
TEST(BnInferGradTiling, NCHW_FP16_Basic)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    auto para = MakeTilingParaEx(gradsShape, 3, ge::DT_FLOAT16, ge::FORMAT_NCHW);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);       // 2*3*4*4
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);         // 4*4
    EXPECT_EQ(td.formatMode, 0);           // NCHW
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.usedCoreNum, 1);
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 96);
    EXPECT_EQ(td.alignedC, 8);            // AlignUp(3, 8) = 8
    EXPECT_GT(td.tileLen, 0);

    // Verify epsilon stored as bits
    float epsilon = 0.0001f;
    int64_t expectedBits = 0;
    memcpy(&expectedBits, &epsilon, sizeof(float));
    EXPECT_EQ(td.epsilonBits, expectedBits);

    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-032: NCHW fp16 large tensor multi-tile multi-core, shape=(8,256,64,64)
TEST(BnInferGradTiling, NCHW_FP16_MultiTileMultiCore)
{
    gert::StorageShape gradsShape({8, 256, 64, 64}, {8, 256, 64, 64});
    auto para = MakeTilingParaEx(gradsShape, 256, ge::DT_FLOAT16, ge::FORMAT_NCHW, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8388608);  // 8*256*64*64
    EXPECT_EQ(td.channelSize, 256);
    EXPECT_EQ(td.formatMode, 0);

    // Multi-core
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 64);

    // Multi-tile
    EXPECT_GT(td.numTiles, 1);
    EXPECT_GT(td.tileLen, 0);
    EXPECT_GT(td.lastTileLen, 0);
    EXPECT_LE(td.lastTileLen, td.tileLen);

    // Tile consistency per core
    EXPECT_EQ((td.numTiles - 1) * td.tileLen + td.lastTileLen, td.elemsPerCore);

    // Core distribution covers all elements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-033: NHWC fp16 basic, shape=(2,4,4,3), C=3
TEST(BnInferGradTiling, NHWC_FP16_Basic)
{
    gert::StorageShape gradsShape({2, 4, 4, 3}, {2, 4, 4, 3});
    auto para = MakeTilingParaEx(gradsShape, 3, ge::DT_FLOAT16, ge::FORMAT_NHWC);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);       // 2*4*4*3
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);         // 4*4
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 96);
    EXPECT_GT(td.tileLen, 0);
}

// TC-T-034: NC1HWC0 fp16 basic, shape=(2,4,8,8,16), C1=4, C0=16
TEST(BnInferGradTiling, NC1HWC0_FP16_Basic)
{
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    int64_t C = 4 * 16;  // 64
    auto para = MakeTilingParaEx(gradsShape, C, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 2 * 4 * 8 * 8 * 16);  // 8192
    EXPECT_EQ(td.channelSize, 64);
    EXPECT_EQ(td.spatialSize, 64);          // H*W = 8*8
    EXPECT_EQ(td.formatMode, 2);            // NC1HWC0
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.C1, 4);
    EXPECT_EQ(td.C0, 16);

    // NC1HWC0-specific fields
    EXPECT_EQ(td.totalTasks, 8);            // N*C1 = 2*4
    EXPECT_GT(td.tasksPerCore, 0);
    EXPECT_GT(td.tileHW, 0);
    EXPECT_LE(td.tileHW, 64);

    // tileLen = tileHW * C0
    EXPECT_EQ(td.tileLen, td.tileHW * td.C0);

    // HW tile consistency
    EXPECT_EQ((td.numTilesHW - 1) * td.tileHW + td.lastTileHW, td.spatialSize);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// ==========================================================================
// Iteration 3: bf16 data type Tiling
// ==========================================================================

// TC-T-035: NCHW bf16 basic, shape=(2,3,4,4)
TEST(BnInferGradTiling, NCHW_BF16_Basic)
{
    gert::StorageShape gradsShape({2, 3, 4, 4}, {2, 3, 4, 4});
    auto para = MakeTilingParaEx(gradsShape, 3, ge::DT_BF16, ge::FORMAT_NCHW);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);       // 2*3*4*4
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);
    EXPECT_EQ(td.formatMode, 0);           // NCHW
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.usedCoreNum, 1);
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 96);
    EXPECT_EQ(td.alignedC, 8);            // AlignUp(3, 8) = 8
    EXPECT_GT(td.tileLen, 0);

    // Verify epsilon stored as bits
    float epsilon = 0.0001f;
    int64_t expectedBits = 0;
    memcpy(&expectedBits, &epsilon, sizeof(float));
    EXPECT_EQ(td.epsilonBits, expectedBits);

    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-036: NCHW bf16 large tensor multi-tile multi-core, shape=(8,256,64,64)
TEST(BnInferGradTiling, NCHW_BF16_MultiTileMultiCore)
{
    gert::StorageShape gradsShape({8, 256, 64, 64}, {8, 256, 64, 64});
    auto para = MakeTilingParaEx(gradsShape, 256, ge::DT_BF16, ge::FORMAT_NCHW, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 8388608);
    EXPECT_EQ(td.channelSize, 256);
    EXPECT_EQ(td.formatMode, 0);

    // Multi-core
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 64);

    // Multi-tile
    EXPECT_GT(td.numTiles, 1);
    EXPECT_GT(td.tileLen, 0);
    EXPECT_GT(td.lastTileLen, 0);
    EXPECT_LE(td.lastTileLen, td.tileLen);

    // Tile consistency per core
    EXPECT_EQ((td.numTiles - 1) * td.tileLen + td.lastTileLen, td.elemsPerCore);

    // Core distribution covers all elements
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-037: NHWC bf16 basic, shape=(2,4,4,3), C=3
TEST(BnInferGradTiling, NHWC_BF16_Basic)
{
    gert::StorageShape gradsShape({2, 4, 4, 3}, {2, 4, 4, 3});
    auto para = MakeTilingParaEx(gradsShape, 3, ge::DT_BF16, ge::FORMAT_NHWC);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 96);
    EXPECT_EQ(td.channelSize, 3);
    EXPECT_EQ(td.spatialSize, 16);
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 96);
    EXPECT_GT(td.tileLen, 0);
}

// TC-T-038: NC1HWC0 bf16 basic, shape=(2,4,8,8,16), C1=4, C0=16
TEST(BnInferGradTiling, NC1HWC0_BF16_Basic)
{
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    int64_t C = 4 * 16;  // 64
    auto para = MakeTilingParaEx(gradsShape, C, ge::DT_BF16, ge::FORMAT_NC1HWC0);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 2 * 4 * 8 * 8 * 16);  // 8192
    EXPECT_EQ(td.channelSize, 64);
    EXPECT_EQ(td.spatialSize, 64);
    EXPECT_EQ(td.formatMode, 2);            // NC1HWC0
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.C1, 4);
    EXPECT_EQ(td.C0, 16);

    // NC1HWC0-specific fields
    EXPECT_EQ(td.totalTasks, 8);
    EXPECT_GT(td.tasksPerCore, 0);
    EXPECT_GT(td.tileHW, 0);
    EXPECT_LE(td.tileHW, 64);

    // tileLen = tileHW * C0
    EXPECT_EQ(td.tileLen, td.tileHW * td.C0);

    // HW tile consistency
    EXPECT_EQ((td.numTilesHW - 1) * td.tileHW + td.lastTileHW, td.spatialSize);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// ==========================================================================
// Iteration 3: fp16/bf16 tiling equivalence with fp32
// Verify that tiling parameters are identical across all dtypes
// for the same shape (since BYTES_PER_ELEM=20 is the same for all).
// ==========================================================================

// TC-T-039: Cross-dtype equivalence for NCHW, shape=(4,64,16,16)
TEST(BnInferGradTiling, NCHW_CrossDtype_Equivalence)
{
    gert::StorageShape gradsShape({4, 64, 16, 16}, {4, 64, 16, 16});

    // fp32 baseline
    auto paraFP32 = MakeTilingParaEx(gradsShape, 64, ge::DT_FLOAT, ge::FORMAT_NCHW);
    TilingInfo infoFP32;
    BnInferGradTilingData tdFP32;
    ASSERT_TRUE(RunTilingAndGetData(paraFP32, infoFP32, tdFP32));

    // fp16
    auto paraFP16 = MakeTilingParaEx(gradsShape, 64, ge::DT_FLOAT16, ge::FORMAT_NCHW);
    TilingInfo infoFP16;
    BnInferGradTilingData tdFP16;
    ASSERT_TRUE(RunTilingAndGetData(paraFP16, infoFP16, tdFP16));

    // bf16
    auto paraBF16 = MakeTilingParaEx(gradsShape, 64, ge::DT_BF16, ge::FORMAT_NCHW);
    TilingInfo infoBF16;
    BnInferGradTilingData tdBF16;
    ASSERT_TRUE(RunTilingAndGetData(paraBF16, infoBF16, tdBF16));

    // All should produce identical tiling parameters
    EXPECT_EQ(tdFP32.totalElements, tdFP16.totalElements);
    EXPECT_EQ(tdFP32.totalElements, tdBF16.totalElements);
    EXPECT_EQ(tdFP32.tileLen, tdFP16.tileLen);
    EXPECT_EQ(tdFP32.tileLen, tdBF16.tileLen);
    EXPECT_EQ(tdFP32.numTiles, tdFP16.numTiles);
    EXPECT_EQ(tdFP32.numTiles, tdBF16.numTiles);
    EXPECT_EQ(tdFP32.lastTileLen, tdFP16.lastTileLen);
    EXPECT_EQ(tdFP32.lastTileLen, tdBF16.lastTileLen);
    EXPECT_EQ(tdFP32.usedCoreNum, tdFP16.usedCoreNum);
    EXPECT_EQ(tdFP32.usedCoreNum, tdBF16.usedCoreNum);
    EXPECT_EQ(tdFP32.elemsPerCore, tdFP16.elemsPerCore);
    EXPECT_EQ(tdFP32.elemsPerCore, tdBF16.elemsPerCore);
    EXPECT_EQ(tdFP32.tailCoreElems, tdFP16.tailCoreElems);
    EXPECT_EQ(tdFP32.tailCoreElems, tdBF16.tailCoreElems);

    // Block dims should match too
    EXPECT_EQ(infoFP32.blockNum, infoFP16.blockNum);
    EXPECT_EQ(infoFP32.blockNum, infoBF16.blockNum);
}

// TC-T-040: Cross-dtype equivalence for NC1HWC0, shape=(2,4,8,8,16)
TEST(BnInferGradTiling, NC1HWC0_CrossDtype_Equivalence)
{
    gert::StorageShape gradsShape({2, 4, 8, 8, 16}, {2, 4, 8, 8, 16});
    int64_t C = 4 * 16;  // 64

    // fp32 baseline
    auto paraFP32 = MakeTilingParaEx(gradsShape, C, ge::DT_FLOAT, ge::FORMAT_NC1HWC0);
    TilingInfo infoFP32;
    BnInferGradTilingData tdFP32;
    ASSERT_TRUE(RunTilingAndGetData(paraFP32, infoFP32, tdFP32));

    // fp16
    auto paraFP16 = MakeTilingParaEx(gradsShape, C, ge::DT_FLOAT16, ge::FORMAT_NC1HWC0);
    TilingInfo infoFP16;
    BnInferGradTilingData tdFP16;
    ASSERT_TRUE(RunTilingAndGetData(paraFP16, infoFP16, tdFP16));

    // bf16
    auto paraBF16 = MakeTilingParaEx(gradsShape, C, ge::DT_BF16, ge::FORMAT_NC1HWC0);
    TilingInfo infoBF16;
    BnInferGradTilingData tdBF16;
    ASSERT_TRUE(RunTilingAndGetData(paraBF16, infoBF16, tdBF16));

    // All should produce identical tiling parameters
    EXPECT_EQ(tdFP32.totalTasks, tdFP16.totalTasks);
    EXPECT_EQ(tdFP32.totalTasks, tdBF16.totalTasks);
    EXPECT_EQ(tdFP32.tileHW, tdFP16.tileHW);
    EXPECT_EQ(tdFP32.tileHW, tdBF16.tileHW);
    EXPECT_EQ(tdFP32.numTilesHW, tdFP16.numTilesHW);
    EXPECT_EQ(tdFP32.numTilesHW, tdBF16.numTilesHW);
    EXPECT_EQ(tdFP32.lastTileHW, tdFP16.lastTileHW);
    EXPECT_EQ(tdFP32.lastTileHW, tdBF16.lastTileHW);
    EXPECT_EQ(tdFP32.usedCoreNum, tdFP16.usedCoreNum);
    EXPECT_EQ(tdFP32.usedCoreNum, tdBF16.usedCoreNum);

    EXPECT_EQ(infoFP32.blockNum, infoFP16.blockNum);
    EXPECT_EQ(infoFP32.blockNum, infoBF16.blockNum);
}

// ==========================================================================
// Iteration 3: Boundary cases - Empty tensor (totalElements = 0)
// ==========================================================================

// TC-T-041: Empty tensor NCHW fp32, shape=(0,3,4,4), totalElements=0
TEST(BnInferGradTiling, NCHW_FP32_EmptyTensor_BatchZero)
{
    gert::StorageShape gradsShape({0, 3, 4, 4}, {0, 3, 4, 4});
    auto para = MakeTilingPara(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    // Empty tensor: all tiling data should be zeroed out
    EXPECT_EQ(td.totalElements, 0);
    EXPECT_EQ(td.channelSize, 0);
    EXPECT_EQ(td.spatialSize, 0);
    EXPECT_EQ(td.tileLen, 0);
    EXPECT_EQ(td.numTiles, 0);
    EXPECT_EQ(td.lastTileLen, 0);
    EXPECT_EQ(td.usedCoreNum, 0);
    EXPECT_EQ(td.elemsPerCore, 0);
    EXPECT_EQ(td.tailCoreElems, 0);

    // Block dim should be 1 for empty tensor
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-042: Empty tensor NCHW fp32, shape=(2,0,4,4), channel=0
TEST(BnInferGradTiling, NCHW_FP32_EmptyTensor_ChannelZero)
{
    gert::StorageShape gradsShape({2, 0, 4, 4}, {2, 0, 4, 4});
    auto para = MakeTilingPara(gradsShape, 0);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 0);
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-043: Empty tensor NCHW fp32, shape=(2,3,0,4), spatial has zero dim
TEST(BnInferGradTiling, NCHW_FP32_EmptyTensor_SpatialZero)
{
    gert::StorageShape gradsShape({2, 3, 0, 4}, {2, 3, 0, 4});
    auto para = MakeTilingPara(gradsShape, 3);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 0);
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-044: Empty tensor NHWC fp16, shape=(0,4,4,3)
TEST(BnInferGradTiling, NHWC_FP16_EmptyTensor)
{
    gert::StorageShape gradsShape({0, 4, 4, 3}, {0, 4, 4, 3});
    auto para = MakeTilingParaEx(gradsShape, 3, ge::DT_FLOAT16, ge::FORMAT_NHWC);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 0);
    EXPECT_EQ(info.blockNum, 1u);
}

// TC-T-045: Empty tensor NC1HWC0 bf16, shape=(0,4,8,8,16)
TEST(BnInferGradTiling, NC1HWC0_BF16_EmptyTensor)
{
    gert::StorageShape gradsShape({0, 4, 8, 8, 16}, {0, 4, 8, 8, 16});
    int64_t C = 4 * 16;
    auto para = MakeTilingParaEx(gradsShape, C, ge::DT_BF16, ge::FORMAT_NC1HWC0);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 0);
    EXPECT_EQ(info.blockNum, 1u);
}

// ==========================================================================
// Iteration 3: Boundary cases - channel=1
// ==========================================================================

// TC-T-046: NCHW channel=1, shape=(4,1,8,8), fp32
// C=1, alignedC = AlignUp(1, 8) = 8
TEST(BnInferGradTiling, NCHW_FP32_ChannelOne)
{
    gert::StorageShape gradsShape({4, 1, 8, 8}, {4, 1, 8, 8});
    auto para = MakeTilingPara(gradsShape, 1);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 256);      // 4*1*8*8
    EXPECT_EQ(td.channelSize, 1);
    EXPECT_EQ(td.spatialSize, 64);         // 8*8
    EXPECT_EQ(td.formatMode, 0);           // NCHW
    EXPECT_EQ(td.N, 4);
    EXPECT_EQ(td.alignedC, 8);            // AlignUp(1, 8) = 8
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_EQ(td.lastTileLen, 256);
    EXPECT_GT(td.tileLen, 0);
}

// TC-T-047: NHWC channel=1, shape=(4,8,8,1), fp16
TEST(BnInferGradTiling, NHWC_FP16_ChannelOne)
{
    gert::StorageShape gradsShape({4, 8, 8, 1}, {4, 8, 8, 1});
    auto para = MakeTilingParaEx(gradsShape, 1, ge::DT_FLOAT16, ge::FORMAT_NHWC);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 256);      // 4*8*8*1
    EXPECT_EQ(td.channelSize, 1);
    EXPECT_EQ(td.spatialSize, 64);         // 8*8
    EXPECT_EQ(td.formatMode, 1);           // NHWC
    EXPECT_EQ(td.N, 4);
    EXPECT_EQ(td.alignedC, 8);            // AlignUp(1, 8) = 8
    EXPECT_EQ(td.numTiles, 1);
    EXPECT_GT(td.tileLen, 0);
}

// TC-T-048: NC1HWC0 with C1=1 (minimum), shape=(2,1,4,4,16), bf16
// C = C1*C0 = 1*16 = 16, totalTasks = N*C1 = 2
TEST(BnInferGradTiling, NC1HWC0_BF16_C1One)
{
    gert::StorageShape gradsShape({2, 1, 4, 4, 16}, {2, 1, 4, 4, 16});
    int64_t C = 1 * 16;  // 16
    auto para = MakeTilingParaEx(gradsShape, C, ge::DT_BF16, ge::FORMAT_NC1HWC0);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 2 * 1 * 4 * 4 * 16);  // 512
    EXPECT_EQ(td.channelSize, 16);
    EXPECT_EQ(td.spatialSize, 16);          // 4*4
    EXPECT_EQ(td.formatMode, 2);            // NC1HWC0
    EXPECT_EQ(td.N, 2);
    EXPECT_EQ(td.C1, 1);
    EXPECT_EQ(td.C0, 16);
    EXPECT_EQ(td.totalTasks, 2);            // N*C1 = 2*1

    // tileLen = tileHW * C0
    EXPECT_EQ(td.tileLen, td.tileHW * td.C0);

    // HW tile consistency
    EXPECT_EQ((td.numTilesHW - 1) * td.tileHW + td.lastTileHW, td.spatialSize);
}

// ==========================================================================
// Iteration 3: fp16/bf16 with NHWC large tensor multi-core
// ==========================================================================

// TC-T-049: NHWC fp16 large multi-core, shape=(8,64,64,128)
TEST(BnInferGradTiling, NHWC_FP16_MultiCore)
{
    gert::StorageShape gradsShape({8, 64, 64, 128}, {8, 64, 64, 128});
    auto para = MakeTilingParaEx(gradsShape, 128, ge::DT_FLOAT16, ge::FORMAT_NHWC, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 4194304);   // 8*64*64*128
    EXPECT_EQ(td.channelSize, 128);
    EXPECT_EQ(td.formatMode, 1);            // NHWC

    // Multi-core
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 64);

    // Core distribution
    EXPECT_EQ((td.usedCoreNum - 1) * td.elemsPerCore + td.tailCoreElems, td.totalElements);

    // tileLen aligned to C=128
    EXPECT_EQ(td.tileLen % 128, 0);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}

// TC-T-050: NC1HWC0 bf16 multi-core many tasks, shape=(4,8,4,4,16)
TEST(BnInferGradTiling, NC1HWC0_BF16_MultiCore)
{
    gert::StorageShape gradsShape({4, 8, 4, 4, 16}, {4, 8, 4, 4, 16});
    int64_t C = 8 * 16;  // 128
    auto para = MakeTilingParaEx(gradsShape, C, ge::DT_BF16, ge::FORMAT_NC1HWC0, 0.0001f, 64);

    TilingInfo info;
    BnInferGradTilingData td;
    ASSERT_TRUE(RunTilingAndGetData(para, info, td));

    EXPECT_EQ(td.totalElements, 4 * 8 * 4 * 4 * 16);  // 32768
    EXPECT_EQ(td.channelSize, 128);
    EXPECT_EQ(td.formatMode, 2);
    EXPECT_EQ(td.totalTasks, 32);           // N*C1 = 4*8

    // Multi-core
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_LE(td.usedCoreNum, 32);

    // Task distribution
    EXPECT_EQ((td.usedCoreNum - 1) * td.tasksPerCore + td.tailCoreTasks, td.totalTasks);

    EXPECT_EQ(info.blockNum, static_cast<size_t>(td.usedCoreNum));
}
