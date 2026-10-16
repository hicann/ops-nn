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
 * \file test_bn3_d_training_update_tiling.cpp
 * \brief BN3DTrainingUpdate arch35 tiling UT。
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../../op_host/arch35/bn3_d_training_update_tiling.h"
#include "../../../../op_kernel/arch35/bn3_d_training_update_tiling_data.h"
#include "../../../../op_kernel/arch35/bn3_d_training_update_tiling_key.h"

namespace {

using Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES;
using Bn3dTrainingUpdateOps::BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateChannelFirstTilingData;
using Bn3dTrainingUpdateOps::Bn3dTrainingUpdateChannelLastTilingData;

constexpr uint64_t kCoreNum = 64U;
constexpr uint64_t kUbSize = 245760U;
constexpr uint64_t kTilingDataCap = 4096U;
constexpr int64_t kVlFp32 = 256 / 4; // Ops::Base::GetVRegSize() / sizeof(float)
constexpr int64_t kBlockBytes = 32;

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

struct TilingRun {
    ge::graphStatus status = ge::GRAPH_FAILED;
    int64_t key = -1;
    size_t dataSize = 0;
    std::vector<uint8_t> data;
    size_t blockNum = 0;
};

// x 的 7 输入 / 5 输出 + 2 attr 上下文;statLen 为六路 (C,) 统计量的长度。
TilingRun RunTiling(const std::vector<int64_t>& xDims, ge::Format xFormat, ge::DataType xDtype, int64_t statLen,
                    float factor = 0.1f, float epsilon = 1.0e-5f, ge::Format statFormat = ge::FORMAT_ND,
                    std::vector<int64_t> statDims = {}, std::vector<int64_t> outStatDims = {})
{
    const gert::StorageShape xShape = MakeShape(xDims);
    const std::vector<int64_t> statDimsEff = statDims.empty() ? std::vector<int64_t>{statLen} : statDims;
    const gert::StorageShape statShape = MakeShape(statDimsEff);
    const gert::StorageShape outStatShape = MakeShape(outStatDims.empty() ? statDimsEff : outStatDims);
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    inputs.emplace_back(xShape, xDtype, xFormat, false, nullptr, xFormat);
    for (int64_t i = 0; i < 6; ++i) {
        inputs.emplace_back(statShape, ge::DT_FLOAT, statFormat, false, nullptr, statFormat);
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs;
    outputs.emplace_back(xShape, xDtype, xFormat, false, nullptr, xFormat);
    for (int64_t i = 0; i < 4; ++i) {
        outputs.emplace_back(outStatShape, ge::DT_FLOAT, statFormat, false, nullptr, statFormat);
    }
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.emplace_back("factor", Ops::NN::AnyValue::CreateFrom<float>(factor));
    attrs.emplace_back("epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon));
    optiling::Bn3dTrainingUpdateCompileInfo compileInfo{static_cast<int64_t>(kCoreNum), static_cast<int64_t>(kUbSize),
                                                        256, kBlockBytes};
    gert::TilingContextPara para("BN3DTrainingUpdate", inputs, outputs, attrs, &compileInfo, kCoreNum, kUbSize,
                                 kTilingDataCap);
    TilingInfo info;
    TilingRun run;
    const bool executed = ExecuteTiling(para, info);
    run.status = executed ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    run.key = info.tilingKey;
    run.dataSize = info.tilingDataSize;
    run.blockNum = info.blockNum;
    if (info.tilingData != nullptr && info.tilingDataSize > 0) {
        run.data.assign(info.tilingData.get(), info.tilingData.get() + info.tilingDataSize);
    }
    return run;
}

const Bn3dTrainingUpdateChannelFirstTilingData* AsCf(const TilingRun& run)
{
    return reinterpret_cast<const Bn3dTrainingUpdateChannelFirstTilingData*>(run.data.data());
}

const Bn3dTrainingUpdateChannelLastTilingData* AsCl(const TilingRun& run)
{
    return reinterpret_cast<const Bn3dTrainingUpdateChannelLastTilingData*>(run.data.data());
}

optiling::Bn3dTrainingUpdatePlatform Platform()
{
    return optiling::Bn3dTrainingUpdatePlatform{static_cast<int64_t>(kCoreNum), static_cast<int64_t>(kUbSize), kVlFp32,
                                                kBlockBytes};
}

} // namespace

class BN3DTrainingUpdateTilingTest : public testing::Test {};

// ============================ TilingKey 映射 + ABI ============================

TEST_F(BN3DTrainingUpdateTilingTest, channel_first_nchw_fp16)
{
    const TilingRun run = RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    ASSERT_EQ(run.data.size(), BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    const auto* td = AsCf(run);
    EXPECT_EQ(td->c, 3);
    EXPECT_EQ(td->spatial, 20); // 4 * 5
    // 单元总数恒等于 cWindows·N·spatialTiles(cWindows = ceil(C/K),N = 2 来自本用例的 shape)。
    ASSERT_GE(td->channelWindow, 1);
    const int64_t cWindows = (td->c + td->channelWindow - 1) / td->channelWindow;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              cWindows * 2 * td->spatialTiles);
    EXPECT_GE(td->spatialTile, 1);
    EXPECT_GE(run.blockNum, 1U);
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_first_ncdhw_bf16)
{
    const TilingRun run = RunTiling({2, 3, 4, 5, 6}, ge::FORMAT_NCDHW, ge::DT_BF16, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    const auto* td = AsCf(run);
    EXPECT_EQ(td->spatial, 120); // 4 * 5 * 6
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_last_nhwc_fp32)
{
    const TilingRun run = RunTiling({2, 4, 5, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    ASSERT_EQ(run.data.size(), BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    const auto* td = AsCl(run);
    EXPECT_EQ(td->c, 3);
    EXPECT_EQ(td->num, 40); // 2 * 4 * 5
    const int64_t channelBlocks = (td->c + td->channelBlock - 1) / td->channelBlock;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              channelBlocks * td->tilesPerBlock);
    EXPECT_GE(td->coeffCap, td->channelBlock);
    EXPECT_GE(td->rowsPerTile, 1);
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_last_ndhwc_fp16)
{
    const TilingRun run = RunTiling({2, 4, 5, 6, 3}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    EXPECT_EQ(run.dataSize, BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    const auto* td = AsCl(run);
    EXPECT_EQ(td->num, 240); // 2 * 4 * 5 * 6
}

// BN3D-1 定向回归:channel-last 且 C == 6144(邻域 6143/6145/8192/12288)必须能出合法 tiling。
TEST_F(BN3DTrainingUpdateTilingTest, channel_last_regression_bn3d_1_neighbourhood)
{
    const std::vector<int64_t> channels = {6143, 6144, 6145, 8192, 12288};
    for (const int64_t c : channels) {
        const TilingRun run = RunTiling({1, 1, 1, 1, c}, ge::FORMAT_NDHWC, ge::DT_FLOAT16, c);
        ASSERT_EQ(run.status, ge::GRAPH_SUCCESS) << "C=" << c;
        ASSERT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST)) << "C=" << c;
        const auto* td = AsCl(run);
        EXPECT_EQ(td->c, c);
        EXPECT_GE(td->channelBlock, 1);
        EXPECT_GE(td->rowsPerTile, 1);
    }
}

// ============================ UB 反解不变量 ============================

TEST_F(BN3DTrainingUpdateTilingTest, channel_first_ub_ledger_within_ub)
{
    const TilingRun run = RunTiling({2, 3, 64, 64}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    const auto plat = Platform();
    EXPECT_LE(optiling::CfKernelBytes(td->spatialTile, td->channelWindow, 4, plat), plat.ubSize);
    EXPECT_LE(optiling::StatPhaseBytes(plat) + optiling::kPipeMetaReserveBytes, plat.ubSize);
    // 切分和恒等于单元总数(每核段 + 尾部分摊);shape {2,3,64,64} ⇒ 总数 = cWindows·N·spatialTiles。
    const int64_t cw = (td->c + td->channelWindow - 1) / td->channelWindow;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              cw * 2 * td->spatialTiles);
    EXPECT_LE(td->channelWindow, td->c); // K <= C
}

TEST_F(BN3DTrainingUpdateTilingTest, channel_last_ub_ledger_within_ub)
{
    const TilingRun run = RunTiling({2, 64, 64, 100}, ge::FORMAT_NHWC, ge::DT_FLOAT, 100);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCl(run);
    const auto plat = Platform();
    EXPECT_LE(optiling::ClKernelBytes(td->rowsPerTile, td->channelBlock, td->coeffCap, 4, plat), plat.ubSize);
    EXPECT_LE(optiling::StatPhaseBytes(plat) + optiling::kPipeMetaReserveBytes, plat.ubSize);
    EXPECT_EQ(td->statUnitCount, (td->c + td->statChunk - 1) / td->statChunk);
    EXPECT_EQ(td->baseStatPerCore * static_cast<int64_t>(run.blockNum) + td->extraStatCount, td->statUnitCount);
    // VF 循环计数器(uint16_t)上界不变量:一个瓦片的元素数必须装得下。
    EXPECT_LE(td->rowsPerTile * td->channelBlock, optiling::kMaxTileElementsPerVf);
}

// 块长 = "UB 上界" 与 "单元数目标" 取小 —— 小形状不得再让单元数塌到 1。
TEST_F(BN3DTrainingUpdateTilingTest, f1_units_target_reached_on_small_shapes)
{
    const int64_t unitsTarget = static_cast<int64_t>(kCoreNum) * optiling::kUnitsPerCoreTarget;
    // tiny_total 形态:NDHWC (1,1,64,64,5) —— 旧实现 blockDim 塌到 1;修正后单元数应 >= unitsTarget。
    const TilingRun cl = RunTiling({1, 1, 64, 64, 5}, ge::FORMAT_NDHWC, ge::DT_BF16, 5);
    ASSERT_EQ(cl.status, ge::GRAPH_SUCCESS);
    const auto* tdc = AsCl(cl);
    const int64_t channelBlocks = (tdc->c + tdc->channelBlock - 1) / tdc->channelBlock;
    EXPECT_GE(channelBlocks * tdc->tilesPerBlock, unitsTarget) << "channel-last 小形状单元数未达标";
    EXPECT_EQ(tdc->baseTilesPerCore * static_cast<int64_t>(cl.blockNum) + tdc->extraCoreCount,
              channelBlocks * tdc->tilesPerBlock);
    // 几何精确钉住(单元数目标生效后的实测值):C=5 ⇒ Cb=5、channelBlocks=1;rows 由单元数目标压到 16
    EXPECT_EQ(tdc->channelBlock, 5);
    EXPECT_EQ(tdc->rowsPerTile, 16);
    EXPECT_EQ(tdc->tilesPerBlock, 256);
    // channel-first 小 S 侧同式:NCDHW (8,1024,1,2,160) 的 S=320 缩到 S=2 的极端
    const TilingRun cf = RunTiling({2, 1024, 1, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT16, 1024);
    ASSERT_EQ(cf.status, ge::GRAPH_SUCCESS);
    const auto* tdf = AsCf(cf);
    const int64_t cw = (tdf->c + tdf->channelWindow - 1) / tdf->channelWindow;
    EXPECT_GE(cw * 2 * tdf->spatialTiles, unitsTarget) << "channel-first 小 S 单元数未达标";
    EXPECT_LE(optiling::CfKernelBytes(tdf->spatialTile, tdf->channelWindow, 2, Platform()), Platform().ubSize);
    // 几何精确钉住:C=1024、N=2、S=2 ⇒ K_ub 被 kParamChunkLen 钳到 512,但 Kcap_units =
    // floor(C·N·spatialTiles / unitsTarget) = floor(2048/256) = 8 更小 ⇒ K = 8(取小,非阈值路由);
    // cWindows = 128 ⇒ 单元数 = 128·2·1 = 256 = unitsTarget ✓
    EXPECT_EQ(tdf->channelWindow, 8);
    EXPECT_EQ(tdf->spatialTile, 2);
    EXPECT_EQ(cw * 2 * tdf->spatialTiles, unitsTarget);
}

// CF 单元放大为 K 个连续通道平面 —— 大 C + 小 S 的形状必须有 K > 1。
TEST_F(BN3DTrainingUpdateTilingTest, f2_channel_window_enlarged_for_small_spatial)
{
    // xpf_00070 形态:NCDHW (4, 32768, 1, 4, 4) ⇒ S = 16
    const TilingRun run = RunTiling({4, 32768, 1, 4, 4}, ge::FORMAT_NCDHW, ge::DT_BF16, 32768);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    EXPECT_GT(td->channelWindow, 1); // 单元不再被钉在"1 个通道平面"
    EXPECT_LE(td->channelWindow, 32768);
    const int64_t cw = (td->c + td->channelWindow - 1) / td->channelWindow;
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount,
              cw * 4 * td->spatialTiles);
    EXPECT_LE(optiling::CfKernelBytes(td->spatialTile, td->channelWindow, 2, Platform()), Platform().ubSize);
    // 几何精确钉住(xpf_00070 形态):K 达折叠分块上界 512、Ss=16(=S)、单元数 = 64·4·1 = 256
    EXPECT_EQ(td->channelWindow, 512);
    EXPECT_EQ(td->spatialTile, 16);
    EXPECT_EQ(cw * 4 * td->spatialTiles, 256);
}

// ============================ 边界:C == 0 支持(空张量)/ num == 0 支持(空 batch)============================

// C == 0(通道数为 0)⇒ **不拒收**:x 与所有 (C,) 统计量均为空张量,产出"全空"tiling(无 X / 相 S 单元),
// kernel 空跑即正确。对齐旧版语义与 A2/A3 支持面,DFX 有 empt_c0_* 覆盖。
TEST_F(BN3DTrainingUpdateTilingTest, zero_channel_supported)
{
    // channel-first:NCHW (2,0,4,5),统计量 (0,) ⇒ C = 0。
    const TilingRun cf = RunTiling({2, 0, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 0);
    EXPECT_EQ(cf.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(cf.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    ASSERT_EQ(cf.data.size(), BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
    EXPECT_EQ(AsCf(cf)->c, 0);
    EXPECT_EQ(AsCf(cf)->statUnitCount, 0);
    EXPECT_EQ(AsCf(cf)->baseTilesPerCore, 0);
    EXPECT_EQ(AsCf(cf)->extraCoreCount, 0);

    // channel-last:NHWC (2,4,5,0),统计量 (0,) ⇒ C = 0。
    const TilingRun cl = RunTiling({2, 4, 5, 0}, ge::FORMAT_NHWC, ge::DT_FLOAT, 0);
    EXPECT_EQ(cl.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(cl.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    ASSERT_EQ(cl.data.size(), BN3D_TRAINING_UPDATE_CL_TILING_DATA_BYTES);
    EXPECT_EQ(AsCl(cl)->c, 0);
    EXPECT_EQ(AsCl(cl)->statUnitCount, 0);
    EXPECT_EQ(AsCl(cl)->baseTilesPerCore, 0);
    EXPECT_EQ(AsCl(cl)->extraCoreCount, 0);
}

// num == 0(C >= 1、某非通道轴为 0)⇒ 零统计契约,**不报错**(与 C == 0 同为空张量,但 C >= 1)。
TEST_F(BN3DTrainingUpdateTilingTest, empty_batch_supported_channel_first)
{
    const TilingRun run = RunTiling({0, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
    const auto* td = AsCf(run);
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount, 0);
    EXPECT_EQ(td->numRec, 0.0f);
    EXPECT_EQ(td->besselScaler, 0.0f);
    EXPECT_GE(td->statUnitCount, 1); // 相 S 单元照常覆盖 [0, C)
}

TEST_F(BN3DTrainingUpdateTilingTest, empty_batch_supported_channel_last)
{
    const TilingRun run = RunTiling({2, 4, 0, 3}, ge::FORMAT_NHWC, ge::DT_FLOAT, 3);
    EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_LAST));
    const auto* td = AsCl(run);
    EXPECT_EQ(td->num, 0);
    EXPECT_EQ(td->baseTilesPerCore * static_cast<int64_t>(run.blockNum) + td->extraCoreCount, 0);
    EXPECT_EQ(td->numRec, 0.0f);
    EXPECT_GE(td->statUnitCount, 1);
}

// num == 1:Bessel scaler 置 0(仅影响 running variance 的 EMA)。
TEST_F(BN3DTrainingUpdateTilingTest, single_sample_batch_bessel_zero)
{
    const TilingRun run = RunTiling({1, 3, 1, 1, 1}, ge::FORMAT_NCDHW, ge::DT_FLOAT, 3);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    EXPECT_EQ(td->numRec, 1.0f);
    EXPECT_EQ(td->besselScaler, 0.0f);
}

TEST_F(BN3DTrainingUpdateTilingTest, bessel_scaler_applied_for_num_gt_one)
{
    const TilingRun run = RunTiling({2, 3, 2, 2}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3);
    ASSERT_EQ(run.status, ge::GRAPH_SUCCESS);
    const auto* td = AsCf(run);
    EXPECT_FLOAT_EQ(td->numRec, 1.0f / 8.0f);
    EXPECT_FLOAT_EQ(td->besselScaler, 8.0f / 7.0f);
}

// ============================ 负向:非法输入干净拒绝 ============================

TEST_F(BN3DTrainingUpdateTilingTest, negative_rank3_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_rank5_with_4d_format_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5, 6}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_unsupported_format_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_FRACTAL_NZ, ge::DT_FLOAT, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_dtype_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_INT32, 3).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, negative_channel_mismatch_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 7).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NHWC, ge::DT_FLOAT, 7).status, ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_format_label_irrelevant_when_rank1)
{
    const ge::Format labels[] = {ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_NZ,
                                 ge::FORMAT_NCDHW};
    for (const ge::Format label : labels) {
        const TilingRun run = RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, label);
        EXPECT_EQ(run.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(run.key, static_cast<int64_t>(BN3D_TRAINING_UPDATE_KEY_CHANNEL_FIRST));
        ASSERT_EQ(run.data.size(), BN3D_TRAINING_UPDATE_CF_TILING_DATA_BYTES);
        EXPECT_EQ(AsCf(run)->c, 3);
    }
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_layout_rank_gt1_rejected_by_rank)
{
    // 对照(同 label NC1HWC0、仅 rank 不同):rank-1 ⇒ 通过;下一行 rank-5 ⇒ 拒 ⇒ 否决权在 rank。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NC1HWC0).status,
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NC1HWC0, {1, 1, 4, 5, 16})
            .status,
        ge::GRAPH_FAILED);
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NCDHW, {1, 3, 4, 5, 6})
            .status,
        ge::GRAPH_FAILED);
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_FRACTAL_NZ, {1, 3, 16, 32})
            .status,
        ge::GRAPH_FAILED);
    // 同 rank 换 label:ND 标签 + rank-5 一样被拒 ⇒ 否决权在 rank,不在 format。
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {1, 3, 4, 5, 6}).status,
        ge::GRAPH_FAILED);
}

// 统计量非 rank-1(带布局的二维/更高维张量)仍然被拒绝——"统计量必须纯一维"的意图未丢。
TEST_F(BN3DTrainingUpdateTilingTest, stat_rank_two_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_NCHW, {1, 3}).status,
              ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {3, 1}).status,
              ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_output_rank_gt1_rejected_by_shape_equality)
{
    // 对照:输出 (3,) 与输入同形 ⇒ 通过。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_SUCCESS);
    // 输出声明成 rank-2 (1,3):与输入 (3,) 形状不等(rank 即不等)⇒ 被 shape 相等校验拒。
    EXPECT_EQ(
        RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {}, {1, 3}).status,
        ge::GRAPH_FAILED);
}

TEST_F(BN3DTrainingUpdateTilingTest, stat_output_shape_mismatch_rejected)
{
    // 对照:输出 (3,) 与输入同形 ⇒ 通过。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3).status, ge::GRAPH_SUCCESS);
    // 输出 rank-1 但长度 (4,) != C = 3 → 仅"输出必须等于输入 (C,) 形状"能拒它。
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 1.0e-5f, ge::FORMAT_ND, {}, {4}).status,
              ge::GRAPH_FAILED);
}

// factor 的取值范围**不**作为拒绝条件(契约 def.cpp/proto.h 未声明,A2 亦不校验;factor==0 是 PyTorch
// momentum 的合法用法 → running 统计量冻结);仅拒绝非有限值。
TEST_F(BN3DTrainingUpdateTilingTest, factor_range_accepted)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.0f, 1.0e-5f).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 1.0f, 1.0e-5f).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 1.5f, 1.0e-5f).status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NHWC, ge::DT_FLOAT, 5, 0.0f, 1.0e-5f).status, ge::GRAPH_SUCCESS);
}

TEST_F(BN3DTrainingUpdateTilingTest, non_finite_attr_rejected)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, nan, 1.0e-5f).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, inf, 1.0e-5f).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, nan).status, ge::GRAPH_FAILED);
}

// epsilon <= 0 仍拒绝:sqrt(saveVar + eps) 在常量特征上退化为 1/sqrt(0) = inf/NaN。
TEST_F(BN3DTrainingUpdateTilingTest, non_positive_epsilon_rejected)
{
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, 0.0f).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling({2, 3, 4, 5}, ge::FORMAT_NCHW, ge::DT_FLOAT, 3, 0.1f, -1.0e-5f).status, ge::GRAPH_FAILED);
}
