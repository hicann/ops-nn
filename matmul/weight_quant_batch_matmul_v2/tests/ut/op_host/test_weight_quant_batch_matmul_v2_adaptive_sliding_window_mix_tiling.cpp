/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "ut_string_utils.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "op_host/op_const_def_nn.h"
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window_mix_tiling.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;
using namespace ut_str;

namespace {
constexpr uint32_t TILE_SKIP_U32 = 0xffffffffU;
constexpr int32_t TILE_SKIP_I32 = -1;
constexpr uint64_t DEFAULT_UB_SIZE = 253952UL;
constexpr uint64_t DEFAULT_L0C_SIZE = 262144UL;
constexpr uint32_t DEFAULT_TILING_CAP = 4096U;

struct ASWMixTilingCaseParam {
    string caseName;

    // expectation
    bool expectSuccess = true;
    ge::graphStatus expectStatus = ge::GRAPH_SUCCESS; // PARAM_INVALID: IsCapable false; FAILED: tiling failure
    bool runFramework = false;         // run registered tilingFunc first (9201: ASWMix(12) -> ASWGeneral(13))
    bool whiteBoxUtils = false;        // check GetSize/GetShapeWithDataType (and Is4BitInput inside) directly
    uint32_t expectBlockDim = 0;       // 0: skip check
    int32_t expectEnablePreAlloc = -1; // -1: skip check
    int32_t expectDbL0c = -1;          // -1: skip check
    uint32_t expectMTailTile = TILE_SKIP_U32;
    uint32_t expectNTailTile = TILE_SKIP_U32;
    int32_t expectEvadeDuplicateBlock = -1; // -1: skip check
    bool whiteBoxNzSplit = false;           // call IsInValidWeighNzTailSplit directly with several splitCnt combos
    bool whiteBoxQuadruple = false;         // call CheckL1QuadrupleBufferSize/OpenL1QuadrupleBufferCond directly
    bool whiteBoxBaseK = false;             // call CalcBasicBlockBaseK/CheckBasicBlockL1BufferSize directly
    bool whiteBoxEvade = false;             // force adaptiveWin_ state then call CalcTailBasicBlock directly
    bool whiteBoxPreAlloc = false;          // force adaptiveWin_ tiles then call ClosePreAllocCond directly
    bool whiteBoxReSplit = false;           // force matmulInfo/adaptiveWin_ then call ReSplitBasicBlock directly
    bool whiteBoxMxReSplit = false;         // force adaptiveWin_ then call ReSplitBasicBlockForCoreNum directly
    bool whiteBoxTailLoop = false;          // force 1-block tail window then re-run CalcTailBasicBlock
    bool noWorkspace = false;               // do not attach workspace -> GetWorkspaceSize fails
};
} // namespace

class TestWeightQuantBatchMatmulV2ASWMixTiling : public testing::TestWithParam<ASWMixTilingCaseParam> {};

using namespace ge;
using namespace optiling;

static void ReplacePlaceholder(string& str, const string& placeholder, const string& replacement)
{
    size_t pos = str.find(placeholder);
    if (pos != string::npos) {
        str.replace(pos, placeholder.length(), replacement);
    }
}

// Directly exercise WeightQuantBatchMatmulV2TilingASWMix (registered with priority 12 for 9201)
// to cover ASWMix methods: DoOpTiling, IsCapable, InstantiateTilingData, AnalyseSlidingWinInfo,
// CalcBasicBlock (CalculateBaseN/ReSplitBasicBlock/ReSplitBasicBlockForCoreNum/pergroup solver),
// CalcTailBasicBlock, IsInValidWeighNzTailSplit, CalUsedCoreNum, CalL1Tiling, CalStepKs,
// CheckAntiQuantScale, SetTilingData, DoLibApiTiling, GetWorkspaceSize, PostTiling,
// ClosePreAllocCond, GetPerGroupCustomConfig, GetTilingKey.
static void TestASWMixOneCase(const ASWMixTilingCaseParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    map<string, ge::DataType> dtypeMap = {{"FLOAT16", ge::DT_FLOAT16},
                                          {"FLOAT", ge::DT_FLOAT},
                                          {"BF16", ge::DT_BF16},
                                          {"INT8", ge::DT_INT8},
                                          {"INT4", ge::DT_INT4},
                                          {"UINT64", ge::DT_UINT64},
                                          {"FP8E4M3", ge::DT_FLOAT8_E4M3FN},
                                          {"HIF8", ge::DT_HIFLOAT8},
                                          {"FLOAT4E2M1", ge::DT_FLOAT4_E2M1},
                                          {"FLOAT4E1M2", ge::DT_FLOAT4_E1M2},
                                          {"HIFLOAT4", Ops::NN::DT_HIFLOAT4},
                                          {"HIFLOAT4SCALE", Ops::NN::DT_HIFLOAT4_SCALE},
                                          {"FLOAT8E8M0", ge::DT_FLOAT8_E8M0}};

    size_t idx = 0;
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t n = stol(testParam[idx++]);
    int64_t antiQuantOffsetExistFlag = stol(testParam[idx++]);
    int64_t biasFlag = stol(testParam[idx++]);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    ge::DataType xDtype = dtypeMap[testParam[idx++]];
    ge::DataType weightDtype = dtypeMap[testParam[idx++]];
    ge::DataType antiQuantScaleDtype = dtypeMap[testParam[idx++]];
    ge::DataType yDtype = dtypeMap[testParam[idx++]];
    uint32_t aicNum = stoul(testParam[idx++]);
    uint32_t aivNum = stoul(testParam[idx++]);
    ge::Format bFormat = ge::FORMAT_ND;
    if (stol(testParam[idx++]) == 1) {
        bFormat = ge::FORMAT_FRACTAL_NZ;
    }
    uint64_t ubSize = testParam.size() > idx ? stoull(testParam[idx++]) : DEFAULT_UB_SIZE;
    uint64_t l0cSize = testParam.size() > idx ? stoull(testParam[idx++]) : DEFAULT_L0C_SIZE;
    uint32_t tilingDataCap = testParam.size() > idx ? stoul(testParam[idx++]) : DEFAULT_TILING_CAP;
    ge::DataType biasDtype = xDtype;
    if (xDtype == ge::DT_BF16) {
        biasDtype = ge::DT_FLOAT;
    }
    // attr group_size only takes positive per-group values; -1(per-channel)/0(per-tensor) both map to 0
    int64_t groupSizeAttr = group > 0 ? group : 0;

    gert::StorageShape xShape;
    gert::StorageShape weigthShape;
    gert::StorageShape antiQuantScaleShape;
    gert::StorageShape antiQuantOffsetShape;
    gert::StorageShape biasShape;
    gert::StorageShape outputShape({m, n}, {m, n});

    if (transA) {
        xShape.MutableStorageShape() = gert::Shape({k, m});
        xShape.MutableOriginShape() = gert::Shape({k, m});
    } else {
        xShape.MutableStorageShape() = gert::Shape({m, k});
        xShape.MutableOriginShape() = gert::Shape({m, k});
    }
    if (transB) {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            weigthShape.MutableStorageShape() = gert::Shape({(k + 15) / 16, (n + 15) / 16, 16, 16});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({n, k});
        }
        weigthShape.MutableOriginShape() = gert::Shape({n, k});
    } else {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            weigthShape.MutableStorageShape() = gert::Shape({(n + 15) / 16, (k + 15) / 16, 16, 16});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({k, n});
        }
        weigthShape.MutableOriginShape() = gert::Shape({k, n});
    }
    if (group > 0) {
        int64_t groupNum = (k + group - 1) / group;
        if (antiQuantScaleDtype == ge::DT_FLOAT8_E8M0 || weightDtype == Ops::NN::DT_HIFLOAT4) {
            // mx scale is 3D: (n, ceil(k / 32 * 2), 2) for fp4, (n, ceil(k / 64), 1) for hif4
            int64_t inner = antiQuantScaleDtype == ge::DT_FLOAT8_E8M0 ? 2 : 1;
            int64_t kGroup = antiQuantScaleDtype == ge::DT_FLOAT8_E8M0 ? (k + 63) / 64 : (k + 63) / 64;
            antiQuantScaleShape.MutableStorageShape() = gert::Shape({n, kGroup, inner});
            antiQuantOffsetShape.MutableStorageShape() = gert::Shape({n, kGroup, inner});
        } else {
            // 920R1 checker expects per-group scale layout (kNum, n) regardless of transB
            antiQuantOffsetShape.MutableStorageShape() = gert::Shape({groupNum, n});
            antiQuantScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
        }
    } else if (group < 0) {
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({n});
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({n});
    } else {
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({1});
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({1});
    }
    biasShape.MutableStorageShape() = gert::Shape({n});

    // platform info: Ascend960DT(DAV_9201), sizes are tunable per case
    string compileInfoStr = R"({
         "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0", "Intrinsic_fix_pipe_l0c2out": true,
                           "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                           "Intrinsic_fix_pipe_pre_conv_cast": true, "Intrinsic_mmad": true,
                           "UB_SIZE": ubSize, "L2_SIZE": 134217728, "L1_SIZE": 1048576,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": l0cSize, "CORE_NUM": aicNum,
                           "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore",
                           "socVersion": "Ascend960DT"}
                            })";
    ReplacePlaceholder(compileInfoStr, "ubSize", to_string(ubSize));
    ReplacePlaceholder(compileInfoStr, "l0cSize", to_string(l0cSize));
    ReplacePlaceholder(compileInfoStr, "aicNum", to_string(aicNum));
    ReplacePlaceholder(compileInfoStr, "aicNum", to_string(aicNum));
    ReplacePlaceholder(compileInfoStr, "aivNum", to_string(aivNum));

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> versionInfos;
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics, versionInfos);
    aicoreSpec["cube_freq"] = "1800";

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    MatmulV3CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    std::string opType("WeightQuantBatchMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(tilingDataCap);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    gert::TilingContextFaker tilingFaker;
    tilingFaker.NodeIoNum(7, 1)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
        .InputShapes({&xShape, &weigthShape, &antiQuantScaleShape,
                      antiQuantOffsetExistFlag ? &antiQuantOffsetShape : nullptr, nullptr, nullptr,
                      biasFlag ? &biasShape : nullptr})
        .OutputShapes({&outputShape})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, weightDtype, ge::FORMAT_ND, bFormat)
        .NodeInputTd(2, antiQuantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, antiQuantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(5, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(6, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(transA != 0)},
                    {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(transB != 0)},
                    {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSizeAttr)},
                    {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                    {"inner_precise", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                    {"ascendc_op_para_size", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                    {"enable_uncache", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
        .TilingData(rawTilingData.get());
    if (!param.noWorkspace) {
        tilingFaker.Workspace(workspace);
    }
    auto holder = tilingFaker.SetOpType(opType).Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", versionInfos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // framework path on DAV_9201: checker -> registerList{ASW_MIX(12), ASW_GENERAL(13)}; ASWMix runs first
    // when IsCapable and succeeds, which exercises the whole ASWMix tiling flow through the registered entry
    if (param.runFramework) {
        auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
        ASSERT_NE(tilingFunc, nullptr);
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
        ASSERT_NE(tilingContext->GetTilingKey(), 0UL);
        if (param.expectBlockDim != 0) {
            ASSERT_EQ(tilingContext->GetBlockDim(), param.expectBlockDim);
        }
    }

    auto aswMixTiling = optiling::weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2TilingASWMix(tilingContext);
    ge::graphStatus status = aswMixTiling.DoTiling();
    ASSERT_EQ(status, param.expectStatus);
    if (status != ge::GRAPH_SUCCESS) {
        return;
    }
    if (!param.expectSuccess) {
        // expectSuccess=false with GRAPH_SUCCESS is not expected here
        FAIL();
        return;
    }

    if (param.expectBlockDim != 0) {
        ASSERT_EQ(tilingContext->GetBlockDim(), param.expectBlockDim);
    }
    auto* tilingData = reinterpret_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData*>(
        tilingContext->GetRawTilingData()->GetData());
    ASSERT_EQ(tilingData->matmulTiling.M, static_cast<int32_t>(m));
    ASSERT_EQ(tilingData->matmulTiling.N, static_cast<int32_t>(n));
    ASSERT_EQ(tilingData->matmulTiling.Ka, static_cast<int32_t>(k));
    ASSERT_EQ(tilingData->matmulTiling.Kb, static_cast<int32_t>(k));
    ASSERT_EQ(tilingData->matmulTiling.stepKa, 4);
    ASSERT_EQ(tilingData->matmulTiling.stepKb, 4);
    ASSERT_EQ(tilingData->matmulTiling.depthA1, 8);
    ASSERT_EQ(tilingData->matmulTiling.depthB1, 8);
    ASSERT_EQ(tilingData->matmulTiling.stepM, 1);
    ASSERT_EQ(tilingData->matmulTiling.stepN, 1);
    ASSERT_EQ(tilingData->matmulTiling.isBias, static_cast<int32_t>(biasFlag));
    ASSERT_EQ(tilingData->groupSize, static_cast<uint64_t>(groupSizeAttr));
    if (param.expectBlockDim != 0) {
        ASSERT_EQ(tilingData->matmulTiling.usedCoreNum, static_cast<int32_t>(param.expectBlockDim));
    }
    if (param.expectEnablePreAlloc != TILE_SKIP_I32) {
        ASSERT_EQ(tilingData->enablePreAlloc, static_cast<uint16_t>(param.expectEnablePreAlloc));
    }
    if (param.expectDbL0c != TILE_SKIP_I32) {
        ASSERT_EQ(tilingData->matmulTiling.dbL0C, param.expectDbL0c);
    }
    if (param.expectMTailTile != TILE_SKIP_U32) {
        ASSERT_EQ(tilingData->mTailTile, param.expectMTailTile);
    }
    if (param.expectNTailTile != TILE_SKIP_U32) {
        ASSERT_EQ(tilingData->nTailTile, param.expectNTailTile);
    }
    if (param.expectEvadeDuplicateBlock != TILE_SKIP_I32) {
        ASSERT_EQ(tilingData->evadeDuplicateBlock, static_cast<uint16_t>(param.expectEvadeDuplicateBlock));
    }
    ASSERT_NE(aswMixTiling.GetTilingKey(), 0UL);

    if (param.whiteBoxUtils) {
        // GetSizeWithDataType/GetShapeWithDataType (which drive Is4BitInput branches) are not reached by
        // every tiling flow, so exercise them directly here
        EXPECT_EQ(aswMixTiling.GetSizeWithDataType(10, ge::DT_INT4), 5UL);
        EXPECT_EQ(aswMixTiling.GetSizeWithDataType(10, ge::DT_FLOAT4_E2M1), 5UL);
        EXPECT_EQ(aswMixTiling.GetSizeWithDataType(11, ge::DT_INT4), 0UL); // odd size -> report err, return 0
        EXPECT_EQ(aswMixTiling.GetSizeWithDataType(10, Ops::NN::DT_HIFLOAT4), 5UL);
        EXPECT_EQ(aswMixTiling.GetSizeWithDataType(10, ge::DT_FLOAT16), 20UL);
        EXPECT_EQ(aswMixTiling.GetSizeWithDataType(10, ge::DT_FLOAT8_E8M0), 10UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_INT4), 64UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_FLOAT4_E2M1), 64UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_FLOAT4_E1M2), 64UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, Ops::NN::DT_HIFLOAT4), 64UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_FLOAT16), 16UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_FLOAT), 8UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_UINT64), 4UL);
        EXPECT_EQ(aswMixTiling.GetShapeWithDataType(32, ge::DT_INT8), 32UL);
        // OpenUBQuadrupleBufferCond branches: lower threshold / baseK bound / baseN bound
        EXPECT_TRUE(aswMixTiling.OpenUBQuadrupleBufferCond(64, 64, 64));
        EXPECT_FALSE(aswMixTiling.OpenUBQuadrupleBufferCond(16, 128, 64)); // baseM * baseN < 4096
        EXPECT_FALSE(aswMixTiling.OpenUBQuadrupleBufferCond(64, 64, 128)); // baseK > 64
        EXPECT_FALSE(aswMixTiling.OpenUBQuadrupleBufferCond(64, 512, 64)); // baseN > 256
        // CheckAntiQuantScale: dbL0c==1 -> 512 bound, dbL0c==2 -> 256 bound
        EXPECT_TRUE(aswMixTiling.CheckAntiQuantScale(512, 1));
        EXPECT_FALSE(aswMixTiling.CheckAntiQuantScale(513, 1));
        EXPECT_TRUE(aswMixTiling.CheckAntiQuantScale(256, 2));
        EXPECT_FALSE(aswMixTiling.CheckAntiQuantScale(257, 2));
        // CalUsedCoreNum(mTile, nTile) = mTile * nTile * tailWinBlockCnt
        EXPECT_EQ(aswMixTiling.CalUsedCoreNum(2U, 3U), 2U * 3U * aswMixTiling.adaptiveWin_.tailWinBlockCnt);
        EXPECT_GT(aswMixTiling.BWUtilization(4, 4), 0.0);
        // CalL1TilingDepthANotfullload/CalStepKs are inlined into CalL1Tiling in the op-host object, so
        // the out-of-line bodies never show up in coverage; call them directly to exercise them
        aswMixTiling.CalL1TilingDepthANotfullload(600000);
        aswMixTiling.CalStepKs();
        EXPECT_EQ(aswMixTiling.basicTiling_.stepKa, 4);
        EXPECT_EQ(aswMixTiling.basicTiling_.depthA1, 8);
    }

    if (param.whiteBoxNzSplit) {
        // splitCnt == 0 is always valid
        EXPECT_FALSE(aswMixTiling.IsInValidWeighNzTailSplit(0, true));
        // M-direction split: baseM/splitCnt must keep >= 16
        if (aswMixTiling.adaptiveWin_.baseM < 32) {
            EXPECT_TRUE(aswMixTiling.IsInValidWeighNzTailSplit(2, true));
        } else {
            EXPECT_FALSE(aswMixTiling.IsInValidWeighNzTailSplit(2, true));
        }
        EXPECT_TRUE(aswMixTiling.IsInValidWeighNzTailSplit(aswMixTiling.adaptiveWin_.baseM, true));
        // N-direction splits with odd/unaligned results violate parity/alignment requirements
        (void)aswMixTiling.IsInValidWeighNzTailSplit(5, false);
        (void)aswMixTiling.IsInValidWeighNzTailSplit(7, false);
        (void)aswMixTiling.IsInValidWeighNzTailSplit(16, false);
    }

    if (param.whiteBoxQuadruple) {
        // huge baseM/baseN/baseK fail the L1/L0A/L0B/L0C checks one by one
        EXPECT_FALSE(aswMixTiling.CheckL1QuadrupleBufferSize(4096, 64, 128)); // l1LoadSize > l1Size
        EXPECT_FALSE(aswMixTiling.CheckL1QuadrupleBufferSize(2048, 64, 64));  // l0a overflow
        EXPECT_FALSE(aswMixTiling.CheckL1QuadrupleBufferSize(16, 4096, 64));  // l0b overflow
        EXPECT_FALSE(aswMixTiling.CheckL1QuadrupleBufferSize(128, 256, 32));  // l0c overflow (16*128*256)
        EXPECT_TRUE(aswMixTiling.CheckL1QuadrupleBufferSize(16, 64, 64));
        (void)aswMixTiling.OpenL1QuadrupleBufferCond();
        (void)aswMixTiling.GetPerGroupCustomConfig();
    }

    if (param.whiteBoxBaseK) {
        // exercise Is4BitDtype / PER_CHANNEL / MX / align / 5120-threshold branches of CalcBasicBlockBaseK
        (void)aswMixTiling.CalcBasicBlockBaseK(16, 64);
        (void)aswMixTiling.CalcBasicBlockBaseK(128, 256);
        (void)aswMixTiling.CalcBasicBlockBaseK(16, 192);
        // exercise PER_CHANNEL-scale / FP4 / HIF4 / generic 4bit / default branches of
        // CheckBasicBlockL1BufferSize, including l1Size >= L1_SIZE rejection
        EXPECT_TRUE(aswMixTiling.CheckBasicBlockL1BufferSize(16, 64, 64));
        EXPECT_FALSE(aswMixTiling.CheckBasicBlockL1BufferSize(1024, 1024, 1024));
    }

    if (param.whiteBoxEvade) {
        // force the tail-window state and re-run CalcTailBasicBlock to cover the evadeDuplicateBlock
        // branch: m/n tiles stay 1x1 while totalBlockCnt > aicNum and mSize <= 128
        aswMixTiling.adaptiveWin_.tailWinBlockCnt = 15;
        aswMixTiling.adaptiveWin_.mTailTile = 1;
        aswMixTiling.adaptiveWin_.nTailTile = 1;
        aswMixTiling.adaptiveWin_.totalBlockCnt = 31;
        aswMixTiling.adaptiveWin_.totalWinCnt = 2;
        aswMixTiling.adaptiveWin_.mTail = 128;
        aswMixTiling.adaptiveWin_.nTail = 140;
        aswMixTiling.adaptiveWin_.baseM = 128;
        aswMixTiling.adaptiveWin_.baseN = 192;
        aswMixTiling.adaptiveWin_.mainTileNum = 1;
        aswMixTiling.CalcTailBasicBlock();
        EXPECT_TRUE(aswMixTiling.adaptiveWin_.evadeDuplicateBlock);
        EXPECT_EQ(aswMixTiling.adaptiveWin_.mainTileNum, 16U);     // FloorAlign(31, 16)
        EXPECT_EQ(aswMixTiling.adaptiveWin_.lastRoundBaseN, 192U); // CeilAlign(CeilDiv(140,16),16)
    }

    if (param.whiteBoxPreAlloc) {
        // cover the !(totalWinCnt == 1 && mTailTile * nTailTile > 1) sub-branch forcing preAlloc on:
        // with one round and tail tiles > 1 the big-L1 condition is neutralized
        aswMixTiling.adaptiveWin_.totalWinCnt = 1;
        aswMixTiling.adaptiveWin_.mTailTile = 2;
        aswMixTiling.adaptiveWin_.nTailTile = 1;
        EXPECT_FALSE(aswMixTiling.ClosePreAllocCond());
    }

    if (param.whiteBoxReSplit) {
        // candidate (128, 240) passes the L1/L0 checks but fails the non-MX bias UB bound -> continue
        aswMixTiling.matmulInfoPtr_->mSize = 480;
        aswMixTiling.matmulInfoPtr_->nSize = 960;
        aswMixTiling.adaptiveWin_.baseM = 128;
        aswMixTiling.adaptiveWin_.baseN = 960;
        aswMixTiling.ReSplitBasicBlock();
        // candidate (128, 256) is rejected by the baseM==128 && baseN>240 guard -> continue
        aswMixTiling.matmulInfoPtr_->mSize = 128;
        aswMixTiling.matmulInfoPtr_->nSize = 4096;
        aswMixTiling.adaptiveWin_.baseM = 128;
        aswMixTiling.adaptiveWin_.baseN = 4096;
        aswMixTiling.ReSplitBasicBlock();
        // perchannel+bias: aligned halves but baseN(31) not divisible by splitCnt(2) -> invalid
        aswMixTiling.adaptiveWin_.baseM = 128;
        aswMixTiling.adaptiveWin_.baseN = 31;
        aswMixTiling.adaptiveWin_.mTail = 31;
        aswMixTiling.adaptiveWin_.nTail = 31;
        EXPECT_TRUE(aswMixTiling.IsInValidWeighNzTailSplit(2, false));
        // fully aligned split on perchannel+bias stays valid
        aswMixTiling.adaptiveWin_.baseN = 128;
        aswMixTiling.adaptiveWin_.nTail = 128;
        EXPECT_FALSE(aswMixTiling.IsInValidWeighNzTailSplit(2, false));
    }

    if (param.whiteBoxMxReSplit) {
        // one M-block of 128 with a 256-wide N uses only 2 blocks (< aicNum/2) -> baseM halving loop
        aswMixTiling.matmulInfoPtr_->mSize = 128;
        aswMixTiling.adaptiveWin_.baseM = 128;
        aswMixTiling.adaptiveWin_.baseN = 256;
        aswMixTiling.ReSplitBasicBlockForCoreNum();
        EXPECT_LT(aswMixTiling.adaptiveWin_.baseM, 128U);
        // baseM already at the floor: the baseN halving branch takes over
        aswMixTiling.matmulInfoPtr_->mSize = 32;
        aswMixTiling.adaptiveWin_.baseM = 16;
        aswMixTiling.adaptiveWin_.baseN = 256;
        aswMixTiling.ReSplitBasicBlockForCoreNum();
        EXPECT_LT(aswMixTiling.adaptiveWin_.baseN, 256U);
    }

    if (param.whiteBoxTailLoop) {
        // a single tail-window block lets both pre/sec splits grow: covers the secSplit branch of
        // CalcTailBasicBlock (IsInValidWeighNzTailSplit on both directions)
        aswMixTiling.adaptiveWin_.tailWinBlockCnt = 1;
        aswMixTiling.adaptiveWin_.mTailTile = 1;
        aswMixTiling.adaptiveWin_.nTailTile = 1;
        aswMixTiling.adaptiveWin_.totalBlockCnt = 17;
        aswMixTiling.adaptiveWin_.totalWinCnt = 1;
        aswMixTiling.adaptiveWin_.mTail = 128;
        aswMixTiling.adaptiveWin_.nTail = 128;
        aswMixTiling.adaptiveWin_.baseM = 128;
        aswMixTiling.adaptiveWin_.baseN = 128;
        aswMixTiling.CalcTailBasicBlock();
        EXPECT_GT(aswMixTiling.adaptiveWin_.mTailTile * aswMixTiling.adaptiveWin_.nTailTile, 1U);
    }
}

TEST_P(TestWeightQuantBatchMatmulV2ASWMixTiling, generalTest)
{
    ASWMixTilingCaseParam param = GetParam();
    TestASWMixOneCase(param);
}

// format: m k n antiQuantOffsetExistFlag biasFlag transA transB group xDtype weightDtype
//         antiQuantScaleDtype yDtype aicNum aivNum weightFormat(0:ND,1:NZ)
//         [ubSize l0cSize tilingDataCap]
// Note: group value: -1: per channel, 0: per tensor, > 0: per group (or MX group when scale dtype is E8M0)
// MX HIFLOAT4 tail-tile expectation: when the CANN runtime registers DT_HIFLOAT4_SCALE in
// ge::GetSizeByDataType (e.g. CANN 9.3.0), ReSplitBasicBlock accepts the (baseM=16, baseN=128)
// candidate, totalBlockCnt equals aicNum, the tail window vanishes and m/nTailTile keep the
// default 1/1. On CANN versions where the dtype is not registered, GetSizeByDataType returns -1,
// the wrapped L1 size check rejects every resplit candidate, the (128, 80) block survives and the
// 4-block tail window is resplit on the M axis (mTailTile = 2). Derive the expectation from the
// registered size so the assertion holds on both runtimes.
static const uint32_t HIF4_MX_EXPECT_MTILE = static_cast<uint32_t>(ge::GetSizeByDataType(Ops::NN::DT_HIFLOAT4_SCALE)) >
                                                     1000U ?
                                                 2U :
                                                 1U;
static ge::graphStatus ParseExpectStatus(const string& value)
{
    if (value == "GRAPH_PARAM_INVALID") {
        return ge::GRAPH_PARAM_INVALID;
    }
    return ut_str::ParseGraphStatus(value); // GRAPH_SUCCESS / GRAPH_FAILED
}

// tail-tile expectation: -1 means skip the check; HIF4_MX maps to the runtime-derived
// HIF4_MX_EXPECT_MTILE (see the note above)
static uint32_t ParseTileExpect(const string& value)
{
    if (value == "HIF4_MX") {
        return HIF4_MX_EXPECT_MTILE;
    }
    if (value == "-1") {
        return TILE_SKIP_U32;
    }
    return static_cast<uint32_t>(stoul(value));
}

// cases live in the csv next to this file: caseName + expectation/config columns, one line per
// case ('#' comment lines are skipped)
static std::vector<ASWMixTilingCaseParam> GetParams()
{
    std::vector<ASWMixTilingCaseParam> params;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(rootPath + "matmul/weight_quant_batch_matmul_v2/tests/ut/op_host/"
                                          "test_weight_quant_batch_matmul_v2_adaptive_sliding_window_mix_tiling.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    std::string line;
    bool skipHeader = true;
    while (std::getline(csvData, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }

        std::vector<std::string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < 20UL) {
            std::cout << "skip invalid csv line: " << line << std::endl;
            continue;
        }

        ASWMixTilingCaseParam param;
        param.caseName = Trim(cols[0]);
        param.expectSuccess = ParseBool(Trim(cols[1]));
        param.expectStatus = ParseExpectStatus(Trim(cols[2]));
        param.runFramework = ParseBool(Trim(cols[3]));
        param.whiteBoxUtils = ParseBool(Trim(cols[4]));
        param.expectBlockDim = static_cast<uint32_t>(stoul(Trim(cols[5])));
        param.expectEnablePreAlloc = stoi(Trim(cols[6]));
        param.expectDbL0c = stoi(Trim(cols[7]));
        param.expectMTailTile = ParseTileExpect(Trim(cols[8]));
        param.expectNTailTile = ParseTileExpect(Trim(cols[9]));
        param.expectEvadeDuplicateBlock = stoi(Trim(cols[10]));
        param.whiteBoxNzSplit = ParseBool(Trim(cols[11]));
        param.whiteBoxQuadruple = ParseBool(Trim(cols[12]));
        param.whiteBoxBaseK = ParseBool(Trim(cols[13]));
        param.whiteBoxEvade = ParseBool(Trim(cols[14]));
        param.whiteBoxPreAlloc = ParseBool(Trim(cols[15]));
        param.whiteBoxReSplit = ParseBool(Trim(cols[16]));
        param.whiteBoxMxReSplit = ParseBool(Trim(cols[17]));
        param.whiteBoxTailLoop = ParseBool(Trim(cols[18]));
        param.noWorkspace = ParseBool(Trim(cols[19]));
        params.push_back(param);
    }
    return params;
}

INSTANTIATE_TEST_CASE_P(MM, TestWeightQuantBatchMatmulV2ASWMixTiling, testing::ValuesIn(GetParams()));
