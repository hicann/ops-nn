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
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_sliding_window_general_tiling.h"
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

struct ASWGeneralTilingCaseParam {
    string caseName;

    // expectation
    bool expectSuccess = true;
    bool runFramework = false;         // run registered tilingFunc (9201: ASWMix -> ASWGeneral) first
    bool whiteBoxUtils = false;        // check GetSize/GetShapeWithDataType (and Is4BitInput inside) directly
    uint32_t expectBlockDim = 0;       // 0: skip check
    int32_t expectEnablePreAlloc = -1; // -1: skip check
    int32_t expectDbL0c = -1;          // -1: skip check
    uint32_t expectMTailTile = TILE_SKIP_U32;
    uint32_t expectNTailTile = TILE_SKIP_U32;
    bool disableTailWinLogic = false; // white-box: clear useTailWinLogic before DoTiling
    bool noWorkspace = false;         // do not attach workspace -> GetWorkspaceSize fails
    bool whiteBoxNzSplit = false;     // call IsInValidWeighNzTailSplit directly on NZ no-transB state
};
} // namespace

class TestWeightQuantBatchMatmulV2ASWGeneralTiling : public testing::TestWithParam<ASWGeneralTilingCaseParam> {};

using namespace ge;
using namespace optiling;

static void ReplacePlaceholder(string& str, const string& placeholder, const string& replacement)
{
    size_t pos = str.find(placeholder);
    if (pos != string::npos) {
        str.replace(pos, placeholder.length(), replacement);
    }
}

// Directly exercise WeightQuantBatchMatmulV2TilingASWGeneral (registered with priority 13 for 9201)
// to cover ASWGeneral methods: DoOpTiling, InstantiateTilingData, AnalyseSlidingWinInfo, CalcBasicBlock,
// CalculateBaseN, CalcTailBasicBlock, IsInValidWeighNzTailSplit, CalUsedCoreNum, CalL1Tiling, CalStepKs,
// SetTilingData, DoLibApiTiling, GetWorkspaceSize, PostTiling, ClosePreAllocCond, GetTilingKey.
static void TestASWGeneralOneCase(const ASWGeneralTilingCaseParam& param)
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
        if (transB) {
            antiQuantOffsetShape.MutableStorageShape() = gert::Shape({n, groupNum});
            antiQuantScaleShape.MutableStorageShape() = gert::Shape({n, groupNum});
        } else {
            antiQuantOffsetShape.MutableStorageShape() = gert::Shape({groupNum, n});
            antiQuantScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
        }
    } else if (group < 0) {
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({n});
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({n});
    } else {
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({1});
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({1});
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

    // framework path on DAV_9201: checker -> registerList{ASW_MIX(12), ASW_GENERAL(13)}
    if (param.runFramework) {
        auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
        ASSERT_NE(tilingFunc, nullptr);
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
        ASSERT_NE(tilingContext->GetTilingKey(), 0UL);
        if (param.expectBlockDim != 0) {
            ASSERT_EQ(tilingContext->GetBlockDim(), param.expectBlockDim);
        }
    }

    auto aswGeneralTiling = optiling::weight_quant_batch_matmul_v2::WeightQuantBatchMatmulV2TilingASWGeneral(
        tilingContext);
    if (param.disableTailWinLogic) {
        // white-box (-fno-access-control): AnalyseSlidingWinInfo skips CalcTailBasicBlock when the flag is off
        aswGeneralTiling.adaptiveWin_.useTailWinLogic = false;
    }
    ge::graphStatus status = aswGeneralTiling.DoTiling();
    if (!param.expectSuccess) {
        ASSERT_NE(status, ge::GRAPH_SUCCESS);
        return;
    }
    ASSERT_EQ(status, ge::GRAPH_SUCCESS);

    if (param.expectBlockDim != 0) {
        ASSERT_EQ(tilingContext->GetBlockDim(), param.expectBlockDim);
    }
    auto* tilingData = reinterpret_cast<const wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData*>(
        tilingContext->GetRawTilingData()->GetData());
    ASSERT_EQ(tilingData->matmulTiling.M, static_cast<int32_t>(m));
    ASSERT_EQ(tilingData->matmulTiling.N, static_cast<int32_t>(n));
    ASSERT_EQ(tilingData->matmulTiling.Ka, static_cast<int32_t>(k));
    ASSERT_EQ(tilingData->matmulTiling.Kb, static_cast<int32_t>(k));
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
    ASSERT_NE(aswGeneralTiling.GetTilingKey(), 0UL);

    if (param.whiteBoxUtils) {
        // GetSizeWithDataType/GetShapeWithDataType (which drive Is4BitInput branches) are not reached by
        // every tiling flow, so exercise them directly here
        EXPECT_EQ(aswGeneralTiling.GetSizeWithDataType(10, ge::DT_INT4), 5UL);
        EXPECT_EQ(aswGeneralTiling.GetSizeWithDataType(11, ge::DT_INT4), 0UL); // odd size -> report err, return 0
        EXPECT_EQ(aswGeneralTiling.GetSizeWithDataType(10, ge::DT_FLOAT16), 20UL);
        EXPECT_EQ(aswGeneralTiling.GetSizeWithDataType(10, ge::DT_FLOAT8_E8M0), 10UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, ge::DT_INT4), 64UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, ge::DT_FLOAT4_E2M1), 64UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, ge::DT_FLOAT4_E1M2), 64UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, Ops::NN::DT_HIFLOAT4), 64UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, ge::DT_FLOAT16), 16UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, ge::DT_FLOAT), 8UL);
        EXPECT_EQ(aswGeneralTiling.GetShapeWithDataType(32, ge::DT_UINT64), 4UL);
        // CalUsedCoreNum(mTile, nTile) is inlined into CalcTailBasicBlock in the op-host object, so the
        // out-of-line body never shows up in coverage; call it directly to exercise it (m*n*tailWinBlockCnt)
        EXPECT_EQ(aswGeneralTiling.CalUsedCoreNum(2U, 3U), 2U * 3U * aswGeneralTiling.adaptiveWin_.tailWinBlockCnt);
    }

    if (param.whiteBoxNzSplit) {
        // NZ + no-transB never reaches IsInValidWeighNzTailSplit from CalcTailBasicBlock (early return),
        // call it directly to cover the !transB leg of the baseNAlignSize ternary
        EXPECT_FALSE(aswGeneralTiling.IsInValidWeighNzTailSplit(2, false));
    }
}

TEST_P(TestWeightQuantBatchMatmulV2ASWGeneralTiling, generalTest)
{
    ASWGeneralTilingCaseParam param = GetParam();
    TestASWGeneralOneCase(param);
}

// format: m k n antiQuantOffsetExistFlag biasFlag transA transB group xDtype weightDtype
//         antiQuantScaleDtype yDtype aicNum aivNum weightFormat(0:ND,1:NZ)
//         [ubSize l0cSize tilingDataCap]
// Note: group value: -1: per channel, 0: per tensor, > 0: per group
// cases live in the csv next to this file: caseName + expectation/config columns, one line per
// case ('#' comment lines are skipped); -1 means skip the check
static std::vector<ASWGeneralTilingCaseParam> GetParams()
{
    std::vector<ASWGeneralTilingCaseParam> params;
    const std::string rootPath(ut_str::GetExeDirPath() + "../../../../");
    const std::string casePath(rootPath +
                               "matmul/weight_quant_batch_matmul_v2/tests/ut/op_host/"
                               "test_weight_quant_batch_matmul_v2_adaptive_sliding_window_general_tiling.csv");
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
        if (cols.size() < 12UL) {
            std::cout << "skip invalid csv line: " << line << std::endl;
            continue;
        }

        ASWGeneralTilingCaseParam param;
        param.caseName = Trim(cols[0]);
        param.expectSuccess = ParseBool(Trim(cols[1]));
        param.runFramework = ParseBool(Trim(cols[2]));
        param.whiteBoxUtils = ParseBool(Trim(cols[3]));
        param.expectBlockDim = static_cast<uint32_t>(stoul(Trim(cols[4])));
        param.expectEnablePreAlloc = stoi(Trim(cols[5]));
        param.expectDbL0c = stoi(Trim(cols[6]));
        const string mTileStr = Trim(cols[7]);
        param.expectMTailTile = mTileStr == "-1" ? TILE_SKIP_U32 : static_cast<uint32_t>(stoul(mTileStr));
        const string nTileStr = Trim(cols[8]);
        param.expectNTailTile = nTileStr == "-1" ? TILE_SKIP_U32 : static_cast<uint32_t>(stoul(nTileStr));
        param.disableTailWinLogic = ParseBool(Trim(cols[9]));
        param.noWorkspace = ParseBool(Trim(cols[10]));
        param.whiteBoxNzSplit = ParseBool(Trim(cols[11]));
        params.push_back(param);
    }
    return params;
}

INSTANTIATE_TEST_CASE_P(MM, TestWeightQuantBatchMatmulV2ASWGeneralTiling, testing::ValuesIn(GetParams()));
