/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdlib.h>

#include <iostream>
#include <thread>
#include <vector>

#include "log/log.h"

#include "tiling_base/tiling_templates_registry.h"
#include "../../../../common/op_host/op_tiling/tiling_type.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "../../../op_host/op_tiling/arch35/quant_batch_matmul_v4_tiling.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../../common/op_host/math_util.h"

using namespace std;

struct QuantBatchMatmulV4TilingTestParam {
    string caseName;
    // output
    uint32_t numBlocks;
    ge::graphStatus tilingResult;
    uint64_t tilingKey;
};

class TestQuantBatchMatmulV4Tiling : public testing::TestWithParam<QuantBatchMatmulV4TilingTestParam> {};
class TestQuantBatchMatmulV4TilingRegbase
    : public testing::TestWithParam<QuantBatchMatmulV4TilingTestParam> {};

using namespace ge;
using namespace ut_util;
using namespace optiling;

static void SplitStr2Vec(const string &input, const string &delimiter, vector<string> &output)
{
    auto delimiterLen = delimiter.size();
    std::string::size_type currPos = 0;
    std::string::size_type nextPos = input.find(delimiter, currPos);
    while (nextPos != std::string::npos) {
        output.emplace_back(input.substr(currPos, nextPos - currPos));
        currPos = nextPos + delimiterLen;
        nextPos = input.find(delimiter, currPos);
    }

    if (currPos < input.size()) {
        output.emplace_back(input.substr(currPos));
    }
}

static void TestOneParamCase(const QuantBatchMatmulV4TilingTestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    map<string, ge::DataType> dtypeMap = {
        {"FP16", ge::DT_FLOAT16},
        {"FP32", ge::DT_FLOAT},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"INT2", ge::DT_INT2},
        {"UINT1", ge::DT_UINT1},
        {"UINT64", ge::DT_UINT64},
        {"FP8-E8M0", ge::DT_FLOAT8_E8M0},
        {"FP8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FP4-E2M1", ge::DT_FLOAT4_E2M1}
    };

    map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"NZ", ge::FORMAT_FRACTAL_NZ}
    };

    size_t idx = 0;
    string socVersion = testParam[idx++];
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t k0 = 16;
    int64_t k1 = ops::CeilDiv(k, k0);
    int64_t n = stol(testParam[idx++]);
    int64_t n0 = 32;
    int64_t n1 = ops::CeilDiv(n, n0);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    ge::Format x1Format = formatMap[testParam[idx++]];
    ge::Format x2Format = formatMap[testParam[idx++]];
    ge::DataType x1Dtype = dtypeMap[testParam[idx++]];
    ge::DataType x2Dtype = dtypeMap[testParam[idx++]];
    if (transB) {
        k0 = 32;
        k1 = ops::CeilDiv(k, k0);
        n0 = 16;
        n1 = ops::CeilDiv(n, n0);
    }
    bool hasBias = true;
    ge::DataType biasDtype = ge::DT_FLOAT;
    string biasDtypeStr = testParam[idx++];
    if (biasDtypeStr == "NULL") {
        hasBias = false;
    } else {
        biasDtype = dtypeMap[biasDtypeStr];
    }

    bool hasX1Scale = true;
    ge::DataType x1ScaleDtype = ge::DT_FLOAT;
    string x1ScaleDtypeStr = testParam[idx++];
    if (x1ScaleDtypeStr == "NULL") {
        hasX1Scale = false;
    } else {
        x1ScaleDtype = dtypeMap[x1ScaleDtypeStr];
    }

    bool hasX2Scale = true;
    ge::DataType x2ScaleDtype = ge::DT_FLOAT;
    string x2ScaleDtypeStr = testParam[idx++];
    if (x2ScaleDtypeStr == "NULL") {
        hasX2Scale = false;
    } else {
        x2ScaleDtype = dtypeMap[x2ScaleDtypeStr];
    }

    bool hasYScale = true;
    ge::DataType yScaleDtype = ge::DT_FLOAT;
    string yScaleDtypeStr = testParam[idx++];
    if (yScaleDtypeStr == "NULL") {
        hasYScale = false;
    } else {
        yScaleDtype = dtypeMap[yScaleDtypeStr];
    }

    ge::DataType x1OffsetDtype = ge::DT_FLOAT;
    ge::DataType x2OffsetDtype = ge::DT_FLOAT;
    ge::DataType yOffsetDtype = ge::DT_FLOAT;
    bool hasX2Table = true;
    ge::DataType x2TableDtype = ge::DT_FLOAT;
    string x2TableDtypeStr = testParam[idx++];
    if (x2TableDtypeStr == "NULL") {
        hasX2Table = false;
    } else {
        x2TableDtype = dtypeMap[x2TableDtypeStr];
    }

    ge::DataType yDtype = dtypeMap[testParam[idx++]];
    uint32_t aicNum = stoul(testParam[idx++]);
    uint32_t aivNum = stoul(testParam[idx++]);
    string compileInfoStr = R"({
         "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                           "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": aicNum,
                           "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore"}
                            })";
    if (socVersion.find("RESERVED") != string::npos) {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l12bt": true,
                           "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": aicNum,
                           "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore",
                           "Intrinsic_mmad": true, "lut_type": "MTE2_QTABLE"}
                            })";
    }

    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape yScaleShape;
    gert::StorageShape x2TableShape;
    gert::StorageShape outputShape({m, n}, {m, n});

    if (transA) {
        x1Shape.MutableStorageShape() = gert::Shape({k, m});
    } else {
        x1Shape.MutableStorageShape() = gert::Shape({m, k});
    }
    x1Shape.MutableOriginShape() = x1Shape.MutableStorageShape();
    if (x2Format == ge::FORMAT_ND) {
        if (transB) {
            x2Shape.MutableStorageShape() = gert::Shape({n, k});
        } else {
            x2Shape.MutableStorageShape() = gert::Shape({k, n});
        }
        x2Shape.MutableOriginShape() = x2Shape.MutableStorageShape();
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        if (transB) {
            x2Shape.MutableOriginShape() = gert::Shape({n, k});
            x2Shape.MutableStorageShape() = gert::Shape({k1, n1, n0, k0});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({k, n});
            x2Shape.MutableStorageShape() = gert::Shape({n1, k1, k0, n0});
        }
    }
    biasShape.MutableOriginShape() = gert::Shape({1, n});
    biasShape.MutableStorageShape() = gert::Shape({1, n});
    int64_t groupSize = 0;
    int64_t groupK = 0;
    int64_t groupN = 0;
    int64_t groupM = 0;
    if (group > 0) {
        groupSize = group;
        groupK = group & 0xFFFF; // 0-15bit group_k
        groupN = static_cast<int64_t>((group & 0xFFFF0000) >> 16); // 16-31bit group_n
        groupM = static_cast<int64_t>((group & 0xFFFF00000000) >> 32); // 32-47bit group_m
        int64_t groupNum = (k + group - 1) / group;
        if (!hasX2Table) {
            x1ScaleShape.MutableStorageShape() = gert::Shape({m, groupNum});
            if (transB) {
                x2ScaleShape.MutableStorageShape() = gert::Shape({n, groupNum});
            } else {
                x2ScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
            }
        } else {
            x2ScaleShape.MutableStorageShape() = gert::Shape({n});
        }
    } else if (group < 0) {
        x2ScaleShape.MutableStorageShape() = gert::Shape({n});
    } else {
        x2ScaleShape.MutableStorageShape() = gert::Shape({1});
    }
    yScaleShape.MutableStorageShape() = gert::Shape({1, n});
    if (hasX2Table && groupK > 0 && groupN > 0 &&
        (x2Dtype == ge::DT_INT2 || x2Dtype == ge::DT_INT4 || x2Dtype == ge::DT_UINT1)) {
        x2TableShape.MutableStorageShape() = gert::Shape({(k + groupK - 1) / groupK, (n + groupN - 1) / groupN * 16});
    }

    x2ScaleShape.MutableOriginShape() = x2ScaleShape.MutableStorageShape();
    x2TableShape.MutableOriginShape() = x2TableShape.MutableStorageShape();

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> version;
    // 6为替换原aicNum字符串的长度，配置CORE_NUM
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aicNum字符串的长度，配置cube_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aivNum字符串的长度，配置vector_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics, version);
    aicoreSpec["cube_freq"] = "1800";

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    QuantBatchMatmulV4CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    std::string opType("QuantBatchMatmulV4");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, hasBias ? &biasShape : nullptr,
                                    hasX1Scale ? &x1ScaleShape : nullptr, hasX2Scale ? &x2ScaleShape : nullptr,
                                    hasYScale ? &yScaleShape : nullptr, nullptr, nullptr, nullptr, hasX2Table ?
                                    &x2TableShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, x1Dtype, x1Format, x1Format)
                      .NodeInputTd(1, x2Dtype, x2Format, x2Format)
                      .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, yScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, x1OffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, x2OffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, yOffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, x2TableDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)},
                                  {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(groupSize)},
                                  {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                                  {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    gert::TilingContext *tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", version);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ge:graphStatus ret = tilingFunc(tilingContext);
    ASSERT_EQ(ret, param.tilingResult);
    if (ret == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
        ASSERT_EQ(tilingContext->GetBlockDim(), param.numBlocks);
    }
}

TEST_P(TestQuantBatchMatmulV4Tiling, generalTest)
{
    QuantBatchMatmulV4TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: caseName m k n transA transB groupSize x1Format x2Format x1Dtype x2Dtype biasDtype x1ScaleDtype
//         x2ScaleDtype yScaleDtype x2TableDtype yDtype aicNum aivNum platform weightFormat
static QuantBatchMatmulV4TilingTestParam casesParams[] = {
    // MX ND
    {"mx-1_Ascend950_128_512_128_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 17UL},
    {"mx-menkan40_Ascend950_944_7680_256_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 30, ge::GRAPH_SUCCESS, 17UL},
    {"mx-menkan18_Ascend950_736_1536_2800_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 17UL},
    {"mx-menkan17_Ascend950_320_1536_224_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 28, ge::GRAPH_SUCCESS, 17UL},
    {"mx-menkan12_Ascend950_48_7680_80_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 9, ge::GRAPH_SUCCESS, 17UL},
    {"mx-random0001_Ascend950_608_1024_704_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 30, ge::GRAPH_SUCCESS, 17UL},
    {"mx-random0003_Ascend950_3840_512_3200_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 17UL},
    {"mx-random0013_Ascend950_2256_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 17UL},
    {"mx-random0022_Ascend950_32_3072_64_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 4, ge::GRAPH_SUCCESS, 17UL},
    {"mx-random0025_Ascend950_2720_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 17UL},
    {"mx-error-x1ScaleDtype_Ascend950_2720_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_BF16_FP8-E8M0_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 17UL},
    {"mx-error-x2ScaleDtype_Ascend950_2720_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 17UL},

    // PERGROUP NZ
    {"UT-A8W4-PerGroup-NZ-Testcase-0_Ascend950_848_640_896_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-1_Ascend950_96_8960_8384_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-2_Ascend950_176_64_128_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 22, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-3_Ascend950_432_192_832_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 30, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-4_Ascend950_592_1856_192_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 30, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-5_Ascend950_605_2304_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 19, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-6_Ascend950_8_7168_576_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 9, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-7_Ascend950_714_128_3392_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-8_Ascend950_5968_128_576_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-9_Ascend950_27_64_320_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 10, ge::GRAPH_SUCCESS, 268UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-10_Ascend950_192_3648_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 12, ge::GRAPH_SUCCESS, 268UL},

    // k>65535
    {"UT-A8W4-PerGroup-NZ-Testcase-11_Ascend950_16_65536_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 1, ge::GRAPH_SUCCESS, 268UL},
    // n>65535
    {"UT-A8W4-PerGroup-NZ-Testcase-12_Ascend950_16_64_65536_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_SUCCESS, 268UL},

    // PERGROUP NZ ERROR
    // BIAS: dtype != bf16 dtype != fp16
    {"UT-A8W4-PerGroup-NZ-Testcase-error-0_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_FP32_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // X2Scale dtype != bf16
    {"UT-A8W4-PerGroup-NZ-Testcase-error-1_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_FP32_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // X1Scale 不为空
    {"UT-A8W4-PerGroup-NZ-Testcase-error-3_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_BF16_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // GroupSize不为32
    {"UT-A8W4-PerGroup-NZ-Testcase-error-5_Ascend950_16_64_64_0_0_96_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // GroupNum不为整数倍
    {"UT-A8W4-PerGroup-NZ-Testcase-error-6_Ascend950_16_64_64_0_0_65_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // yScale非UINT64
    {"UT-A8W4-PerGroup-NZ-Testcase-error-7_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_FP32_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // yScale为空
    {"UT-A8W4-PerGroup-NZ-Testcase-error-8_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_NULL_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // K不为64对齐
    {"UT-A8W4-PerGroup-NZ-Testcase-error-9_Ascend950_16_65_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // x2: dtype!=fp4_e2m1
    {"UT-A8W4-PerGroup-NZ-Testcase-error-13_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP8-E4M3_BF16_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // y: dtype!=bf16
    {"UT-A8W4-PerGroup-NZ-Testcase-error-14_Ascend950_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_NULL_FP16_32_64", 32, ge::GRAPH_FAILED, 268UL},
    // x2 NZ不支持转置
    {"UT-A8W4-PerGroup-NZ-Testcase-error-15_Ascend950_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 268UL},

    // MX NZ ERROR
    // X2: dtype != FP4-E2M1
    {"UT-A8W4-MX-NZ-Testcase-error-3_Ascend950_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP8-E4M3_BF16_FP8-E8M0_FP8-E8M0_NULL_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 273UL},

    // RESERVED
    {"A8W2-LUT-Testcase-0_RESERVED_3072_2048_4096_0_0_16777344_ND_NZ_INT8_INT2_NULL_NULL_UINT64_NULL_INT4_INT8_14_14", 14, ge::GRAPH_SUCCESS, 768UL},
    {"A8W2-LUT-Testcase-1_RESERVED_1_2048_4096_0_0_16777344_ND_NZ_INT8_INT2_NULL_NULL_UINT64_NULL_INT4_INT8_2_2", 2, ge::GRAPH_SUCCESS, 1280UL},
    {"A8W4-LUT-Testcase-2_RESERVED_3072_2048_4096_0_0_16777344_ND_NZ_INT8_INT4_NULL_NULL_UINT64_NULL_INT8_INT8_14_14", 14, ge::GRAPH_SUCCESS, 768UL},
    {"A8W4-LUT-Testcase-3_RESERVED_1_2048_4096_0_0_16777344_ND_NZ_INT8_INT4_NULL_NULL_UINT64_NULL_INT8_INT8_2_2", 2, ge::GRAPH_SUCCESS, 1280UL},
    {"A8W1-LUT-Testcase-4_RESERVED_3072_2048_4096_0_0_16777344_ND_NZ_INT8_UINT1_NULL_NULL_UINT64_NULL_INT4_INT8_14_14", 14, ge::GRAPH_SUCCESS, 768UL},
    {"A8W1-LUT-Testcase-5_RESERVED_1_2048_4096_0_0_16777344_ND_NZ_INT8_UINT1_NULL_NULL_UINT64_NULL_INT4_INT8_2_2", 2, ge::GRAPH_SUCCESS, 1280UL},
 };
INSTANTIATE_TEST_CASE_P(MM, TestQuantBatchMatmulV4Tiling, testing::ValuesIn(casesParams));
