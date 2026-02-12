/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_v4_compile_info.h"
#include "../../../../common/op_host/math_util.h"

using namespace std;

struct QuantBatchMatmulV4TilingMsdTestParam {
    string socVersion;
    string caseName;

    // output
    uint32_t blockDim;
    ge::graphStatus tilingResult;
    uint64_t tilingKey;
};

class TestQuantBatchMatmulV4TilingMsd : public testing::TestWithParam<QuantBatchMatmulV4TilingMsdTestParam> {};

using namespace ge;
using namespace ut_util;

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
namespace {
    void InitPlatformInfo(const std::string &socVersion,  string &compileInfoStr)
    {
        if (socVersion.compare("Ascend910B4") == 0) {
            compileInfoStr = R"({
                "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                                "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                                "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                                "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524032,
                                "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": aicNum,
                                "cube_core_cnt": aicNum, "vector_core_cnt": aivNum, "core_type_list": "CubeCore,VectorCore"}
                                })";
        }
    }
}

static void TestOneParamCase(const QuantBatchMatmulV4TilingMsdTestParam &param)
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
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t k0 = 16;
    int64_t k1 = ops::CeilDiv(k, k0);
    int64_t n = stol(testParam[idx++]);
    int64_t n0 = 32;
    int64_t n1 = ops::CeilDiv(n, n0);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t groupSize = stol(testParam[idx++]);
    ge::Format x1Format = formatMap[testParam[idx++]];
    ge::Format x2Format = formatMap[testParam[idx++]];
    ge::DataType x1Dtype = dtypeMap[testParam[idx++]];
    ge::DataType x2Dtype = dtypeMap[testParam[idx++]];
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

    bool hasYOffset = true;
    ge::DataType yOffsetDtype = ge::DT_FLOAT;
    string yOffsetDtypeStr = testParam[idx++];
    if (yOffsetDtypeStr == "NULL") {
        hasYOffset = false;
    } else {
        yOffsetDtype = dtypeMap[yOffsetDtypeStr];
    }
    ge::DataType yDtype = dtypeMap[testParam[idx++]];
    uint32_t aicNum = stoul(testParam[idx++]);
    uint32_t aivNum = stoul(testParam[idx++]);
    string compileInfoStr;
    InitPlatformInfo(param.socVersion, compileInfoStr);

    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape yScaleShape;
    gert::StorageShape yOffsetShape;
    gert::StorageShape outputShape({m, n}, {m, n});

    x1Shape.MutableStorageShape() = gert::Shape({m, k});
    x2Shape.MutableStorageShape() = gert::Shape({k, n});
    if (x1Dtype == DT_INT8 && x2Dtype == DT_INT4) {
        if (groupSize != 0) {
            x1ScaleShape.MutableStorageShape() = gert::Shape({m, 1});
            int64_t groupNum = (k + groupSize - 1) / groupSize;
            x2ScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
        } else {
            x1ScaleShape.MutableStorageShape() = gert::Shape({m});
            x2ScaleShape.MutableStorageShape() = gert::Shape({n});
        }
        yOffsetShape.MutableStorageShape() = gert::Shape({n});
        x1Shape.MutableStorageShape() = x1Shape.MutableStorageShape();
        x2Shape.MutableStorageShape() = x2Shape.MutableStorageShape();
        x2Shape.MutableOriginShape() = x2Shape.MutableOriginShape();
        x2ScaleShape.MutableStorageShape() = x2ScaleShape.MutableStorageShape();
    }
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    // 6为替换原aicNum字符串的长度，配置CORE_NUM
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aicNum字符串的长度，配置cube_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(aicNum));
    // 6为替换原aivNum字符串的长度，配置vector_core_cnt
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    if (param.socVersion.compare("Ascend910B4") == 0) {
            aicoreSpec["cube_freq"] = "1650";
    }

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::QuantBatchMatmulV4CompileInfo compileInfo;

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
                                    hasYScale ? &yScaleShape : nullptr, nullptr, nullptr,
                                    hasYOffset ? &yOffsetShape : nullptr, nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, x1Dtype, x1Format, x1Format)
                      .NodeInputTd(1, x2Dtype, x2Format, x2Format)
                      .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, yScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, yOffsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> soc_version_infos;
    if (param.socVersion.compare("Ascend910B4") == 0) {
        soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910B"));
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ge:graphStatus ret = tilingFunc(tilingContext);
    ASSERT_EQ(ret, param.tilingResult);
    if (ret == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
        ASSERT_EQ(tilingContext->GetBlockDim(), param.blockDim);
    }
}

TEST_P(TestQuantBatchMatmulV4TilingMsd, generalTest)
{
    QuantBatchMatmulV4TilingMsdTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: platform
//         caseName m k n transA transB groupSize x1Format x2Format x1Dtype x2Dtype biasDtype x1ScaleDtype x2ScaleDtype yScaleDtype yOffset yDtype aicNum aivNum
//         bolckdim, result, tilingkey
static QuantBatchMatmulV4TilingMsdTestParam casesParams[] = {
    // A int8 W int4
    {"Ascend910B4", "UT-A8W4-MSD-ND-Testcase-0_16_256_256_0_0_256_ND_ND_INT8_INT4_NULL_FP32_UINT64_NULL_FP32_BF16_20_40", 20, ge::GRAPH_SUCCESS, 24UL},
    {"Ascend910B4", "UT-A8W4-MSD-ND-Testcase-1_16_256_256_0_0_256_ND_ND_INT8_INT4_NULL_FP32_UINT64_NULL_FP32_FP16_20_40", 20, ge::GRAPH_SUCCESS, 24UL},
    {"Ascend910B4", "UT-A8W4-MSD-ND-Testcase-2_16_256_256_0_0_0_ND_ND_INT8_INT4_NULL_FP32_UINT64_NULL_FP32_FP16_20_40", 20, ge::GRAPH_SUCCESS, 20UL}
 };

INSTANTIATE_TEST_CASE_P(MMA8W4MSD, TestQuantBatchMatmulV4TilingMsd, testing::ValuesIn(casesParams));
