/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <vector>
#include <string>
#include <algorithm>

#include "log/log.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_inplace_add_tiling.h"
#include "../../../../common/op_host/math_util.h"

using namespace std;

namespace {

struct QuantBatchMatmulInplaceAddCompileInfo {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l0aSize;
    uint64_t l0bSize;
    uint32_t workspaceNum;
    uint32_t aivNum;
    uint32_t aicNum;
    bool supportL0c2Out;
    bool supportL12BtBf16;
    bool supportMmadS8S4;
    platform_ascendc::SocVersion socVersion;
};

struct QuantBatchMatmulInplaceAddTilingTestParam {
    string caseName;
    // output
    uint32_t blockDim;
    ge::graphStatus tilingResult;
    uint64_t tilingKey;
};

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

class QuantBatchMatmulInplaceAddTiling : public testing::TestWithParam<QuantBatchMatmulInplaceAddTilingTestParam> {
    virtual void SetUp()
    {}

protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantBatchMatmulInplaceAddTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantBatchMatmulInplaceAddTiling TearDown" << std::endl;
    }
};

class QuantBatchMatmulInplaceAddTilingRegbase
    : public testing::TestWithParam<QuantBatchMatmulInplaceAddTilingTestParam> {};


static void TestOneParamCase(const QuantBatchMatmulInplaceAddTilingTestParam &param)
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
        {"FP8-E5M2", ge::DT_FLOAT8_E5M2},
        {"FP4-E2M1", ge::DT_FLOAT4_E2M1},
        {"FP4-E1M2", ge::DT_FLOAT4_E1M2}
    };

    map<string, ge::Format> formatMap = {
        {"ND", ge::FORMAT_ND},
        {"NZ", ge::FORMAT_FRACTAL_NZ}
    };

    size_t idx = 0;
    string socVersion = testParam[idx++];
    std::replace(socVersion.begin(), socVersion.end(), '-', '_');
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t k0 = 64;
    int64_t k1 = ops::CeilDiv(k, k0);
    int64_t n = stol(testParam[idx++]);
    int64_t n0 = 64;
    int64_t n1 = ops::CeilDiv(n, n0);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    ge::Format x1Format = formatMap[testParam[idx++]];
    ge::Format x2Format = formatMap[testParam[idx++]];
    ge::DataType x1Dtype = dtypeMap[testParam[idx++]];
    ge::DataType x2Dtype = dtypeMap[testParam[idx++]];

    bool hasBias = false;
    ge::DataType biasDtype = ge::DT_FLOAT;

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

    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape yScaleShape;
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
            x1ScaleShape.MutableStorageShape() = gert::Shape({m, groupNum, 2});
            if (transB) {
                x2ScaleShape.MutableStorageShape() = gert::Shape({n, groupNum, 2});
            } else {
                x2ScaleShape.MutableStorageShape() = gert::Shape({groupNum, n, 2});
            }
        } 
    }
    yScaleShape.MutableStorageShape() = gert::Shape({1, n});
    x1ScaleShape.MutableOriginShape() = gert::Shape({k1, m, 2});
    x1ScaleShape.MutableStorageShape() = gert::Shape({k1, m, 2});
    x2ScaleShape.MutableOriginShape() = gert::Shape({k1, n, 2});
    x2ScaleShape.MutableStorageShape() = gert::Shape({k1, n, 2});

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
    aicoreSpec["cube_freq"] = "1800";

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    QuantBatchMatmulInplaceAddCompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    std::string opType("QuantBatchMatmulInplaceAdd");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, hasBias ? &biasShape : nullptr,
                                    hasX1Scale ? &x1ScaleShape : nullptr, hasX2Scale ? &x2ScaleShape : nullptr,
                                    hasYScale ? &yScaleShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, x1Dtype, x1Format, x1Format)
                      .NodeInputTd(1, x2Dtype, x2Format, x2Format)
                      .NodeInputTd(2, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, yScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
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
    soc_version_infos.insert(make_pair("Short_SoC_version", socVersion));
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

TEST_P(QuantBatchMatmulInplaceAddTiling, generalTest)
{
    QuantBatchMatmulInplaceAddTilingTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: caseName m k n transA transB groupSize x1Format x2Format x1Dtype x2Dtype biasDtype x1ScaleDtype
//         x2ScaleDtype yScaleDtype yDtype aicNum aivNum platform weightFormat
static QuantBatchMatmulInplaceAddTilingTestParam casesParams[] = {

    // MX ND
    {"mx-test1_Ascend910-95_128_512_128_1_0_32_ND_ND_FP8-E4M3_FP8-E5M2_FP8-E8M0_FP8-E8M0_UINT64_NULL_FP32_32_64", 30, ge::GRAPH_SUCCESS, 1UL},
    {"mx-2_Ascend910-95_1024_512_1024_1_0_32_ND_ND_FP8-E5M2_FP8-E5M2_FP8-E8M0_FP8-E8M0_UINT64_NULL_FP32_32_64", 32, ge::GRAPH_SUCCESS, 1UL},
    {"mx-3_Ascend910-95_128_512_128_1_0_32_ND_ND_FP8-E5M2_FP8-E5M2_FP8-E8M0_FP8-E8M0_UINT64_NULL_FP32_32_64", 30, ge::GRAPH_SUCCESS, 1UL},
    {"mx-4_Ascend910-95_1024_512_1024_1_0_32_ND_ND_FP8-E4M3_FP8-E4M3_FP8-E8M0_FP8-E8M0_UINT64_NULL_FP32_32_64", 32, ge::GRAPH_SUCCESS, 1UL},

 };

INSTANTIATE_TEST_CASE_P(QuantBatchMatmulInplaceAdd950, QuantBatchMatmulInplaceAddTiling, testing::ValuesIn(casesParams));

} // using namespace