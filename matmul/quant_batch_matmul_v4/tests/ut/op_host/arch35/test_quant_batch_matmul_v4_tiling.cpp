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
#include "tiling/tiling_type.h"
#include "common/utils/ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "op_tiling/op_tiling_util.h"
#include "../../../../op_host/op_tiling/arch35/quant_batch_matmul_v4_tiling.h"
#include "../../../../../../built-in/tests/ut/op_tiling_test/common_unittest.h"
#include "../../../../../../built-in/tests/ut/op_tiling_test/test_cube_util.h"

using namespace std;

struct QuantBatchMatmulV4TilingTestParam {
    string caseName;
    // output
    uint32_t blockDim;
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
        {"UINT64", ge::DT_UINT64},
        {"FP8-E8M0", ge::DT_FLOAT8_E8M0},
        {"FP8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FP4-E2M1", ge::DT_FLOAT4_E2M1},
        {"FP4-E1M2", ge::DT_FLOAT4_E1M2}
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
    if (x2Format == ge::FORMAT_ND) {
        if (transB) {
            x2Shape.MutableStorageShape() = gert::Shape({n, k});
        } else {
            x2Shape.MutableStorageShape() = gert::Shape({k, n});
        }
    } else if (x2Format == ge::FORMAT_FRACTAL_NZ) {
        if (transB) {
            x2Shape.MutableStorageShape() = gert::Shape({k1, n1, n0, k0});
        } else {
            x2Shape.MutableStorageShape() = gert::Shape({n1, k1, k0, n0});
        }
    }
    biasShape.MutableOriginShape() = gert::Shape({1, n});
    biasShape.MutableStorageShape() = gert::Shape({1, n});
    int64_t groupSize = 0;
    if (group > 0) {
        groupSize = group;
        int64_t groupNum = (k + group - 1) / group;
        x1ScaleShape.MutableStorageShape() = gert::Shape({m, groupNum});
        if (transB) {
            x2ScaleShape.MutableStorageShape() = gert::Shape({n, groupNum});
        } else {
            x2ScaleShape.MutableStorageShape() = gert::Shape({groupNum, n});
        }
    } else if (group < 0) {
        x2ScaleShape.MutableStorageShape() = gert::Shape({n});
    } else {
        x2ScaleShape.MutableStorageShape() = gert::Shape({1});
    }
    yScaleShape.MutableStorageShape() = gert::Shape({1, n});


    x1Shape.MutableStorageShape() = x1Shape.MutableStorageShape();
    x2Shape.MutableStorageShape() = x2Shape.MutableStorageShape();
    x2Shape.MutableOriginShape() = x2Shape.MutableOriginShape();
    x2ScaleShape.MutableStorageShape() = x2ScaleShape.MutableStorageShape();

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
                                    hasYScale ? &yScaleShape : nullptr, nullptr, nullptr, nullptr, nullptr})
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
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910D"));
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

TEST_P(TestQuantBatchMatmulV4Tiling, generalTest)
{
    QuantBatchMatmulV4TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: caseName m k n transA transB groupSize x1Format x2Format x1Dtype x2Dtype x2ScaleDtype yScaleDtype yDtype
//         aicNum aivNum platform weightFormat
static QuantBatchMatmulV4TilingTestParam casesParams[] = {
    // PERGROUP ND
    {"UT-A8W4-PerGroup-ND-Testcase-0_128_128_128_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-1_17_4096_1024_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-2_4097_1024_1024_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-3_16_10240_1024_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-4_256_1024_64_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-5_17_256_1024_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-6_257_1024_64_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 17, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-7_16_256_1024_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-8_16_65472_64_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 2, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-9_65472_64_64_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},
    {"UT-A8W4-PerGroup-ND-Testcase-10_17_256_65472_0_1_32_ND_ND_FP8-E4M3_FP4-E1M2_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 310UL},

    // MX ND
    {"mx-1_128_512_128_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 410UL},
    {"mx-menkan40_944_7680_256_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 30, ge::GRAPH_SUCCESS, 410UL},
    {"mx-menkan18_736_1536_2800_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 410UL},
    {"mx-menkan17_320_1536_224_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 28, ge::GRAPH_SUCCESS, 410UL},
    {"mx-menkan12_48_7680_80_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 9, ge::GRAPH_SUCCESS, 410UL},
    {"mx-random0001_608_1024_704_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 30, ge::GRAPH_SUCCESS, 410UL},
    {"mx-random0003_3840_512_3200_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 410UL},
    {"mx-random0013_2256_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 410UL},
    {"mx-random0022_32_3072_64_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 4, ge::GRAPH_SUCCESS, 410UL},
    {"mx-random0025_2720_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 410UL},
    {"mx-error-x1ScaleDtype_2720_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_BF16_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 410UL},
    {"mx-error-x2ScaleDtype_2720_512_192_0_1_32_ND_ND_FP8-E4M3_FP4-E2M1_BF16_FP8-E8M0_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 410UL},

    // PERGROUP NZ
    {"UT-A8W4-PerGroup-NZ-Testcase-0_848_640_896_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-1_96_8960_8384_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-2_176_64_128_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 22, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-3_432_192_832_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 30, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-4_592_1856_192_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 30, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-5_605_2304_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 19, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-6_8_7168_576_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 9, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-7_714_128_3392_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-8_5968_128_576_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-9_27_64_320_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 10, ge::GRAPH_SUCCESS, 10300UL},
    {"UT-A8W4-PerGroup-NZ-Testcase-10_192_3648_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_NULL_NULL_BF16_UINT64_BF16_32_64", 12, ge::GRAPH_SUCCESS, 10300UL},

    // MX NZ
    {"UT-A8W4-MX-NZ-Testcase-0_128_128_128_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-1_17_4096_1024_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-2_4097_1024_1024_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-3_16_10240_1024_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-4_256_1024_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-5_17_256_1024_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-6_257_1024_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 17, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-7_16_256_1024_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-8_16_65472_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 4, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-9_65472_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NZ-Testcase-10_17_256_65472_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},
    {"UT-A8W4-MX-NK-NZ-Testcase-0_128_128_128_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_UINT64_BF16_32_64", 32, ge::GRAPH_SUCCESS, 10410UL},

    // PERGROUP NZ ERROR
    // BIAS: dtype != bf16 dtype != fp16
    {"UT-A8W4-PerGroup-NZ-Testcase-error-0_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_FP32_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // X2Scale dtype != bf16
    {"UT-A8W4-PerGroup-NZ-Testcase-error-1_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_FP32_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // X1Scale 不为空
    {"UT-A8W4-PerGroup-NZ-Testcase-error-3_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_BF16_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // GroupSize不为32
    {"UT-A8W4-PerGroup-NZ-Testcase-error-5_16_64_64_0_0_96_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // GroupNum不为整数倍
    {"UT-A8W4-PerGroup-NZ-Testcase-error-6_16_64_64_0_0_65_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // yScale非UINT64
    {"UT-A8W4-PerGroup-NZ-Testcase-error-7_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_FP32_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // yScale为空
    {"UT-A8W4-PerGroup-NZ-Testcase-error-8_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // K不为64对齐
    {"UT-A8W4-PerGroup-NZ-Testcase-error-9_16_65_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // k>65535
    {"UT-A8W4-PerGroup-NZ-Testcase-error-11_16_65536_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // n>65525
    {"UT-A8W4-PerGroup-NZ-Testcase-error-12_16_64_65536_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // x2: dtype!=fp4_e2m1
    {"UT-A8W4-PerGroup-NZ-Testcase-error-13_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP8-E4M3_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // y: dtype!=bf16
    {"UT-A8W4-PerGroup-NZ-Testcase-error-14_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_FP16_32_64", 32, ge::GRAPH_FAILED, 10300UL},
    // x2 NZ不支持转置
    {"UT-A8W4-PerGroup-NZ-Testcase-error-15_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E2M1_BF16_NULL_BF16_UINT64_BF16_32_64", 32, ge::GRAPH_FAILED, 10300UL},

    // MX NZ ERROR
    // X1: dtype != FP8-E4M3
    {"UT-A8W4-MX-NZ-Testcase-error-0_16_64_64_0_1_32_ND_NZ_FP4-E1M2_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // X1: NZ
    {"UT-A8W4-MX-NZ-Testcase-error-1_16_64_64_0_1_32_NZ_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // X1: 非不转置
    {"UT-A8W4-MX-NZ-Testcase-error-2_16_64_64_1_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // X2: dtype != FP4-E1M2 dtype != FP4-E2M1
    {"UT-A8W4-MX-NZ-Testcase-error-3_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP8-E4M3_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // X2: 非转置
    {"UT-A8W4-MX-NZ-Testcase-error-4_16_64_64_0_0_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // y: dtype != bf16
    {"UT-A8W4-MX-NZ-Testcase-error-5_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_FP16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // n > 65535
    {"UT-A8W4-MX-NZ-Testcase-error-6_16_64_65536_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // k > 65535
    {"UT-A8W4-MX-NZ-Testcase-error-7_16_65536_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // k 不为64对齐
    {"UT-A8W4-MX-NZ-Testcase-error-8_16_65_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // n 不为64对齐
    {"UT-A8W4-MX-NZ-Testcase-error-9_16_64_65_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // BIAS: dtype != bf16 dtype != fp16
    {"UT-A8W4-MX-NZ-Testcase-error-10_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_FP32_FP8-E8M0_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // X1Scale: dtype != FP8-E8M0
    {"UT-A8W4-MX-NZ-Testcase-error-11_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP32_FP8-E8M0_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // X2Scale: dtype != FP8-E8M0
    {"UT-A8W4-MX-NZ-Testcase-error-12_16_64_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP32_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // GroupSize 不为32
    {"UT-A8W4-MX-NZ-Testcase-error-13_16_64_64_0_1_96_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP32_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},
    // GroupNum 不为偶数倍
    {"UT-A8W4-MX-NZ-Testcase-error-14_16_80_64_0_1_32_ND_NZ_FP8-E4M3_FP4-E1M2_BF16_FP8-E8M0_FP32_NULL_BF16_32_64", 32, ge::GRAPH_FAILED, 10410UL},

 };

INSTANTIATE_TEST_CASE_P(MM, TestQuantBatchMatmulV4Tiling, testing::ValuesIn(casesParams));
