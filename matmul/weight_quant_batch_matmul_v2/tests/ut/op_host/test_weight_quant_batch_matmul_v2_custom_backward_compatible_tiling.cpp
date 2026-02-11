/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <vector>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;

struct WeightQuantBatchMatmulV2TilingCustomBackwardCompatibleTestParam {
    string caseName;

    // output
    uint32_t blockDim;
    uint64_t tilingKey;
};

class TestWeightQuantBatchMatmulV2CustomBackwardCompatibleTiling : public testing::TestWithParam<WeightQuantBatchMatmulV2TilingCustomBackwardCompatibleTestParam>
{
};

using namespace ge;
using namespace optiling;

static void SplitStr2Vec(const string& input, const string& delimiter, vector<string>& output)
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

static void TestOneParamCase(const WeightQuantBatchMatmulV2TilingCustomBackwardCompatibleTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);
    map<string, ge::DataType> dtypeMap = {
        {"FLOAT16", ge::DT_FLOAT16},
        {"FLOAT", ge::DT_FLOAT},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"UINT64", ge::DT_UINT64},
        {"FLOAT4E2M1", ge::DT_FLOAT4_E2M1},
        {"FLOAT4E1M2", ge::DT_FLOAT4_E1M2},
    };

    size_t idx = 0;
    int64_t m = stol(testParam[idx++]);
    int64_t k = stol(testParam[idx++]);
    int64_t n = stol(testParam[idx++]);
    int64_t antiQuantOffsetExistFlag = stol(testParam[idx++]);
    int64_t quantScaleExistFlag = stol(testParam[idx++]);
    int64_t quantOffsetExistFlag = stol(testParam[idx++]);
    int64_t biasFlag = stol(testParam[idx++]);
    int64_t transA = stol(testParam[idx++]);
    int64_t transB = stol(testParam[idx++]);
    int64_t group = stol(testParam[idx++]);
    ge::DataType xDtype = dtypeMap[testParam[idx++]];
    ge::DataType weightDtype = dtypeMap[testParam[idx++]];
    ge::DataType quantScaleDtype = dtypeMap[testParam[idx++]];
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
    ge::Format bFormat = ge::FORMAT_ND;
    if (stol(testParam[idx++]) == 1) {
        bFormat = ge::FORMAT_FRACTAL_NZ;
    }
    ge::graphStatus expectedStatus = ge::GRAPH_SUCCESS;
    if (stol(testParam[idx++]) == 1) {
        expectedStatus = ge::GRAPH_FAILED;
    }
    ge::DataType biasDtype = xDtype;
    if (xDtype == ge::DT_BF16) {
        biasDtype = ge::DT_FLOAT;
    }
    gert::StorageShape xShape;
    gert::StorageShape weigthShape;
    gert::StorageShape antiQuantScaleShape;
    gert::StorageShape antiQuantOffsetShape;
    gert::StorageShape quantScaleShape;
    gert::StorageShape quantOffsetShape;
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
            weigthShape.MutableStorageShape() = gert::Shape({(k + 32) / 32, (n + 16) / 16L, 16L, 32});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({n, k});
        }
        weigthShape.MutableOriginShape() = gert::Shape({n, k});
    } else {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            weigthShape.MutableStorageShape() = gert::Shape({(n + 32) / 32, (k + 16) / 16L, 16L, 32});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({k, n});
        }
        weigthShape.MutableOriginShape() = gert::Shape({k, n});
    }
    int64_t groupSize = 0;
    if (group > 0) {
        groupSize = group;
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
        quantScaleShape.MutableStorageShape() = gert::Shape({n});
        quantOffsetShape.MutableStorageShape() = gert::Shape({n});
    } else {
        antiQuantOffsetShape.MutableStorageShape() = gert::Shape({1});
        antiQuantScaleShape.MutableStorageShape() = gert::Shape({1});
        quantScaleShape.MutableStorageShape() = gert::Shape({1});
        quantOffsetShape.MutableStorageShape() = gert::Shape({1});
    }

    biasShape.MutableStorageShape() = gert::Shape({n});
    xShape.MutableStorageShape() = xShape.MutableStorageShape();
    xShape.MutableOriginShape() = xShape.MutableOriginShape();
    weigthShape.MutableStorageShape() = weigthShape.MutableStorageShape();
    weigthShape.MutableOriginShape() = weigthShape.MutableOriginShape();
    antiQuantScaleShape.MutableStorageShape() = antiQuantScaleShape.MutableStorageShape();
    antiQuantOffsetShape.MutableStorageShape() = antiQuantOffsetShape.MutableStorageShape();
    quantScaleShape.MutableStorageShape() = quantScaleShape.MutableStorageShape();
    quantOffsetShape.MutableStorageShape() = quantOffsetShape.MutableStorageShape();
    biasShape.MutableStorageShape() = biasShape.MutableStorageShape();

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
    MatmulV3CompileInfo compileInfo;

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    std::string opType("WeightQuantBatchMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&xShape, &weigthShape, &antiQuantScaleShape,
                           antiQuantOffsetExistFlag ? &antiQuantOffsetShape : nullptr,
                           quantScaleExistFlag ? &quantScaleShape : nullptr,
                           quantOffsetExistFlag ? &quantOffsetShape : nullptr, biasFlag ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_ND, bFormat)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, quantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(transA)},
                           {"transpose_weight", Ops::NN::AnyValue::CreateFrom<bool>(transB)},
                           {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(groupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910_95"));
    soc_version_infos.insert(make_pair("NpuArch", "3510"));
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
        ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
        ASSERT_EQ(tilingContext->GetBlockDim(), param.blockDim);
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
    }
}

TEST_P(TestWeightQuantBatchMatmulV2CustomBackwardCompatibleTiling, generalTest)
{
    WeightQuantBatchMatmulV2TilingCustomBackwardCompatibleTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: m k n antiQuantOffsetExistFlag quantScaleExistFlag quantOffsetExistFlag biasFlag transA transB group Xdtype
//         weigthDtype quantScaleDtype yDtype aicNum aivNum weightFormat(0:ND,1:NZ) ExceptionTest
// Note: group value
//       -1: per channel, 1: per tensor, > 1: per group
static WeightQuantBatchMatmulV2TilingCustomBackwardCompatibleTestParam casesParams[] = {
    {"davidCase_16_960_1024_0_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_32_64_1_0", 32, 34661756930UL},
    {"davidCase_16_960_1024_1_0_0_0_0_0_-1_FLOAT16_INT8_UINT64_FLOAT16_32_64_1_0", 32, 38956724226UL},
    {"davidCase_16_960_1024_0_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_32_64_1_0", 32, 34670145538UL},
    {"davidCase_16_960_1024_1_0_0_0_0_1_-1_FLOAT16_INT8_UINT64_FLOAT16_32_64_1_0", 32, 38965112834UL},
    {"davidCase_16_960_1024_0_0_0_0_0_0_64_FLOAT16_INT8_UINT64_FLOAT16_32_64_1_0", 32, 34678534146UL},
    {"davidCase_16_960_1024_1_0_0_0_0_0_64_FLOAT16_INT8_UINT64_FLOAT16_32_64_1_0", 32, 38973501442UL},
};

INSTANTIATE_TEST_CASE_P(MM, TestWeightQuantBatchMatmulV2CustomBackwardCompatibleTiling, testing::ValuesIn(casesParams));