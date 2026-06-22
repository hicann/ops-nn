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

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "log/log.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_inplace_add_tiling.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "../../../../common/op_host/math_util.h"
#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/quant_batch_matmul_v3_tiling_util.h"
#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_compile_info.h"
#include "../../../../../tests/ut/common/ut_string_utils.h"

using namespace std;
using namespace ut_str;
using namespace QuantBatchMatmulInplaceAddArch35TilingKey;

namespace {

using QuantBatchMatmulInplaceAddCompileInfo = optiling::QuantBatchMatmulV3CompileInfo;

struct QuantBatchMatmulInplaceAddTilingTestParam {
    string caseGroup;
    string caseName;
    string socVersion;
    int64_t m;
    int64_t k;
    int64_t n;
    bool transA;
    bool transB;
    int64_t groupSize;
    ge::Format x1Format;
    ge::Format x2Format;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    bool hasX1Scale;
    ge::DataType x1ScaleDtype;
    bool hasX2Scale;
    ge::DataType x2ScaleDtype;
    ge::DataType yDtype;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t expectBlockDim;
    ge::graphStatus expectTilingResult;
    uint64_t expectTilingKey;
};

bool IsMxFp8Param(const QuantBatchMatmulInplaceAddTilingTestParam& param)
{
    return param.groupSize > 0 &&
           (param.x1Dtype == ge::DT_FLOAT8_E4M3FN || param.x1Dtype == ge::DT_FLOAT8_E5M2) &&
           (param.x2Dtype == ge::DT_FLOAT8_E4M3FN || param.x2Dtype == ge::DT_FLOAT8_E5M2);
}

bool ShouldSkipMxCheckWithoutTensorApi(const QuantBatchMatmulInplaceAddTilingTestParam& param)
{
    return param.socVersion == "Ascend950" && IsMxFp8Param(param) && !optiling::IsTensorapiCapable();
}

struct QuantBatchMatmulInplaceAddTilingCsvLoadResult {
    vector<QuantBatchMatmulInplaceAddTilingTestParam> params;
    vector<string> errors;
};

static QuantBatchMatmulInplaceAddTilingCsvLoadResult LoadParams()
{
    QuantBatchMatmulInplaceAddTilingCsvLoadResult result;
    string rootPath(ut_str::GetExeDirPath() + "../../../../");
    string casePath(
        rootPath + "matmul/quant_batch_matmul_inplace_add/tests/ut/op_host/test_quant_batch_matmul_inplace_add_tiling.csv");
    ifstream csvData(casePath, ios::in);
    if (!csvData.is_open()) {
        result.errors.push_back("cannot open case file: " + casePath);
        return result;
    }

    string line;
    bool skipHeader = true;
    constexpr size_t kExpectedCols = 23UL;
    size_t lineNo = 0UL;
    while (getline(csvData, line)) {
        ++lineNo;
        const string trimLine = Trim(line);
        if (trimLine.empty() || trimLine[0] == '#') {
            continue;
        }
        if (skipHeader) {
            skipHeader = false;
            continue;
        }

        vector<string> cols;
        SplitStr2Vec(line, ",", cols);
        if (cols.size() < kExpectedCols) {
            result.errors.push_back("skip invalid csv line " + std::to_string(lineNo) + " in " + casePath +
                                    ": expected at least " + std::to_string(kExpectedCols) + " columns, got " +
                                    std::to_string(cols.size()));
            continue;
        }

        size_t idx = 0UL;
        QuantBatchMatmulInplaceAddTilingTestParam param;
        param.caseGroup = Trim(cols[idx++]);
        param.caseName = Trim(cols[idx++]);
        param.socVersion = Trim(cols[idx++]);
        param.m = ParseInt64OrDefault(cols[idx++], 0);
        param.k = ParseInt64OrDefault(cols[idx++], 0);
        param.n = ParseInt64OrDefault(cols[idx++], 0);
        param.transA = ParseBool(cols[idx++]);
        param.transB = ParseBool(cols[idx++]);
        param.groupSize = ParseInt64OrDefault(cols[idx++], 0);
        param.x1Format = ParseFormat(cols[idx++]);
        param.x2Format = ParseFormat(cols[idx++]);
        param.x1Dtype = ParseDtype(cols[idx++]);
        param.x2Dtype = ParseDtype(cols[idx++]);
        param.hasX1Scale = ParseBool(cols[idx++]);
        param.x1ScaleDtype = ParseDtype(cols[idx++]);
        param.hasX2Scale = ParseBool(cols[idx++]);
        param.x2ScaleDtype = ParseDtype(cols[idx++]);
        param.yDtype = ParseDtype(cols[idx++]);
        param.aicNum = static_cast<uint32_t>(ParseIntOrDefault(cols[idx++], 0));
        param.aivNum = static_cast<uint32_t>(ParseIntOrDefault(cols[idx++], 0));
        param.expectBlockDim = static_cast<uint32_t>(ParseIntOrDefault(cols[idx++], 0));
        param.expectTilingResult = ParseGraphStatus(cols[idx++]);
        param.expectTilingKey = static_cast<uint64_t>(ParseInt64OrDefault(cols[idx++], 0));
        result.params.push_back(param);
    }
    if (result.params.empty()) {
        result.errors.push_back("no valid tiling cases loaded from: " + casePath);
    }
    return result;
}

static const QuantBatchMatmulInplaceAddTilingCsvLoadResult &GetParamsLoadResult()
{
    static const QuantBatchMatmulInplaceAddTilingCsvLoadResult result = LoadParams();
    return result;
}

static vector<QuantBatchMatmulInplaceAddTilingTestParam> GetParams()
{
    return GetParamsLoadResult().params;
}

static string SanitizeCaseName(const testing::TestParamInfo<QuantBatchMatmulInplaceAddTilingTestParam> &info)
{
    string name = info.param.caseName;
    for (char &ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }
    return name;
}

class QuantBatchMatmulInplaceAddTiling : public testing::TestWithParam<QuantBatchMatmulInplaceAddTilingTestParam> {
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

TEST(QuantBatchMatmulInplaceAddTilingCsv, ShouldLoadValidCases)
{
    const auto &loadResult = GetParamsLoadResult();
    for (const auto &error : loadResult.errors) {
        ADD_FAILURE() << error;
    }
    EXPECT_FALSE(loadResult.params.empty());
}

static void TestOneParamCase(const QuantBatchMatmulInplaceAddTilingTestParam &param)
{
    std::cout << "run case " << param.caseName << std::endl;
    const int64_t k0 = 64;
    const int64_t k1 = ops::CeilDiv(param.k, k0);
    const int64_t n0 = 64;
    const int64_t n1 = ops::CeilDiv(param.n, n0);

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
    gert::StorageShape x1ScaleShape;
    gert::StorageShape x2ScaleShape;
    gert::StorageShape yInputShape;
    gert::StorageShape outputShape({param.m, param.n}, {param.m, param.n});

    if (param.transA) {
        x1Shape.MutableStorageShape() = gert::Shape({param.k, param.m});
    } else {
        x1Shape.MutableStorageShape() = gert::Shape({param.m, param.k});
    }
    x1Shape.MutableOriginShape() = x1Shape.MutableStorageShape();
    if (param.x2Format == ge::FORMAT_ND) {
        if (param.transB) {
            x2Shape.MutableStorageShape() = gert::Shape({param.n, param.k});
        } else {
            x2Shape.MutableStorageShape() = gert::Shape({param.k, param.n});
        }
        x2Shape.MutableOriginShape() = x2Shape.MutableStorageShape();
    } else if (param.x2Format == ge::FORMAT_FRACTAL_NZ) {
        if (param.transB) {
            x2Shape.MutableOriginShape() = gert::Shape({param.n, param.k});
            x2Shape.MutableStorageShape() = gert::Shape({k1, n1, n0, k0});
        } else {
            x2Shape.MutableOriginShape() = gert::Shape({param.k, param.n});
            x2Shape.MutableStorageShape() = gert::Shape({n1, k1, k0, n0});
        }
    }

    int64_t groupSize = 0;
    if (param.groupSize > 0) {
        groupSize = param.groupSize;
        const int64_t groupNum = (param.k + param.groupSize - 1) / param.groupSize;
        x1ScaleShape.MutableStorageShape() = gert::Shape({param.m, groupNum, 2});
        if (param.transB) {
            x2ScaleShape.MutableStorageShape() = gert::Shape({param.n, groupNum, 2});
        } else {
            x2ScaleShape.MutableStorageShape() = gert::Shape({groupNum, param.n, 2});
        }
    }
    yInputShape.MutableOriginShape() = gert::Shape({param.m, param.n});
    yInputShape.MutableStorageShape() = gert::Shape({param.m, param.n});
    if (param.x1Dtype == ge::DT_HIFLOAT8 && param.x2Dtype == ge::DT_HIFLOAT8 &&
        param.x1ScaleDtype == ge::DT_FLOAT && param.x2ScaleDtype == ge::DT_FLOAT) {
        x1ScaleShape.MutableOriginShape() = gert::Shape({1});
        x1ScaleShape.MutableStorageShape() = gert::Shape({1});
        x2ScaleShape.MutableOriginShape() = gert::Shape({1});
        x2ScaleShape.MutableStorageShape() = gert::Shape({1});
    } else {
        x1ScaleShape.MutableOriginShape() = gert::Shape({k1, param.m, 2});
        x1ScaleShape.MutableStorageShape() = gert::Shape({k1, param.m, 2});
        x2ScaleShape.MutableOriginShape() = gert::Shape({k1, param.n, 2});
        x2ScaleShape.MutableStorageShape() = gert::Shape({k1, param.n, 2});
    }

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(param.aicNum));
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aicNum"), 6, to_string(param.aicNum));
    compileInfoStr = compileInfoStr.replace(compileInfoStr.find("aivNum"), 6, to_string(param.aivNum));
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

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
                      .InputShapes({&x1Shape, &x2Shape, param.hasX2Scale ? &x2ScaleShape : nullptr,
                                    &yInputShape, param.hasX1Scale ? &x1ScaleShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, param.x1Dtype, param.x1Format, param.x1Format)
                      .NodeInputTd(1, param.x2Dtype, param.x2Format, param.x2Format)
                      .NodeInputTd(2, param.x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, param.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, param.x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(param.transA)},
                                  {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(param.transB)},
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
    map<string, string> socVersionInfos = {
        {"Short_SoC_version", param.socVersion},
        {"SoC_version", param.socVersion},
    };
    const map<string, string> soc2Arch = {{"Ascend950", "3510"}};
    auto soc2ArchIter = soc2Arch.find(param.socVersion);
    if (soc2ArchIter != soc2Arch.end()) {
        socVersionInfos["NpuArch"] = soc2ArchIter->second;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ge::graphStatus ret = tilingFunc(tilingContext);
    ASSERT_EQ(ret, param.expectTilingResult);
    if (ret == ge::GRAPH_SUCCESS) {
        if (ShouldSkipMxCheckWithoutTensorApi(param)) {
            return;
        }
        ASSERT_EQ(tilingContext->GetTilingKey(), param.expectTilingKey);
        ASSERT_EQ(tilingContext->GetBlockDim(), param.expectBlockDim);
        size_t expectTilingDataSize = IsMxFp8Param(param) ?
            sizeof(QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData) :
            sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData);
        ASSERT_EQ(tilingContext->GetRawTilingData()->GetDataSize(), expectTilingDataSize);
    }
}

TEST_P(QuantBatchMatmulInplaceAddTiling, generalTest)
{
    TestOneParamCase(GetParam());
}

INSTANTIATE_TEST_SUITE_P(QuantBatchMatmulInplaceAdd950,
                         QuantBatchMatmulInplaceAddTiling,
                         testing::ValuesIn(GetParams()),
                         SanitizeCaseName);

} // namespace
