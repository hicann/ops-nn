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
#include <thread>
#include <mutex>
#include <vector>
#include <fstream>

#include "log/log.h"
#include "nlohmann/json.hpp"

#define protected public
#define private public
#include "tiling_base/tiling_templates_registry.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "matmul/common/op_host/math_util.h"

#include "../../../op_host/op_tiling/dual_level_quant_batch_matmul_adaptive_sliding_window_tiling.h"
#include "../../../op_kernel/dual_level_quant_batch_matmul_tiling_data.h"

using namespace std;
using namespace ge;
using namespace ut_util;
using namespace optiling;

static constexpr uint64_t M = 2048;
static constexpr uint64_t N = 2048;
static constexpr uint64_t K = 512;

class DualLevelQuantBatchMatmulTilingTestParam {
public:
    void Prepare(DualLevelQuantBatchMatmulCompileInfo& compileInfo) const;
    void InvokeTilingFunc(DualLevelQuantBatchMatmulCompileInfo& compileInfo) const;
    void Test() const;
    std::string npuArch;
    std::string caseName;
    bool enable;
    std::string prefix;
    uint64_t coreNum;

    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t level0GroupSize = 512;
    uint64_t level1GroupSize = 32;
    bool transposeX1 = false;
    bool transposeX2 = true;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType x1Level0ScaleDtype;
    ge::DataType x1Level1ScaleDtype;
    ge::DataType x2Level0ScaleDtype;
    ge::DataType x2Level1ScaleDtype;
    ge::DataType biasDtype;
    ge::DataType yDtype;

    bool hasBias;
    bool weightNz;

    // output
    bool result; // false means tiling fail
    uint64_t blockDim;
    uint64_t tilingKey;
    std::string tilingData;
    bool tilingStub; // 是否tililg打桩，只给kernel的用例，此时tiling ut里不校验tiling出参
};

class TestDualLevelQuantBatchMatmulTiling : public testing::TestWithParam<DualLevelQuantBatchMatmulTilingTestParam> {
protected:
    static void SetUpTestCase()
    {}

    static void TearDownTestCase()
    {}
};

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

static string TilingData2Str(const DualLevelQuantBatchMatmulBasicTilingData& tilingData, size_t tilingSize)
{
    string result;
    result.append(std::to_string((uint32_t)tilingData.l1BufferNum) + " ");
    result.append(std::to_string((uint32_t)tilingData.hasBias) + " ");
    result.append(std::to_string((uint32_t)tilingData.l2CacheDisable) + " ");
    result.append(std::to_string((uint32_t)tilingData.usedCoreNum) + " ");
    result.append(std::to_string((uint32_t)tilingData.mSize) + " ");
    result.append(std::to_string((uint32_t)tilingData.nSize) + " ");
    result.append(std::to_string((uint32_t)tilingData.kSize) + " ");
    result.append(std::to_string((uint32_t)tilingData.mL1Size) + " ");
    result.append(std::to_string((uint32_t)tilingData.nL1Size) + " ");
    result.append(std::to_string((uint32_t)tilingData.kL1Size) + " ");
    result.append(std::to_string((uint32_t)tilingData.level0GroupSize) + " ");
    result.append(std::to_string((uint32_t)tilingData.mTailTile) + " ");
    result.append(std::to_string((uint32_t)tilingData.nTailTile) + " ");
    result.append(std::to_string((uint32_t)tilingData.mBaseTailSplitCnt) + " ");
    result.append(std::to_string((uint32_t)tilingData.nBaseTailSplitCnt) + " ");
    result.append(std::to_string((uint32_t)tilingData.mTailMain) + " ");
    result.append(std::to_string((uint32_t)tilingData.nTailMain));
    return result;
}

bool Str2TilingData(const std::string& tilingStr, DualLevelQuantBatchMatmulBasicTilingData& tilingData)
{
    std::vector<string> output;
    output.reserve(sizeof(DualLevelQuantBatchMatmulBasicTilingData) / sizeof(uint32_t));
    SplitStr2Vec(tilingStr, " ", output);
    if (output.size() != sizeof(DualLevelQuantBatchMatmulBasicTilingData) / sizeof(uint32_t)) {
        return false;
    }

    tilingData.l1BufferNum = stol(output[0]);
    tilingData.hasBias = stol(output[1]);
    tilingData.l2CacheDisable = static_cast<L2CacheMode>(stol(output[2]));
    tilingData.usedCoreNum = stol(output[3]);
    tilingData.mSize = stol(output[4]);
    tilingData.nSize = stol(output[5]);
    tilingData.kSize = stol(output[6]);
    tilingData.mL1Size = stol(output[7]);
    tilingData.nL1Size = stol(output[8]);
    tilingData.kL1Size = stol(output[9]);
    tilingData.level0GroupSize = stol(output[10]);
    tilingData.mTailTile = stol(output[11]);
    tilingData.nTailTile = stol(output[12]);
    tilingData.mBaseTailSplitCnt = stol(output[13]);
    tilingData.nBaseTailSplitCnt = stol(output[14]);
    tilingData.mTailMain = stol(output[15]);
    tilingData.nTailMain = stol(output[16]);
    return true;
}

static gert::Shape TransNd2Nz(const gert::Shape& inShape)
{
    constexpr int64_t W0 = 16;
    constexpr int64_t SCALE_RATIO = 2;
    constexpr int64_t H0 = 32 * SCALE_RATIO;

    gert::Shape outShape;
    for (size_t idx = 0; idx < inShape.GetDimNum() - 2; ++idx) {
        outShape.AppendDim(inShape.GetDim(idx));
    }

    int64_t m = inShape.GetDim(inShape.GetDimNum() - 2);
    int64_t n = inShape.GetDim(inShape.GetDimNum() - 1);
    outShape.AppendDim((n + H0 - 1) / H0);
    outShape.AppendDim((m + W0 - 1) / W0);
    outShape.AppendDim(W0);
    outShape.AppendDim(H0);
    return outShape;
}

static void SetNpuArch(DualLevelQuantBatchMatmulCompileInfo& compileInfo, const std::string& npuArch)
{
    static std::unordered_map<std::string, NpuArch> npuArchMap{
        {"DAV_3510", NpuArch::DAV_3510}, {"UnknowNpuArch", NpuArch::DAV_RESV}};
    compileInfo.npuArch = npuArchMap.count(npuArch) > 0 ? npuArchMap[npuArch] : NpuArch::DAV_RESV;
}

static void InitPlatformInfo(
    const std::string& npuArch, gert::TilingContext* tilingContext, string& compileInfoStr, int64_t coreNum = -1)
{
    map<string, string> npuArchInfos = {
        {"SoC_version", npuArch},
    };
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 201326592, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
    if (npuArch.compare("DAV_3510") == 0) {
        compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "intrinsic_fix_pipe_l0c2out_f322bf16": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": true,
                          "Intrinsic_fix_pipe_pre_conv_cast": true,
                          "Intrinsic_data_move_l12bt": true,
                          "UB_SIZE": 245760, "L2_SIZE": 134217728, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 32,
                          "cube_core_cnt": 32, "vector_core_cnt": 64, "core_type_list": "CubeCore,VectorCore",
                          "socVersion": "Ascend950"}
                          })";
    }
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics, npuArchInfos);
    aicoreSpec["cube_freq"] = "1650";

    if (coreNum > 0) {
        socInfos["ai_core_cnt"] = std::to_string(coreNum);
        socInfos["cube_core_cnt"] = std::to_string(coreNum);
        socInfos["vector_core_cnt"] = std::to_string(coreNum * 2);
    }

    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", npuArchInfos);
}

static std::vector<DualLevelQuantBatchMatmulTilingTestParam> GetParams(const std::string& npuArch)
{
    std::vector<DualLevelQuantBatchMatmulTilingTestParam> params;
    std::string rootPath(GetExeDirPath() + "../../../../");
    std::string casePath(
        rootPath + "matmul/dual_level_quant_batch_matmul/tests/ut/op_host/test_dual_level_quant_batch_matmul.csv");
    std::ifstream csvData(casePath, std::ios::in);
    if (!csvData.is_open()) {
        std::cout << "cannot open case file " << casePath << ", maybe not exist" << std::endl;
        return params;
    }

    map<string, ge::DataType> dtypeMap = {
        {"FLOAT16", ge::DT_FLOAT16},
        {"FLOAT", ge::DT_FLOAT},
        {"BF16", ge::DT_BF16},
        {"INT8", ge::DT_INT8},
        {"INT4", ge::DT_INT4},
        {"UINT64", ge::DT_UINT64},
        {"INT32", ge::DT_INT32},
        {"INT64", ge::DT_INT64},
        {"FLOAT8-E8M0", ge::DT_FLOAT8_E8M0},
        {"HIFLOAT8", ge::DT_HIFLOAT8},
        {"FLOAT8-E5M2", ge::DT_FLOAT8_E5M2},
        {"FLOAT8-E4M3", ge::DT_FLOAT8_E4M3FN},
        {"FLOAT4-E2M1", ge::DT_FLOAT4_E2M1},
        {"FLOAT4-E1M2", ge::DT_FLOAT4_E1M2}};

    std::string line;
    while (std::getline(csvData, line)) {
        std::vector<std::string> testParam;
        SplitStr2Vec(line, ",", testParam);

        DualLevelQuantBatchMatmulTilingTestParam param;
        size_t idx = 0UL;
        param.npuArch = testParam[idx++];
        if (param.npuArch != npuArch) {
            continue;
        }

        param.caseName = testParam[idx++];
        param.enable = testParam[idx++] == "true";
        if (!param.enable) {
            continue;
        }
        param.prefix = testParam[idx++];
        auto coreNum = testParam[idx++];
        if (coreNum.empty()) {
            param.coreNum = -1;
        } else {
            param.coreNum = stol(coreNum);
        }

        param.m = stol(testParam[idx++]);
        param.n = stol(testParam[idx++]);
        param.k = stol(testParam[idx++]);
        param.level0GroupSize = stol(testParam[idx++]);
        param.level1GroupSize = stol(testParam[idx++]);
        param.transposeX1 = testParam[idx++] == "true";
        param.transposeX2 = testParam[idx++] == "true";
        param.x1Dtype = dtypeMap[testParam[idx++]];
        param.x2Dtype = dtypeMap[testParam[idx++]];
        param.x1Level0ScaleDtype = dtypeMap[testParam[idx++]];
        param.x1Level1ScaleDtype = dtypeMap[testParam[idx++]];
        param.x2Level0ScaleDtype = dtypeMap[testParam[idx++]];
        param.x2Level1ScaleDtype = dtypeMap[testParam[idx++]];
        param.biasDtype = dtypeMap[testParam[idx++]];
        param.yDtype = dtypeMap[testParam[idx++]];
        param.hasBias = testParam[idx++] == "true";
        param.weightNz = testParam[idx++] == "true";
        param.result = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        param.blockDim = stol(testParam[idx++]);
        param.tilingKey = stol(testParam[idx++]);
        param.tilingData = testParam[idx++];
        param.tilingStub = (strcasecmp(testParam[idx++].c_str(), "true") == 0);
        params.push_back(param);
    }

    return params;
}

void DualLevelQuantBatchMatmulTilingTestParam::Prepare(DualLevelQuantBatchMatmulCompileInfo& compileInfo) const
{
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x1Level0Shape;
    gert::StorageShape x1Level1Shape;
    gert::StorageShape x2Level0Shape;
    gert::StorageShape x2Level1Shape;
    gert::StorageShape outputShape;

    x1Shape.MutableOriginShape() = gert::Shape({m, k});
    x2Shape.MutableOriginShape() = gert::Shape({n, k});
    outputShape.MutableOriginShape() = gert::Shape({m, n});

    x1Level0Shape.MutableStorageShape() = gert::Shape({m, ops::CeilDiv(k, level0GroupSize)});
    x1Level1Shape.MutableStorageShape() = gert::Shape({m, ops::CeilDiv(k, level1GroupSize * 2), 2});
    x2Level0Shape.MutableStorageShape() = gert::Shape({ops::CeilDiv(k, level0GroupSize), n});
    x2Level1Shape.MutableStorageShape() = gert::Shape({n, ops::CeilDiv(k, level1GroupSize * 2), 2});
    biasShape.MutableStorageShape() = gert::Shape({n});

    x1Level0Shape.MutableOriginShape() = x1Level0Shape.MutableStorageShape();
    x1Level1Shape.MutableOriginShape() = x1Level1Shape.MutableStorageShape();
    x2Level0Shape.MutableOriginShape() = x2Level0Shape.MutableStorageShape();
    x2Level1Shape.MutableOriginShape() = x2Level1Shape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();

    x1Shape.MutableStorageShape() = x1Shape.MutableOriginShape();
    if (weightNz) {
        x2Shape.MutableStorageShape() = TransNd2Nz(x2Shape.MutableOriginShape());
    } else {
        x2Shape.MutableStorageShape() = x2Shape.MutableOriginShape();
    }
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    std::string opType("DualLevelQuantBatchMatmul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&x1Shape, &x2Shape, &x1Level0Shape, &x1Level1Shape, &x2Level0Shape, &x2Level1Shape,
                           hasBias ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
                           {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transposeX1)},
                           {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transposeX2)},
                           {"level0_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(level0GroupSize)},
                           {"level1_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(level1GroupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    string compileInfoStr;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    InitPlatformInfo(npuArch, tilingContext, compileInfoStr, coreNum);

    auto kernelHold = gert::KernelRunContextFaker()
                          .KernelIONum(2, 1)
                          .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                          .Outputs({&compileInfo})
                          .Build();

    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

void DualLevelQuantBatchMatmulTilingTestParam::InvokeTilingFunc(DualLevelQuantBatchMatmulCompileInfo& compileInfo) const
{
    gert::StorageShape x1Shape;
    gert::StorageShape x2Shape;
    gert::StorageShape biasShape;
    gert::StorageShape x1Level0Shape;
    gert::StorageShape x1Level1Shape;
    gert::StorageShape x2Level0Shape;
    gert::StorageShape x2Level1Shape;
    gert::StorageShape outputShape;

    printf("\n======= prefix: %s =======\n", prefix.c_str());

    x1Shape.MutableOriginShape() = gert::Shape({m, k});
    x2Shape.MutableOriginShape() = gert::Shape({n, k});
    outputShape.MutableOriginShape() = gert::Shape({m, n});

    x1Level0Shape.MutableStorageShape() = gert::Shape({m, ops::CeilDiv(k, level0GroupSize)});
    x1Level1Shape.MutableStorageShape() = gert::Shape({m, ops::CeilDiv(k, level1GroupSize * 2), 2});
    x2Level0Shape.MutableStorageShape() = gert::Shape({ops::CeilDiv(k, level0GroupSize), n});
    x2Level1Shape.MutableStorageShape() = gert::Shape({n, ops::CeilDiv(k, level1GroupSize * 2), 2});
    biasShape.MutableStorageShape() = gert::Shape({n});

    x1Level0Shape.MutableOriginShape() = x1Level0Shape.MutableStorageShape();
    x1Level1Shape.MutableOriginShape() = x1Level1Shape.MutableStorageShape();
    x2Level0Shape.MutableOriginShape() = x2Level0Shape.MutableStorageShape();
    x2Level1Shape.MutableOriginShape() = x2Level1Shape.MutableStorageShape();
    biasShape.MutableOriginShape() = biasShape.MutableStorageShape();

    x1Shape.MutableStorageShape() = x1Shape.MutableOriginShape();
    if (weightNz) {
        x2Shape.MutableStorageShape() = TransNd2Nz(x2Shape.MutableOriginShape());
    } else {
        x2Shape.MutableStorageShape() = x2Shape.MutableOriginShape();
    }
    outputShape.MutableStorageShape() = outputShape.MutableOriginShape();

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    SetNpuArch(compileInfo, npuArch);

    std::string opType("DualLevelQuantBatchMatmul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(7, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&x1Shape, &x2Shape, &x1Level0Shape, &x1Level1Shape, &x2Level0Shape, &x2Level1Shape,
                           hasBias ? &biasShape : nullptr})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, x2Dtype, ge::FORMAT_ND, weightNz ? ge::FORMAT_FRACTAL_NZ : ge::FORMAT_ND)
                      .NodeInputTd(2, x1Level0ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, x1Level1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, x2Level0ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, x2Level1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, biasDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT16)},
                           {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(transposeX1)},
                           {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(transposeX2)},
                           {"level0_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(level0GroupSize)},
                           {"level1_group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(level1GroupSize)}})
                      .TilingData(rawTilingData.get())
                      .Workspace(workspace)
                      .SetOpType(opType)
                      .Build();

    string compileInfoStr;
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    InitPlatformInfo(npuArch, tilingContext, compileInfoStr, coreNum);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    if (result) {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
        if (tilingStub) {
            return;
        }
        EXPECT_EQ(tilingContext->GetTilingKey(), hasBias ? 0x38D0000 : 0x28D0000);
        EXPECT_EQ(tilingContext->GetBlockDim(), blockDim);

        DualLevelQuantBatchMatmulBasicTilingData& actualTilingData =
            *reinterpret_cast<DualLevelQuantBatchMatmulBasicTilingData*>(tilingContext->GetRawTilingData()->GetData());
        // 这里通过重置预期结果里的部分字段来忽略不关心的tiling字段，后续有新增的话可以仿照这个方法来忽略其他字段
        // expectTilingData.shareL1Size = actualTilingData.shareL1Size;
        string actualTilingDataStr = TilingData2Str(actualTilingData, tilingContext->GetRawTilingData()->GetDataSize());
        ASSERT_EQ(actualTilingDataStr, tilingData);
    } else {
        ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_FAILED);
    }
}

void DualLevelQuantBatchMatmulTilingTestParam::Test() const
{
    DualLevelQuantBatchMatmulCompileInfo compileInfo;
    Prepare(compileInfo);
    InvokeTilingFunc(compileInfo);
}

static void ThreadFunc(
    const DualLevelQuantBatchMatmulTilingTestParam* params, size_t testcaseNum, size_t threadIdx, size_t threadNum)
{
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    for (size_t idx = threadIdx; idx < testcaseNum; idx += threadNum) {
        params[idx].Test();
    }
}

static void TestMultiThread(
    const DualLevelQuantBatchMatmulTilingTestParam* params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcaseNum, idx, threadNum);
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }
}

TEST_P(TestDualLevelQuantBatchMatmulTiling, generalTest)
{
    GetParam().Test();
}

INSTANTIATE_TEST_CASE_P(DLQBMM_DAV_3510, TestDualLevelQuantBatchMatmulTiling, testing::ValuesIn(GetParams("DAV_3510")));

mutex compileMutex;

static void ThreadFuncInvokeTilingFunc(
    const DualLevelQuantBatchMatmulTilingTestParam* params, size_t testcaseNum, size_t threadIdx, size_t threadNum,
    DualLevelQuantBatchMatmulCompileInfo& compileInfo)
{
    if (threadIdx >= testcaseNum)
        return;
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    params[threadIdx].InvokeTilingFunc(compileInfo);
}

static void ThreadFuncPrepare(
    const DualLevelQuantBatchMatmulTilingTestParam* params, size_t testcaseNum, size_t threadIdx, size_t threadNum,
    map<size_t, DualLevelQuantBatchMatmulCompileInfo>& compileInfos)
{
    if (threadIdx >= testcaseNum)
        return;
    int32_t logLevel = 0;
    int32_t enableEvent = 0;
    DualLevelQuantBatchMatmulCompileInfo compileInfo;
    params[threadIdx].Prepare(compileInfo);

    {
        lock_guard<mutex> lock(compileMutex);
        compileInfos[threadIdx] = compileInfo;
    }
}

static void TestMultiThreadSeparate(
    const DualLevelQuantBatchMatmulTilingTestParam* params, size_t testcaseNum, size_t threadNum)
{
    std::thread threads[threadNum];
    map<size_t, DualLevelQuantBatchMatmulCompileInfo> compileInfos;
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx] = std::thread(ThreadFuncPrepare, params, testcaseNum, idx, threadNum, std::ref(compileInfos));
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threads[idx].join();
    }

    std::thread threadsInvoke[threadNum];
    for (size_t idx = 0; idx < threadNum; ++idx) {
        threadsInvoke[idx] =
            std::thread(ThreadFuncInvokeTilingFunc, params, testcaseNum, idx, threadNum, std::ref(compileInfos[idx]));
    }

    for (size_t idx = 0; idx < threadNum; ++idx) {
        threadsInvoke[idx].join();
    }
}