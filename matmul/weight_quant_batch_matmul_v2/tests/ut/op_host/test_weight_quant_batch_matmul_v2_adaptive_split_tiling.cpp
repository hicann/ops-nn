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
#include <vector>
#include <thread>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "../../../op_host/op_tiling/arch35/weight_quant_batch_matmul_v2_adaptive_split_tiling.h"
#include "test_cube_util.h"
#include "../../../../mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"

using namespace std;

struct WeightQuantBatchMatmulV2TilingTestParam {
    string caseName;

    // output
    uint32_t blockDim;
    uint64_t tilingKey;
};

class TestWeightQuantBatchMatmulV2AdaptiveSplitTiling
    : public testing::TestWithParam<WeightQuantBatchMatmulV2TilingTestParam>
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

static void TestOneParamCase(const WeightQuantBatchMatmulV2TilingTestParam& param)
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
        {"FP8E5M2", ge::DT_FLOAT8_E5M2},
        {"FP8E4M3", ge::DT_FLOAT8_E4M3FN},
        {"HIF8", ge::DT_HIFLOAT8},
        {"FLOAT8E8M0", ge::DT_FLOAT8_E8M0},
        {"FLOAT4E2M1", ge::DT_FLOAT4_E2M1},
        {"FLOAT4E1M2", ge::DT_FLOAT4_E1M2}};

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
    ge::DataType antiQuantScaleDtype = dtypeMap[testParam[idx++]];
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
    uint32_t soc = stol(testParam[idx++]);
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
            weigthShape.MutableStorageShape() = gert::Shape({(k + 16) / 16, (n + 16) / 16L, 16L, 16});
        } else {
            weigthShape.MutableStorageShape() = gert::Shape({n, k});
        }
        weigthShape.MutableOriginShape() = gert::Shape({n, k});
    } else {
        if (bFormat == ge::FORMAT_FRACTAL_NZ) {
            weigthShape.MutableStorageShape() = gert::Shape({(n + 16) / 16, (k + 16) / 16L, 16L, 16});
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
    if (soc == 1) {
        aicoreSpec["cube_freq"] = "1500";
    }

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
                      .NodeInputTd(2, antiQuantScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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

    map<string, string> soc_version_infos;
    soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910_95"));
    if (soc == 1) {
        soc_version_infos.clear();
        soc_version_infos.insert(make_pair("Short_SoC_version", "Ascend910_55"));
    }
    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernelHold.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    ASSERT_EQ(tilingContext->GetTilingKey(), param.tilingKey);
    ASSERT_EQ(tilingContext->GetBlockDim(), param.blockDim);
}

TEST_P(TestWeightQuantBatchMatmulV2AdaptiveSplitTiling, generalTest)
{
    WeightQuantBatchMatmulV2TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

// format: m k n antiQuantOffsetExistFlag quantScaleExistFlag quantOffsetExistFlag biasFlag transA transB group Xdtype
//         weigthDtype antiQuantScaleDtype quantScaleDtype yDtype aicNum aivNum weightFormat(0:ND,1:NZ) socversion
// Note: group value
//       -1: per channel, 0: per tensor, > 0: per group
// Note: socversion
//        0: Ascend910_95,  1: Ascend910_55
static WeightQuantBatchMatmulV2TilingTestParam casesParams[] = {
    {"davidCase_2_1024_4096_0_0_0_0_0_0_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 16, 285421570UL},
    {"davidCase_2_1024_4096_1_0_0_1_0_0_0_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 16, 21760258050UL},
    {"davidCase_128_1024_4096_1_0_0_0_0_1_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 32, 4588843010UL},
    {"davidCase_260_1024_4096_0_0_0_0_0_1_-1_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 310583298UL},
    {"davidCase_16_1024_4096_0_0_0_0_0_1_-1_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 310452226UL},
    {"davidCase_16_1024_4096_0_0_0_1_0_0_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 16, 285421570UL},
    {"davidCase_124_64_64_0_0_0_0_0_1_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 1, 293875714UL},
    {"davidCase_12_64_64_0_0_0_0_0_1_0_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 1, 293875714UL},
    {"davidCase_1_640_65472_1_0_0_0_0_1_0_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 4588843010UL},
    {"davidCase_1_65472_512_1_0_0_1_0_1_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 4, 4588843010UL},
    {"davidCase_1_640_65540_1_0_0_0_0_1_0_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 4588843010UL},
    {"davidCase_1_65540_512_1_0_0_1_0_1_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 4, 4588843010UL},
    {"davidCase_16_1024_8192_0_0_0_0_0_1_-1_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 32, 310583298UL},
    {"davidCase_16_1024_5120_1_0_0_0_0_0_0_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 20, 4580388866UL},
    {"davidCase_16_1024_12288_0_0_0_1_0_0_0_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 17465290754UL},
    {"davidCase_16_1024_12288_1_0_0_1_0_1_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 32, 4588843010UL},
    {"davidCase_131_1024_12288_0_0_0_1_0_1_0_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_0_0", 32, 293875714UL},
    {"davidCase_270_8192_1536_0_0_0_1_0_1_-1_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 17490386946UL},
    {"davidCase_200_8192_1536_0_0_0_1_0_1_-1_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 30, 17490255874UL},
    {"davidCase_1_128_8384_0_0_0_1_0_1_-1_BF16_INT4_BF16_UINT64_BF16_32_64_0_0", 32, 17490452482UL},
    {"davidCase_16_1024_12288_0_0_0_1_0_0_-1_BF16_FP8E5M2_BF16_UINT64_BF16_32_64_0_0", 32, 17482067970UL},
    {"davidCase_12_64_64_0_0_0_0_0_1_-1_BF16_FP8E4M3_BF16_UINT64_BF16_32_64_0_0", 4, 310517762UL},
    {"davidCase_16_1024_17152_0_0_0_1_0_1_-1_FLOAT16_HIF8_FLOAT16_UINT64_FLOAT16_32_64_0_0", 32, 310583298UL},
    {"davidCase_2_1024_8192_0_0_0_0_0_0_-1_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_1_0", 32, 34661933058UL},
    {"davidCase_1_1024_8192_0_0_0_0_0_1_32_FLOAT16_FLOAT4E2M1_FLOAT8E8M0_UINT64_FLOAT16_32_64_0_0", 32,
     344072194UL},
    {"davidCase_1_1024_8192_0_0_0_0_0_0_32_FLOAT16_FLOAT4E1M2_FLOAT8E8M0_UINT64_FLOAT16_32_64_1_0", 32,
     34695487490UL},
    {"davidCase_1_10240_2560_0_0_0_0_0_0_32_FLOAT16_FLOAT4E1M2_FLOAT8E8M0_UINT64_FLOAT16_32_64_1_0", 32,
     34695356418UL},
    {"solomonCase_16_1024_8192_1_0_0_0_0_0_-1_BF16_INT8_BF16_UINT64_BF16_24_24_0_1", 24, 4597297157UL},
    {"solomonCase_2_1024_4096_0_0_0_0_0_1_-1_FLOAT16_INT8_FLOAT16_UINT64_FLOAT16_24_24_0_1", 24, 310390789UL},
    {"solomonCase_64_5120_5120_0_0_0_1_0_1_-1_BF16_INT8_BF16_UINT64_BF16_24_24_0_1", 24, 17490259973UL},
    {"solomonCase_128_800_128_1_0_0_1_0_1_-1_BF16_INT8_BF16_UINT64_BF16_24_24_0_1", 1, 21785227269UL},
    {"solomonCase_2_1024_4096_0_0_0_1_0_0_-1_FLOAT16_INT8_FLOAT16_UINT64_FLOAT16_24_24_0_1", 16, 302329861UL},
    {"solomonCase_16_256_512_1_0_0_1_0_0_-1_FLOAT16_INT8_FLOAT16_UINT64_FLOAT16_24_24_0_1", 2, 4597297157UL},
    // a16w4(int4) pergroup(32,64,128) nz
    {"davidCase_1_8192_2560_0_0_0_1_0_0_128_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_1_0", 32, 34678579202UL},
    {"davidCase_1_8192_2560_1_0_0_1_0_0_128_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_1_0", 32, 38973546498UL},
    {"davidCase_1983_1280_1088_0_0_0_1_0_0_128_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_1_0", 32,
     34678579202UL},
    {"davidCase_1983_1280_1088_1_0_0_1_0_0_128_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_1_0", 32,
     38973546498UL},
    {"davidCase_1_64_64_0_0_0_1_0_0_32_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_4_8_1_0", 4, 34678579202UL},
    {"davidCase_8_8192_512_1_0_0_0_0_0_128_FLOAT16_INT4_FLOAT16_UINT64_FLOAT16_32_64_1_0", 32, 38973546498UL},
    {"davidCase_1_8192_2560_0_0_0_1_0_0_128_BF16_INT4_BF16_UINT64_BF16_32_64_1_0", 32, 51858448386UL},
    {"davidCase_1_8192_2560_1_0_0_1_0_0_128_BF16_INT4_BF16_UINT64_BF16_32_64_1_0", 32, 56153415682UL},
    {"davidCase_1983_1280_1088_0_0_0_1_0_0_128_BF16_INT4_BF16_UINT64_BF16_32_64_1_0", 32, 51858448386UL},
    {"davidCase_1983_1280_1088_1_0_0_1_0_0_128_BF16_INT4_BF16_UINT64_BF16_32_64_1_0", 32, 56153415682UL},
    {"davidCase_1_64_64_0_0_0_1_0_0_32_BF16_INT4_BF16_UINT64_BF16_4_8_1_0", 4, 51858448386UL},
    {"davidCase_8_8192_512_1_0_0_1_0_0_128_BF16_INT4_BF16_UINT64_BF16_32_64_1_0", 32, 56153415682UL},
    // a16F8 perchannel
    {"davidCase_8_8192_512_0_0_0_0_0_1_-1_BF16_FP8E4M3_BF16_UINT64_BF16_32_64_0_0", 32, 310517762UL},
    // a16F8 perchannel support n=1(reuse perchannel)
    {"davidCase_8_8192_512_0_0_0_0_0_1_0_BF16_FP8E4M3_BF16_UINT64_BF16_32_64_0_0", 32, 310517762UL},
};

INSTANTIATE_TEST_CASE_P(MM, TestWeightQuantBatchMatmulV2AdaptiveSplitTiling, testing::ValuesIn(casesParams));
