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
#include <nlohmann/json.hpp>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

struct TransQuantParamV2TilingTestParam {
    string caseName;

    // output
    uint32_t scaleLength;
    uint32_t offsetLength;
};

class TestTransQuantParamV2Tiling : public testing::TestWithParam<TransQuantParamV2TilingTestParam> {};

using namespace ge;
using namespace ut_util;

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

struct TransQuantParamV2CompileInfo {};

static TransQuantParamV2TilingTestParam casesParams[] = {
    {"test_2_2", 2, 2},       {"test_1_1", 1, 1},       {"test_8193_8193", 8193, 8193},
    {"test_8193_1", 8193, 1}, {"test_1_8193", 1, 8193},
};

static void TestOneParamCase(const TransQuantParamV2TilingTestParam& param)
{
    std::cout << "run case " << param.caseName << std::endl;
    std::vector<string> testParam;
    SplitStr2Vec(param.caseName.substr(param.caseName.find_first_of('_') + 1), "_", testParam);

    size_t idx = 0;
    int64_t shape = stol(testParam[idx++]);
    int64_t hasOffset = stol(testParam[idx++]);

    ge::DataType scaleDtype = ge::DT_FLOAT;
    ge::DataType offsetDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_UINT64;

    gert::StorageShape scaleShape;
    gert::StorageShape offsetShape;
    gert::StorageShape outputShape;

    scaleShape.MutableStorageShape() = gert::Shape({shape});
    offsetShape.MutableStorageShape() = gert::Shape({shape});
    outputShape.MutableStorageShape() = gert::Shape({shape});
    scaleShape.MutableOriginShape() = scaleShape.MutableStorageShape();
    offsetShape.MutableOriginShape() = offsetShape.MutableStorageShape();
    outputShape.MutableOriginShape() = outputShape.MutableStorageShape();

    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    string compileInfoStr = R"({
        "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0",
                          "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196352, "L2_SIZE": 33554432, "L1_SIZE": 524032,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 24,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
    GetPlatFormInfos(compileInfoStr.c_str(), socInfos, aicoreSpec, intrinsics);
    aicoreSpec["cube_freq"] = "1800";

    // platform info
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    TransQuantParamV2CompileInfo compileInfo;

    std::string opType("TransQuantParamV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto rawTilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(rawTilingData, nullptr);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&scaleShape, &offsetShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, offsetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
            .Outputs({&compileInfoStr})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    ASSERT_EQ(tilingParseFunc(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    ASSERT_NE(tilingFunc, nullptr);
    ASSERT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

TEST_P(TestTransQuantParamV2Tiling, generalTest)
{
    TransQuantParamV2TilingTestParam param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_CASE_P(TQPV2, TestTransQuantParamV2Tiling, testing::ValuesIn(casesParams));

static void ThreadFunc(
    const TransQuantParamV2TilingTestParam* params, size_t testcase_num, size_t thread_idx, size_t thread_num)
{
    for (size_t idx = thread_idx; idx < testcase_num; idx += thread_num) {
        TestOneParamCase(params[idx]);
    }
}

static void TestMultiThread(const TransQuantParamV2TilingTestParam* params, size_t testcase_num, size_t thread_num)
{
    std::thread threads[thread_num];
    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx] = std::thread(ThreadFunc, params, testcase_num, idx, thread_num);
    }

    for (size_t idx = 0; idx < thread_num; ++idx) {
        threads[idx].join();
    }
}

TEST_F(TestTransQuantParamV2Tiling, multi_thread)
{
    TestMultiThread(casesParams, sizeof(casesParams) / sizeof(TransQuantParamV2TilingTestParam), 3);
}