/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_div_loss_grad_dag_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/elewise/elewise_tiling.h"
#include "../../../../op_host/arch35/kl_div_loss_grad_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class KlDivLossGradDagTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "KlDivLossGradDagTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "KlDivLossGradDagTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                        map<string, string>& aicoreSpec, map<string, string>& intrinsics, map<string, string>& socVersion)
{
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    // platform info
    platFormInfo.Init();
}

static void DoKlDivLossGradDagTilingCase(std::initializer_list<int64_t>& inputShape1, std::initializer_list<int64_t>& inputShape2,
                                         std::initializer_list<int64_t>& inputShape3, std::initializer_list<int64_t>& outputShape, 
                                         ge::DataType inputDtype, std::string& reduction, bool& logTarget, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    optiling::KlDivLossGradCompileInfo compileInfo;
    std::string opType("KlDivLossGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    string compileInfoStr = R"({})";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

     // tilingFunc simulate
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    gert::StorageShape gradShape = {inputShape1, inputShape1};
    gert::StorageShape input_Shape = {inputShape2, inputShape2};
    gert::StorageShape targetShape = {inputShape3, inputShape3};
    gert::StorageShape youtShape = {outputShape, outputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradShape, &input_Shape, &targetShape})
                      .OutputShapes({&youtShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)},
                                  {"log_target", Ops::NN::AnyValue::CreateFrom<bool>(logTarget)}})
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling1)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048, 1, 48};
    std::initializer_list<int64_t> inputShape2 = {2048, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "mean";
    bool logTarget = false;
    std::string expectStr =
        "8589934592 687194767360 128 13 1 1 13 7680 13 2048 48 0 0 0 0 0 0 48 1 0 0 0 0 0 0 0 0 160 128 0 0 1 48 0 0 0 0 0 0 2048 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 48 1 0 0 0 0 0 0 973078528 0 ";
     DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling2)
{
    // HALF
    std::initializer_list<int64_t> inputShape1 = {2048, 1, 48};
    std::initializer_list<int64_t> inputShape2 = {2048, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "sum";
    bool logTarget = false;
    std::string expectStr =
        "8589934592 687194767360 128 13 1 1 13 7680 13 2048 48 0 0 0 0 0 0 48 1 0 0 0 0 0 0 0 0 160 128 0 0 1 48 0 0 0 0 0 0 2048 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 48 1 0 0 0 0 0 0 1065353216 0 ";
    DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling3)
{
    // BF16
    std::initializer_list<int64_t> inputShape1 = {2048, 1, 48};
    std::initializer_list<int64_t> inputShape2 = {2048, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "batchmean";
    bool logTarget = false;
    std::string expectStr =
        "8589934592 687194767360 128 13 1 1 13 7680 13 2048 48 0 0 0 0 0 0 48 1 0 0 0 0 0 0 0 0 160 128 0 0 1 48 0 0 0 0 0 0 2048 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 48 1 0 0 0 0 0 0 973078528 0 ";
    DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling4)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048, 1, 48};
    std::initializer_list<int64_t> inputShape2 = {2048, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {1, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "batchmean";
    bool logTarget = true;
    std::string expectStr =
        "8589934592 687194767360 128 13 1 1 13 7680 13 2048 48 0 0 0 0 0 0 48 1 0 0 0 0 0 0 0 0 160 128 0 0 1 48 0 0 0 0 0 0 2048 48 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 48 1 0 0 0 0 0 0 973078528 0 ";
    DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling5)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1024, 1, 36};
    std::initializer_list<int64_t> inputShape2 = {1, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {};
    std::initializer_list<int64_t> outputShape = {1024, 1, 36};
    std::string reduction = "sum";
    bool logTarget = true;
    std::string expectStr = "6 4947802326144 32 1 1 1065353216 0 ";
    DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling6)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1024, 1, 36};
    std::initializer_list<int64_t> inputShape2 = {1, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {};
    std::initializer_list<int64_t> outputShape = {1024, 1, 36};
    std::string reduction = "mean";
    bool logTarget = true;
    std::string expectStr = "6 4947802326144 32 1 1 1065353216 0 ";
    DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}

TEST_F(KlDivLossGradDagTiling, kl_div_loss_grad_david_tiling7)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1024, 1, 36};
    std::initializer_list<int64_t> inputShape2 = {1, 1, 1};
    std::initializer_list<int64_t> inputShape3 = {};
    std::initializer_list<int64_t> outputShape = {1024, 1, 36};
    std::string reduction = "batchmean";
    bool logTarget = true;
    std::string expectStr = "6 4947802326144 32 1 1 1065353216 0 ";
    DoKlDivLossGradDagTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction, logTarget, expectStr);
}
