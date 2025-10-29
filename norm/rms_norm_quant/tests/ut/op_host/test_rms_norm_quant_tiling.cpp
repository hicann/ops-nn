/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_norm_quant_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class RmsNormAtbTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNormAtbTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNormAtbTiling TearDown" << std::endl;
    }
};

struct NormCommonCompileInfo {
    uint32_t coreNumAiv = 0;
    uint32_t ubSize = 0;
    uint32_t sysWorkspaceSize = 0;
};

template <typename T, bool EN_QUANT, bool EN_PRE_POST>
void TilingTest(
    std::string opName, std::initializer_list<int64_t>& xShape, std::initializer_list<int64_t>& gammaShape,
    std::initializer_list<int64_t>& betaShape, std::initializer_list<int64_t>& resultShape, float epsilon,
    bool gemma_mode, bool high_precision_mode, ge::DataType datatype, ge::Format format, const ge::graphStatus status,
    uint64_t tilingKeyValue)
{
    std::string op_type(opName);
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 编译信息
    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                        "Intrinsic_fix_pipe_l0c2out": false,
                                                        "Intrinsic_data_move_l12ub": true,
                                                        "Intrinsic_data_move_l0c2ub": true,
                                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                                        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                                                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                                                        "CORE_NUM": 48}
                                    })";

    // 平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // 编译信息
    NormCommonCompileInfo compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::StorageShape x = {xShape, xShape};
    gert::StorageShape gamma = {gammaShape, gammaShape};
    gert::StorageShape beta = {betaShape, betaShape};
    gert::StorageShape scale_and_offset_shape = {{1}, {1}};

    gert::StorageShape result = {resultShape, resultShape};

    int32_t inputNum = 3;
    int32_t outputNum = 1;

    if (EN_PRE_POST) {
        inputNum += 1;
        outputNum += 1;
    }

    if (EN_QUANT) {
        inputNum += 2;
    }

    std::vector<uint32_t> irInstanceNum(inputNum, 1);

    ge::DataType resdatatype = datatype;
    if (EN_QUANT) {
        resdatatype = ge::DT_INT8;
    }

    auto holderBuilder = gert::TilingContextFaker()
                             .NodeIoNum(inputNum, outputNum)
                             .IrInstanceNum(irInstanceNum)
                             .InputShapes({&x, &gamma, &beta, &scale_and_offset_shape, &scale_and_offset_shape})
                             .OutputShapes({&result})
                             .CompileInfo(&compile_info)
                             .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                             .NodeInputTd(0, datatype, format, format)
                             .NodeInputTd(1, datatype, format, format)
                             .NodeInputTd(2, datatype, format, format)
                             .NodeInputTd(3, datatype, format, format)
                             .NodeInputTd(4, ge::DT_INT8, format, format)
                             .NodeOutputTd(0, resdatatype, format, format)
                             .TilingData(param.get())
                             .NodeAttrs({
                                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(epsilon)},
                                 {"high_precision_mode", Ops::NN::AnyValue::CreateFrom<bool>(high_precision_mode)},
                                 {"gemma_mode", Ops::NN::AnyValue::CreateFrom<bool>(gemma_mode)},
                                 // {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                 // {"offset", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                             })
                             .Workspace(ws_size)
                             .Build();

    gert::TilingContext* tiling_context = holderBuilder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holderBuilder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holderBuilder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holderBuilder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holderBuilder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_SUCCESS) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKeyValue);
    }
}

// RmsNormQuant
TEST_F(RmsNormAtbTiling, RmsNormQuantTilingData_test_float16_success_tilingKey_0b1_1_1_0_000001)
{
    // half & FastCompute
    std::initializer_list<int64_t> xShape = {32, 32, 512};
    std::initializer_list<int64_t> gamma = {1, 512};
    std::initializer_list<int64_t> beta = {1, 512};
    std::initializer_list<int64_t> result = {32, 32, 512};

    float epsilon = 0.01;
    const ge::graphStatus status = ge::GRAPH_SUCCESS;

    // PreRmsNorm  RmsNormQuant  PreRmsNormQuant
    TilingTest<int64_t, true, false>(
        "RmsNormQuant", xShape, gamma, beta, result, epsilon, true, true, ge::DT_FLOAT16, ge::FORMAT_ND, status,
        0b1110000001);
}

TEST_F(RmsNormAtbTiling, RmsNormQuantTilingData_test_bfloat16_success_tilingKey_0b1_1_1_0_011011)
{
    // bhalf & FastCompute
    std::initializer_list<int64_t> xShape = {32, 32, 512};
    std::initializer_list<int64_t> gamma = {1, 512};
    std::initializer_list<int64_t> beta = {1, 512};
    std::initializer_list<int64_t> result = {32, 32, 512};
    float epsilon = 0.01;
    const ge::graphStatus status = ge::GRAPH_SUCCESS;

    TilingTest<int64_t, true, false>(
        "RmsNormQuant", xShape, gamma, beta, result, epsilon, true, true, ge::DT_BF16, ge::FORMAT_ND, status,
        0b1110011011);
}
