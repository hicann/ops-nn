/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/apply_adam_w_quant_tiling_def.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "tests/ut/common/any_value.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ApplyAdamWQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdamWQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyAdamWQuantTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize();) {
        if (i == 6 * sizeof(int64_t) || i == 6 * sizeof(int64_t) + sizeof(float) ||
            i == 6 * sizeof(int64_t) + 2 * sizeof(float) || i == 6 * sizeof(int64_t) + 3 * sizeof(float) ||
            i == 6 * sizeof(int64_t) + 4 * sizeof(float) || i == 6 * sizeof(int64_t) + 5 * sizeof(float)) {
            result += std::to_string((reinterpret_cast<const float*>(tiling_data->GetData())[i / sizeof(float)]));
            i += sizeof(float);
        } else {
            if (i == 7 * sizeof(int64_t) + 6 * sizeof(float)) {
                i += sizeof(float);
            }
            result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
            i += sizeof(int64_t);
        }
        result += " ";
    }

    return result;
}

void TilingTest(
    std::string opName,
    std::initializer_list<int64_t>& varShape,        // 输入var的shape
    std::initializer_list<int64_t>& gradShape,       // 输入grad的shape
    std::initializer_list<int64_t>& mRefShape,       // 输入mRefShape的shape
    std::initializer_list<int64_t>& vRefShape,       // 输入vRefShape的shape
    std::initializer_list<int64_t>& qmapMShape,      // 输入qmapMShape的shape
    std::initializer_list<int64_t>& qmapVShape,      // 输入qmapVShape的shape
    std::initializer_list<int64_t>& absmaxMRefShape, // 输入absmaxMRefShape的shape
    std::initializer_list<int64_t>& absmaxVRefShape, // 输入absmaxVRefShape的shape
    std::initializer_list<int64_t>& stepShape,       // 输入stepShape的shape
    float lr_in,                                     // attr的输入
    float beta1_in,                                  // attr的输入
    float beta2_in,                                  // attr的输入
    float weight_decay_in,                           // attr的输入
    float eps_in,                                    // attr的输入
    float gnorm_scale_in,                            // attr的输入
    int64_t block_size_in,                           // attr的输入
    string quantModeOptiona_in,
    const int32_t inputNum,       // 输入个数
    const int32_t outputNum,      // 输出个数
    ge::DataType datatype1,       // 输入var和grad的dtype
    ge::DataType datatype2,       // 输入mRef和vRef的dtype
    ge::DataType datatype3,       // 输入其余所有的dtype，float32
    ge::DataType datatype4,       // 输入step的dtype，int64
    ge::Format format,            // 所有的format都一样
    const ge::graphStatus status, // 成功的结果
    uint64_t tilingKeyValue       // Tiling Key 值
)
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
    optiling::Tiling4ApplyAdamWQuantCompileInfo compile_info;

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

    gert::StorageShape input1Shape = {varShape, varShape};
    gert::StorageShape input2Shape = {gradShape, gradShape};
    gert::StorageShape input3Shape = {mRefShape, mRefShape};
    gert::StorageShape input4Shape = {vRefShape, vRefShape};
    gert::StorageShape input5Shape = {qmapMShape, qmapMShape};
    gert::StorageShape input6Shape = {qmapVShape, qmapVShape};
    gert::StorageShape input7Shape = {absmaxMRefShape, absmaxMRefShape};
    gert::StorageShape input8Shape = {absmaxVRefShape, absmaxVRefShape};
    gert::StorageShape input9Shape = {stepShape, stepShape};
    gert::StorageShape output1Shape = {varShape, varShape};
    gert::StorageShape output2Shape = {mRefShape, mRefShape};
    gert::StorageShape output3Shape = {vRefShape, vRefShape};
    gert::StorageShape output4Shape = {absmaxMRefShape, absmaxMRefShape};
    gert::StorageShape output5Shape = {absmaxVRefShape, absmaxVRefShape};

    std::vector<uint32_t> irInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1});
    ge::DataType varGradDtype = datatype1;
    ge::DataType mAndVDtype = datatype2;
    ge::DataType othersDtype = datatype3;
    ge::DataType stepDtype = datatype4;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(inputNum, outputNum)
                      .IrInstanceNum(irInstanceNum)
                      .InputShapes({
                          &input1Shape,
                          &input2Shape,
                          &input3Shape,
                          &input4Shape,
                          &input5Shape,
                          &input6Shape,
                          &input7Shape,
                          &input8Shape,
                          &input9Shape,
                      })
                      .OutputShapes({
                          &output1Shape,
                          &output2Shape,
                          &output3Shape,
                          &output4Shape,
                          &output5Shape,
                      })
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varGradDtype, format, format)
                      .NodeInputTd(1, varGradDtype, format, format)
                      .NodeInputTd(2, mAndVDtype, format, format)
                      .NodeInputTd(3, mAndVDtype, format, format)
                      .NodeInputTd(4, othersDtype, format, format)
                      .NodeInputTd(5, othersDtype, format, format)
                      .NodeInputTd(6, othersDtype, format, format)
                      .NodeInputTd(7, othersDtype, format, format)
                      .NodeInputTd(8, stepDtype, format, format)
                      .NodeOutputTd(0, varGradDtype, format, format)
                      .NodeOutputTd(1, mAndVDtype, format, format)
                      .NodeOutputTd(2, mAndVDtype, format, format)
                      .NodeOutputTd(3, othersDtype, format, format)
                      .NodeOutputTd(4, othersDtype, format, format)
                      .TilingData(param.get())
                      .NodeAttrs(
                          {{"lr", Ops::NN::AnyValue::CreateFrom<float>(lr_in)},
                           {"beta1", Ops::NN::AnyValue::CreateFrom<float>(beta1_in)},
                           {"beta2", Ops::NN::AnyValue::CreateFrom<float>(beta2_in)},
                           {"weight_decay", Ops::NN::AnyValue::CreateFrom<float>(weight_decay_in)},
                           {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps_in)},
                           {"gnorm_scale", Ops::NN::AnyValue::CreateFrom<float>(gnorm_scale_in)},
                           {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>(quantModeOptiona_in)},
                           {"block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(block_size_in)}})
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_SUCCESS) {
        auto tiling_key = tiling_context->GetTilingKey();
        ASSERT_EQ(tiling_key, tilingKeyValue);
    }
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    std::cout << tiling_data_result << std::endl;
}

TEST_F(ApplyAdamWQuantTiling, ApplyAdamWQuantTiling_test_case_tilingkey_100)
{
    std::initializer_list<int64_t> var_shape = {96, 256};
    std::initializer_list<int64_t> grad_shape = {96, 256};
    std::initializer_list<int64_t> m_ref_shape = {96, 256};
    std::initializer_list<int64_t> v_ref_shape = {96, 256};
    std::initializer_list<int64_t> qmapm_shape = {256};
    std::initializer_list<int64_t> qmapv_shape = {256};
    std::initializer_list<int64_t> absmaxm_shape = {96};
    std::initializer_list<int64_t> absmaxv_shape = {96};
    std::initializer_list<int64_t> step_shape = {1};

    float lr = 0.1;
    float beta1 = 0.9;
    float beta2 = 0.9;
    float weight_decay = 0.9;
    float eps = 0.0001;
    float gnorm_scale = 0.9;
    int64_t block_size = 256;
    std::string mode = "BLOCKWISE";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    uint64_t tilingKeyValue = 100;

    TilingTest(
        "ApplyAdamWQuant", var_shape, grad_shape, m_ref_shape, v_ref_shape, qmapm_shape, qmapv_shape, absmaxm_shape,
        absmaxv_shape, step_shape, lr, beta1, beta2, weight_decay, eps, gnorm_scale, block_size, mode, 9, 5,
        ge::DT_FLOAT, ge::DT_UINT8, ge::DT_FLOAT, ge::DT_INT64, ge::FORMAT_ND, status, tilingKeyValue);
}

TEST_F(ApplyAdamWQuantTiling, ApplyAdamWQuantTiling_test_case_tilingkey_200)
{
    std::initializer_list<int64_t> var_shape = {96, 256};
    std::initializer_list<int64_t> grad_shape = {96, 256};
    std::initializer_list<int64_t> m_ref_shape = {96, 256};
    std::initializer_list<int64_t> v_ref_shape = {96, 256};
    std::initializer_list<int64_t> qmapm_shape = {256};
    std::initializer_list<int64_t> qmapv_shape = {256};
    std::initializer_list<int64_t> absmaxm_shape = {96};
    std::initializer_list<int64_t> absmaxv_shape = {96};
    std::initializer_list<int64_t> step_shape = {1};

    float lr = 0.1;
    float beta1 = 0.9;
    float beta2 = 0.9;
    float weight_decay = 0.9;
    float eps = 0.0001;
    float gnorm_scale = 0.9;
    int64_t block_size = 256;
    std::string mode = "BLOCKWISE";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    uint64_t tilingKeyValue = 200;

    TilingTest(
        "ApplyAdamWQuant", var_shape, grad_shape, m_ref_shape, v_ref_shape, qmapm_shape, qmapv_shape, absmaxm_shape,
        absmaxv_shape, step_shape, lr, beta1, beta2, weight_decay, eps, gnorm_scale, block_size, mode, 9, 5,
        ge::DT_FLOAT16, ge::DT_UINT8, ge::DT_FLOAT, ge::DT_INT64, ge::FORMAT_ND, status, tilingKeyValue);
}

TEST_F(ApplyAdamWQuantTiling, ApplyAdamWQuantTiling_test_case_tilingkey_300)
{
    std::initializer_list<int64_t> var_shape = {96, 256};
    std::initializer_list<int64_t> grad_shape = {96, 256};
    std::initializer_list<int64_t> m_ref_shape = {96, 256};
    std::initializer_list<int64_t> v_ref_shape = {96, 256};
    std::initializer_list<int64_t> qmapm_shape = {256};
    std::initializer_list<int64_t> qmapv_shape = {256};
    std::initializer_list<int64_t> absmaxm_shape = {96};
    std::initializer_list<int64_t> absmaxv_shape = {96};
    std::initializer_list<int64_t> step_shape = {1};

    float lr = 0.1;
    float beta1 = 0.9;
    float beta2 = 0.9;
    float weight_decay = 0.9;
    float eps = 0.0001;
    float gnorm_scale = 0.9;
    int64_t block_size = 256;
    std::string mode = "BLOCKWISE";
    const ge::graphStatus status = ge::GRAPH_SUCCESS;
    uint64_t tilingKeyValue = 300;

    TilingTest(
        "ApplyAdamWQuant", var_shape, grad_shape, m_ref_shape, v_ref_shape, qmapm_shape, qmapv_shape, absmaxm_shape,
        absmaxv_shape, step_shape, lr, beta1, beta2, weight_decay, eps, gnorm_scale, block_size, mode, 9, 5,
        ge::DT_BF16, ge::DT_UINT8, ge::DT_FLOAT, ge::DT_INT64, ge::FORMAT_ND, status, tilingKeyValue);
}