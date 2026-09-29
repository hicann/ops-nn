/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_max_pool_grad_with_argmax_base_extra_tiling.cpp
 * \brief MaxPoolGradWithArgmax tiling base(max_pool_grad_with_argmax_tiling_base.cpp /
 *        pool_grad_common/max_pool_grad_with_argmax_tiling_common.cpp)补充用例:
 *        1. 直接实例化 MaxPoolGradWithArgmaxBaseTiling 执行完整 DoTiling 流程, 覆盖 PostTiling
 *           及公共 tiling 基类 MaxPoolGradWithArgmaxTilingCommon 的 IsCapable/DoOpTiling/GetTilingKey;
 *        2. 直接实例化 MaxPoolGradWithArgmaxTilingCommon 执行完整 DoTiling 流程, 覆盖其
 *           GetShapeAttrsInfo/PostTiling 默认实现;
 *        3. GetShapeAttrsInfo 的各类参数校验错误分支(shape/dtype/format/ksize/strides/padding/pad/grad shape)。
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/max_pool_grad_with_argmax_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
// 执行模式: 0-仅调用 BaseTiling::GetShapeAttrsInfo; 1-BaseTiling 完整 DoTiling; 2-TilingCommon 完整 DoTiling
constexpr int32_t RUN_MODE_SHAPE_ATTRS_ONLY = 0;
constexpr int32_t RUN_MODE_BASE_DO_TILING = 1;
constexpr int32_t RUN_MODE_COMMON_DO_TILING = 2;
} // namespace

class MaxPoolGradWithArgmaxBaseExtraTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolGradWithArgmaxBaseExtraTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolGradWithArgmaxBaseExtraTiling TearDown" << std::endl; }
};

static void ExecuteBaseExtraTestCase(gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape gradShape,
                                     gert::StorageShape argmaxShape, std::vector<int64_t> ksize,
                                     std::vector<int64_t> strides, std::string padding, ge::DataType dtype,
                                     ge::DataType dtypeIdx, bool include_batch_in_index, std::string data_format,
                                     int32_t runMode, ge::graphStatus expectStatus, uint64_t except_tilingkey)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolGradWithArgmaxCompileInfo compile_info;

    std::string op_type("MaxPoolGradWithArgmax");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs(std::vector<void*>{&compile_info})
                             .Build();

    ASSERT_TRUE((kernel_holder.GetContext<gert::TilingParseContext>())->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func((kernel_holder.GetContext<gert::KernelContext>())), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtypeIdx, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                           {"include_batch_in_index", Ops::NN::AnyValue::CreateFrom<bool>(include_batch_in_index)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    if (runMode == RUN_MODE_SHAPE_ATTRS_ONLY) {
        optiling::MaxPoolGradWithArgmaxBaseTiling base(tiling_context);
        EXPECT_EQ(base.GetShapeAttrsInfo(), expectStatus);
    } else if (runMode == RUN_MODE_BASE_DO_TILING) {
        optiling::MaxPoolGradWithArgmaxBaseTiling base(tiling_context);
        EXPECT_EQ(base.DoTiling(), expectStatus);
        if (expectStatus == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(tiling_context->GetTilingKey(), except_tilingkey);
        }
    } else {
        optiling::MaxPoolGradWithArgmaxTilingCommon common(tiling_context);
        EXPECT_EQ(common.DoTiling(), expectStatus);
        if (expectStatus == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(tiling_context->GetTilingKey(), except_tilingkey);
        }
    }
}

// ==================== 完整 DoTiling 流程(PostTiling 与公共基类默认实现) ====================

// BaseTiling 完整流程: NCHW 合法输入, 走公共基类 IsCapable/DoOpTiling/GetTilingKey 默认实现与
// BaseTiling::PostTiling, tilingKey=0
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, MaxPoolGradWithArgmaxBaseExtra_DoTilingNchwSuccess)
{
    gert::StorageShape xShape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 3, 3}, {1, 1, 3, 3}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_BASE_DO_TILING, ge::GRAPH_SUCCESS, 0);
}

// TilingCommon 完整流程: 覆盖 MaxPoolGradWithArgmaxTilingCommon::GetShapeAttrsInfo/PostTiling 默认实现,
// tilingKey=0
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, MaxPoolGradWithArgmaxBaseExtra_TilingCommonDoTilingSuccess)
{
    gert::StorageShape xShape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 3, 3}, {1, 1, 3, 3}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_COMMON_DO_TILING, ge::GRAPH_SUCCESS, 0);
}

// ==================== GetShapeAttrsInfo 错误分支 ====================

// xShape 维度不等于4 (line 51-54)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_x_shapedim)
{
    gert::StorageShape xShape = {{2, 3, 64}, {2, 3, 64}};
    gert::StorageShape yShape = {{2, 3, 64}, {2, 3, 64}};
    gert::StorageShape gradShape = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1}, {2, 3, 1}};
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// xShape size <= 0 (line 56-60)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_x_shapesize_zero)
{
    gert::StorageShape xShape = {{2, 0, 64, 64}, {2, 0, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = gradShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// x dtype 非法, 不属于 float16/float/bfloat16 (line 65-70)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_x_dtype)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = gradShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_INT32;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// grad shape size <= 0 (line 75-79)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_grad_shapesize_zero)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 0, 1, 1}, {2, 0, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// argmax shape size <= 0 (line 84-88)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_argmax_shapesize_zero)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 0, 1, 1}, {2, 0, 1, 1}};
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// argmax dtype 非法, 不属于 int32/int64 (line 93-97)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_argmax_dtype)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = gradShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_FLOAT;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// grad shape 与 argmax shape 不一致 (line 100-104)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_grad_argmax_shape_mismatch)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 2, 2}, {2, 3, 2, 2}};
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// 输出 y shape 与输入 x shape 不一致 (line 109-113)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_y_x_shape_mismatch)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 32, 32}, {2, 3, 32, 32}};
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = gradShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// data_format 既非 NCHW 也非 NHWC (line 139-142)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_data_format)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape gradShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = gradShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCDHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// NHWC ksize 非法: ksize[0] != 1 (line 148-159)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_nhwc_ksize)
{
    gert::StorageShape xShape = {{1, 8, 8, 1}, {1, 8, 8, 1}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 3, 3, 1}, {1, 3, 3, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 4, 5, 1};
    std::vector<int64_t> strides = {1, 2, 2, 1};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// NHWC strides 非法: strides[0] != 1 (line 165-175)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_nhwc_strides)
{
    gert::StorageShape xShape = {{1, 8, 8, 1}, {1, 8, 8, 1}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 3, 3, 1}, {1, 3, 3, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 3, 3, 1};
    std::vector<int64_t> strides = {2, 2, 2, 1};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NHWC";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// NCHW ksize 非法: ksize[0] != 1 (line 181-192)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_nchw_ksize)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2, 4, 4};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// NCHW strides 非法: strides[0] != 1 (line 197-207)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_nchw_strides)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {2, 2, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// padding 模式非法, 既非 VALID 也非 SAME (line 216-219)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_padding_mode)
{
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "FULL";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// SAME 模式下 grad shape 与 stride 不匹配导致推导 pad 超过 kernel/2 (line 232-239)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_invalid_pad_exceed_kernel)
{
    gert::StorageShape xShape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 1, 2, 2};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "SAME";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}

// VALID 模式下 grad shape 与 ksize/stride 推导结果不一致 (line 241-245)
TEST_F(MaxPoolGradWithArgmaxBaseExtraTiling, tiling_grad_shape_mismatch_expected)
{
    gert::StorageShape xShape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {1, 1, 3, 3};
    std::vector<int64_t> strides = {1, 1, 2, 2};
    std::string padding = "VALID";
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtypeIdx = ge::DT_INT32;
    bool include_batch_in_index = false;
    std::string data_format = "NCHW";
    ExecuteBaseExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, padding, dtype, dtypeIdx,
                             include_batch_in_index, data_format, RUN_MODE_SHAPE_ATTRS_ONLY, ge::GRAPH_FAILED, 0);
}
