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
 * \file test_max_pool_grad_with_argmax_v3_nchw_extra_tiling.cpp
 * \brief MaxPoolGradWithArgmaxV3 NCHW tiling 模板(max_pool_grad_with_argmax_v3_nchw_tiling /
 *        max_pool_grad_nchw_tiling_common / max_pool_grad_with_argmax_v3_nchw_tiling_scalar)补充用例:
 *        1. NCHW 正常场景: 覆盖 SearchBestTiling 的 TrySplitNC(整切/二分)、TrySplitAlignH、TrySplitAlignW、
 *           SplitUnalignHW(对齐/非对齐起点)分支, 以及 DoBlockTiling/SetTilingData/PostTiling;
 *        2. int64 index 场景: 覆盖 GetTilingKey 的 CHECK_RANGE(101/111)与 T3_INT64(+10)组合;
 *        3. NCHW scalar 模板: 覆盖 CalcBase 的 perH/perW/hw 切分分支与
 *           CalcGradArgmaxInner/InnerTail 的四类 inner 切分分支及早退分支。
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
#include "../../../../op_host/arch35/max_pool_grad_with_argmax_v3_nchw_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MaxPoolGradWithArgmaxV3NchwExtraTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MaxPoolGradWithArgmaxV3NchwExtraTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MaxPoolGradWithArgmaxV3NchwExtraTiling TearDown" << std::endl; }
};

static void ExecuteNchwExtraTestCase(gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape gradShape,
                                     gert::StorageShape argmaxShape, std::vector<int64_t> ksize,
                                     std::vector<int64_t> strides, std::vector<int64_t> pads,
                                     std::vector<int64_t> dilation, ge::DataType dtype, int64_t index_dtype,
                                     ge::DataType index_dtype_enum, bool ceil_mode, std::string data_format,
                                     uint64_t except_tilingkey)
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
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolGradWithArgmaxCompileInfo compile_info;

    std::string op_type("MaxPoolGradWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

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
                      .NodeInputTd(2, index_dtype_enum, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(index_dtype)},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceil_mode)},
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

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, except_tilingkey);
}

// ==================== NCHW tiling 模板(SearchBestTiling 各分支) ====================

// TrySplitNC 整切直接成功: nc=64, 单个 NC 块 + 完整 H/W 平面满足 UB 且达到目标核数, tilingKey=100
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_TrySplitNcDirect)
{
    gert::StorageShape xShape = {{16, 4, 56, 56}, {16, 4, 56, 56}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{16, 4, 55, 55}, {16, 4, 55, 55}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// TrySplitNC 二分搜索分支: nc=128, 整切(2个NC/块)超UB失败, 降为1后满足, 走 SearchMaxSplit, tilingKey=100
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_TrySplitNcSearch)
{
    gert::StorageShape xShape = {{8, 16, 80, 80}, {8, 16, 80, 80}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{8, 16, 79, 79}, {8, 16, 79, 79}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// TrySplitAlignH 成功: 无pad无overlap, NC整切不满足核数, 按 hStride 对齐切 H 并二分, tilingKey=100
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_TrySplitAlignH)
{
    gert::StorageShape xShape = {{2, 2, 64, 64}, {2, 2, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{2, 2, 32, 32}, {2, 2, 32, 32}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// TrySplitAlignW 成功: 宽方向超长(wX=256), H对齐切与NC整切均不满足, 按 wStride 对齐切 W 并二分, tilingKey=100
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_TrySplitAlignW)
{
    gert::StorageShape xShape = {{1, 1, 4, 256}, {1, 1, 4, 256}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 2, 128}, {1, 1, 2, 128}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 100;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// SplitUnalignHW overlap 起点: kernel>stride 产生 overlap, 跳过对齐切分, 循环切 H 至 1 后退出兜底分支,
// isCheckRange=1, tilingKey=101
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_SplitUnalignHWOverlap)
{
    gert::StorageShape xShape = {{1, 1, 56, 56}, {1, 1, 56, 56}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 55, 55}, {1, 1, 55, 55}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 101;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// SplitUnalignHW 对齐起点: 无pad无overlap 但核数不足, 对齐切 H/W 均失败, 兜底切分循环内命中返回,
// isCheckRange=1, tilingKey=101
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_SplitUnalignHWAligned)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {16, 16};
    std::vector<int64_t> strides = {16, 16};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 101;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// int64 index 且 H*W > INT32_MAX: isInt32Meet=0, 对齐切 W 成功, tilingKey=100+10=110
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_Int64NoCheckRange)
{
    gert::StorageShape xShape = {{1, 1, 65536, 32768}, {1, 1, 65536, 32768}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 4096, 2048}, {1, 1, 4096, 2048}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {16, 16};
    std::vector<int64_t> strides = {16, 16};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 110;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// int64 index + overlap 兜底切分: isCheckRange=1 且 isInt32Meet=0, tilingKey=101+10=111
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_Int64CheckRange)
{
    gert::StorageShape xShape = {{1, 1, 65536, 32768}, {1, 1, 65536, 32768}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 32767, 16383}, {1, 1, 32767, 16383}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {3, 3};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT64;
    int64_t index_dtype = 9;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 111;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// ==================== NCHW scalar tiling 模板(CalcBase/CalcGradArgmax 各分支) ====================

// NCHW 模板 UB 不足落到 scalar: CalcBase 走 perH 切分分支(hOutputInner=inputUb/perHSize),
// CalcGradArgmaxInner/InnerTail 走第三分支(wInputInner <= argmaxCountInUB), tilingKey=301
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3ScalarExtra_CalcBasePerH)
{
    gert::StorageShape xShape = {{1, 1, 4096, 64}, {1, 1, 4096, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 2049, 63}, {1, 1, 2049, 63}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2048, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// NCHW 模板全 overlap 落到 scalar: nc=65 时 ncSizePerCore*hwSize 超 UB 而 hwSize 可容纳,
// CalcBase 走 hw 切分分支(highAxisInner=inputUb/hwSize), tilingKey=301
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3ScalarExtra_CalcBaseHwSplit)
{
    gert::StorageShape xShape = {{65, 1, 128, 64}, {65, 1, 128, 64}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{65, 1, 1, 1}, {65, 1, 1, 1}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {128, 64};
    std::vector<int64_t> strides = {128, 64};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// wX 超大(wX=16384): perHSize 超 UB, CalcBase 走 perW 切分分支(wOutputInner=inputUb/4),
// wInputInner < wGrad 时 CalcGradArgmaxInner/InnerTail 走早退分支, tilingKey=301
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3ScalarExtra_CalcBasePerWEarlyReturn)
{
    gert::StorageShape xShape = {{1, 1, 4096, 16384}, {1, 1, 4096, 16384}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 2049, 16383}, {1, 1, 2049, 16383}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2048, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// nc 超大(nc=491521): CalcBase 走整切分支, highAxisInner*plane 超 argmaxCountInUB 而 plane 可容纳,
// CalcGradArgmaxInner/InnerTail 走第二分支(argmaxNc=count/plane), 同时覆盖带 pad 的 isPad 置位, tilingKey=301
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3ScalarExtra_CalcGradArgmaxSplitNc)
{
    gert::StorageShape xShape = {{491521, 1, 1, 1}, {491521, 1, 1, 1}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{491521, 1, 2, 2}, {491521, 1, 2, 2}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {16, 16};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {8, 8};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// wGrad 超过 argmaxCountInUB(wGrad=7681>7680): plane 与 wInputInner 均超 UB 容量,
// CalcGradArgmaxInner/InnerTail 走第四分支(argmaxW=count), tilingKey=301
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3ScalarExtra_CalcGradArgmaxSplitW)
{
    gert::StorageShape xShape = {{1, 1, 4096, 7682}, {1, 1, 4096, 7682}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{1, 1, 2049, 7681}, {1, 1, 2049, 7681}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2048, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    ge::DataType dtype_index = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 301;
    ExecuteNchwExtraTestCase(xShape, yShape, gradShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                             dtype_index, ceil_mode, data_format, except_tilingkey);
}

// 直接构造 NCHW tiling 模板实例, 覆盖 MaxPoolGradNCHWTilingCommon::GetBaseData/GetSplitData 接口
TEST_F(MaxPoolGradWithArgmaxV3NchwExtraTiling, MaxPoolGradWithArgmaxV3NCHWExtra_GetBaseDataAndGetSplitData)
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
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    gert::StorageShape xShape = {{16, 4, 56, 56}, {16, 4, 56, 56}};
    gert::StorageShape yShape = xShape;
    gert::StorageShape argmaxShape = {{16, 4, 55, 55}, {16, 4, 55, 55}};
    gert::StorageShape gradShape = argmaxShape;
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    // faker 的 Build 要求 CompileInfo 非空, 传入占位对象
    optiling::MaxPoolGradWithArgmaxCompileInfo placeholder_compile_info;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPoolGradWithArgmaxV3")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &gradShape, &argmaxShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&placeholder_compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                                  {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                                  {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                                  {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    optiling::MaxPoolGradWithArgmaxV3NCHWTiling nchwTiling(tiling_context);
    nchwTiling.inputData.inputFormat = ge::Format::FORMAT_NCHW;
    nchwTiling.inputData.nX = 16;
    nchwTiling.inputData.cX = 4;
    nchwTiling.inputData.hX = 56;
    nchwTiling.inputData.wX = 56;
    nchwTiling.inputData.nGrad = 16;
    nchwTiling.inputData.cGrad = 4;
    nchwTiling.inputData.hGrad = 55;
    nchwTiling.inputData.wGrad = 55;
    nchwTiling.inputData.hKernel = 2;
    nchwTiling.inputData.wKernel = 2;
    nchwTiling.inputData.hStride = 1;
    nchwTiling.inputData.wStride = 1;
    nchwTiling.inputData.hPad = 0;
    nchwTiling.inputData.wPad = 0;
    nchwTiling.inputData.hDilation = 1;
    nchwTiling.inputData.wDilation = 1;
    nchwTiling.inputData.inputDtype = ge::DT_FLOAT;
    nchwTiling.inputData.indexDtype = ge::DT_INT32;
    nchwTiling.hardwareData.coreNum = 64;
    nchwTiling.hardwareData.ubSize = 245760;

    EXPECT_TRUE(nchwTiling.IsCapable());
    auto baseData = nchwTiling.nchwTilingCommon.GetBaseData();
    EXPECT_EQ(baseData.totalCoreNum, 64);
    EXPECT_EQ(baseData.availableUb, 244736);
    EXPECT_EQ(baseData.inputNCSize, 64);
    auto splitData = nchwTiling.nchwTilingCommon.GetSplitData();
    EXPECT_EQ(splitData.highAxisInner, 1);
    EXPECT_EQ(splitData.hOutputInner, 1);
    EXPECT_EQ(splitData.wOutputInner, 56);
}
