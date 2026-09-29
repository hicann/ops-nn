/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
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
#include "../../../../op_host/arch35/max_pool_with_argmax_v3_tiling.h"
#include "../../../../op_host/arch35/max_pool_with_argmax_v3_simt_tiling.h"
#include "../../../../op_host/arch35/max_pool_with_argmax_v3_gather_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

/*
 * 覆盖增强用例，模板优先级: gather(0) -> big_kernel_mul_core(4) -> big_kernel(6) -> nhwc(20) -> simt(100)
 * 1. gather: dilation > 1 时 IsCapable 直接失效, 可用于将流量导到后续模板;
 *    NCHW 下 dilation > 1 时 nhwc/big_kernel/mul_core 同样失效, 最终落到 simt 模板。
 * 2. base: platform info 为空时 GetPlatformInfo 走 compile info 分支;
 *    3维输入(NCHW->CHW/HWC)走 base 的 CHW/HWC 解析分支。
 * 3. gather: TrySplitNC/TrySplitH/TrySplitW 失败后触发 BinarySearch/IsMeetUBSize。
 */
class MaxPoolWithArgmaxV3TilingExtra : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        // 开启 INFO 日志, 使 simt DoOpTiling 中 OP_LOGI 内的 ToString(tiling) 真正执行
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", true);
        std::cout << "MaxPoolWithArgmaxV3TilingExtra SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "0", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "1", true);
        std::cout << "MaxPoolWithArgmaxV3TilingExtra TearDown" << std::endl;
    }
};

// 暴露 protected 接口, 用于直接测试 simt 模板自身的参数校验分支(这些分支在完整 tiling 流水线中
// 会被优先级更高的 gather 模板(base GetShapeAttrsInfo)提前拦截, 无法通过 tiling_func 触发)
class MaxPoolWithArgmaxV3TilingSimtForTest : public optiling::MaxPoolWithArgmaxV3TilingSIMT {
public:
    using optiling::MaxPoolWithArgmaxV3TilingSIMT::MaxPoolWithArgmaxV3TilingSIMT;

    ge::graphStatus GetShapeAttrsInfoForTest() { return optiling::MaxPoolWithArgmaxV3TilingSIMT::GetShapeAttrsInfo(); }

    std::string ToStringForTest(optiling::MaxPoolWithArgmaxV3SimtTilingData& tilingData)
    {
        return optiling::MaxPoolWithArgmaxV3TilingSIMT::ToString(tilingData);
    }
};

static void ExecuteExtraTestCase(gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape argmaxShape,
                                 std::vector<int64_t> ksize, std::vector<int64_t> strides, std::vector<int64_t> pads,
                                 std::vector<int64_t> dilation, ge::DataType dtype, int64_t index_dtype, bool ceil_mode,
                                 std::string data_format, bool with_platform, ge::graphStatus expect_status,
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
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
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

    // tilingFunc simulate, with_platform 为 false 时不设置 platform info,
    // 触发 base GetPlatformInfo 走 compile info 分支
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(with_platform ? reinterpret_cast<const void*>(&platform_info) :
                                                    static_cast<const void*>(nullptr))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_NE(tiling_context, nullptr);
    if (with_platform) {
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                    intrinsics);
    } else {
        ASSERT_EQ(tiling_context->GetPlatformInfo(), nullptr);
    }

    EXPECT_EQ(tiling_func(tiling_context), expect_status);
    if (expect_status == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tiling_context->GetTilingKey(), except_tilingkey);
    }
}

// 直接调用 simt 模板自身的 GetShapeAttrsInfo, 覆盖其独立校验分支
static void ExecuteSimtGetShapeAttrsInfo(gert::StorageShape xShape, gert::StorageShape yShape,
                                         gert::StorageShape argmaxShape, std::vector<int64_t> ksize,
                                         std::vector<int64_t> strides, std::vector<int64_t> pads,
                                         std::vector<int64_t> dilation, ge::DataType dtype, int64_t index_dtype,
                                         bool ceil_mode, std::string data_format, ge::graphStatus expectStatus)
{
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    // faker 的 Build 要求 CompileInfo/PlatformInfo 均非空
    optiling::MaxPoolWithArgmaxV3CompileInfo placeholder_compile_info;
    fe::PlatFormInfos direct_platform_info;
    direct_platform_info.Init();
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPoolWithArgmaxV3")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&placeholder_compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&direct_platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_NE(tiling_context, nullptr);
    MaxPoolWithArgmaxV3TilingSimtForTest simtTiling(tiling_context);
    EXPECT_EQ(simtTiling.GetShapeAttrsInfoForTest(), expectStatus);
}

/*
 * simt: NHWC 4维 + dilation > 1, 前序模板(gather/mul_core/big_kernel/nhwc)均不匹配,
 * 落到 simt 模板, 覆盖 NHWC 维度映射(nDimPos/cDimPos/hDimPos/wDimPos)与 NHWC int32 tiling key(500002)
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_Nhwc_Dilation_Fp16)
{
    gert::StorageShape xShape = {{1, 64, 64, 3}, {1, 64, 64, 3}};
    gert::StorageShape yShape = {{1, 32, 32, 3}, {1, 32, 32, 3}};
    gert::StorageShape argmaxShape = {{1, 32, 32, 3}, {1, 32, 32, 3}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {2, 2};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NHWC";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 500002);
}

// simt: NHWC + 输出元素个数超过 INT32 上限, 覆盖 SIMT_NHWC_TILING_KEY_INT64(500012)
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_Nhwc_Int64TilingKey)
{
    gert::StorageShape xShape = {{1, 92684, 92684, 1}, {1, 92684, 92684, 1}};
    gert::StorageShape yShape = {{1, 46342, 46342, 1}, {1, 46342, 46342, 1}};
    gert::StorageShape argmaxShape = {{1, 46342, 46342, 1}, {1, 46342, 46342, 1}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {2, 2};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NHWC";
    // 1 * 1 * 46342 * 46342 = 2147792964 > INT32_MAX(2147483647)
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 500012);
}

// simt: NCHW + 输出元素个数超过 INT32 上限, 覆盖 SIMT_NCHW_TILING_KEY_INT64(500011)
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_Nchw_Int64TilingKey)
{
    gert::StorageShape xShape = {{1, 1, 92684, 92684}, {1, 1, 92684, 92684}};
    gert::StorageShape yShape = {{1, 1, 46342, 46342}, {1, 1, 46342, 46342}};
    gert::StorageShape argmaxShape = {{1, 1, 46342, 46342}, {1, 1, 46342, 46342}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {2, 2};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    // 1 * 1 * 46342 * 46342 = 2147792964 > INT32_MAX(2147483647)
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 500011);
}

/*
 * simt: 4维输入 + data_format="CHW", base GetShapeAttrsInfo 按 CHW 分支解析成功,
 * dilation > 1 使前序模板全部失效后落到 simt, simt 仅支持 NCHW/NHWC, 覆盖非法 format 校验分支
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_InvalidFormat_Chw)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {2, 2};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "CHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

/*
 * simt: 3维输入(NCHW), base 按 CHW 分支解析成功, dilation > 1 使前序模板全部失效后落到 simt,
 * simt 要求输入必须为 4 维, 覆盖维度校验分支
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_InvalidDim_ThreeDims)
{
    gert::StorageShape xShape = {{3, 32, 32}, {3, 32, 32}};
    gert::StorageShape yShape = {{3, 16, 16}, {3, 16, 16}};
    gert::StorageShape argmaxShape = {{3, 16, 16}, {3, 16, 16}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {2, 2};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

/*
 * simt: 非法 dtype 校验分支。完整流水线中该分支被 gather 模板(base GetShapeAttrsInfo 的 dtype 校验,
 * 优先级 0)提前拦截无法触达, 因此直接调用 simt 自身的 GetShapeAttrsInfo 验证
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_DtypeInvalid_Direct)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteSimtGetShapeAttrsInfo(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                                 ceil_mode, data_format, ge::GRAPH_FAILED);
}

/*
 * simt: indices 与 y shape 不一致的校验分支。完整流水线中该分支同样被 base GetShapeAttrsInfo 提前拦截,
 * 直接调用 simt 自身的 GetShapeAttrsInfo 验证
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_IndicesShapeMismatch_Direct)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 31, 32}, {1, 1, 31, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteSimtGetShapeAttrsInfo(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype,
                                 ceil_mode, data_format, ge::GRAPH_FAILED);
}

/*
 * simt: 直接调用 ToString 覆盖 tiling 数据序列化逻辑。
 * ToString 仅在日志级别允许 INFO 时经 DoOpTiling 的 OP_LOGI 触发(本测试套件 SetUp 已开启 INFO 日志),
 * 此处直接调用保证该函数被稳定覆盖并校验输出内容
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Simt_ToString_Direct)
{
    MaxPoolWithArgmaxV3TilingSimtForTest simtTiling(nullptr);
    optiling::MaxPoolWithArgmaxV3SimtTilingData tilingData;
    tilingData.set_threadNums(256);
    tilingData.set_blockNums(64);
    tilingData.set_nDim(4);
    tilingData.set_cDim(50);
    tilingData.set_hInDim(16);
    tilingData.set_wInDim(16);
    tilingData.set_hOutDim(8);
    tilingData.set_wOutDim(8);
    tilingData.set_kSizeH(2);
    tilingData.set_kSizeW(2);
    tilingData.set_stridesH(2);
    tilingData.set_stridesW(2);
    tilingData.set_padH(0);
    tilingData.set_padW(0);
    tilingData.set_dilationH(2);
    tilingData.set_dilationW(2);
    tilingData.set_ceilMode(1);
    std::string tilingStr = simtTiling.ToStringForTest(tilingData);
    EXPECT_NE(tilingStr.find("threadNums:256"), std::string::npos);
    EXPECT_NE(tilingStr.find("blockNums:64"), std::string::npos);
    EXPECT_NE(tilingStr.find("nDim:4"), std::string::npos);
    EXPECT_NE(tilingStr.find("cDim:50"), std::string::npos);
    EXPECT_NE(tilingStr.find("hInDim:16"), std::string::npos);
    EXPECT_NE(tilingStr.find("wInDim:16"), std::string::npos);
    EXPECT_NE(tilingStr.find("hOutDim:8"), std::string::npos);
    EXPECT_NE(tilingStr.find("wOutDim:8"), std::string::npos);
    EXPECT_NE(tilingStr.find("kSizeH:2"), std::string::npos);
    EXPECT_NE(tilingStr.find("kSizeW:2"), std::string::npos);
    EXPECT_NE(tilingStr.find("stridesH:2"), std::string::npos);
    EXPECT_NE(tilingStr.find("stridesW:2"), std::string::npos);
    EXPECT_NE(tilingStr.find("padH:0"), std::string::npos);
    EXPECT_NE(tilingStr.find("padW:0"), std::string::npos);
    EXPECT_NE(tilingStr.find("dilationH:2"), std::string::npos);
    EXPECT_NE(tilingStr.find("dilationW:2"), std::string::npos);
    EXPECT_NE(tilingStr.find("ceilMode:1"), std::string::npos);
}

/*
 * base: 直接实例化 MaxPoolWithArgmaxV3BaseTiling 执行完整 DoTiling 流程,
 * 覆盖 base 的 IsCapable(恒 true)/DoOpTiling/GetTilingKey(恒 0)/PostTiling(空实现)。
 * 这些接口被全部 5 个子类模板覆盖, 通过注册流水线无法触达 base 自身的实现
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_DoTiling_Direct)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};

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

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPoolWithArgmaxV3")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<const void*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    optiling::MaxPoolWithArgmaxV3BaseTiling baseTiling(tiling_context);
    EXPECT_EQ(baseTiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 0ULL);
}

/*
 * base: tiling context 不设置 platform info(nullptr) 时 GetPlatformInfo 走 compile info 分支:
 * metadef OpTilingContextBuilder::BuildTilingContext 对空 platform info 走 GE 断言报错路径
 * ("Platform info is nullptr"), TilingContextFaker 无法构造 GetPlatformInfo()==nullptr 的合法
 * TilingContext, 该分支在本 UT 框架下不可达, 详见覆盖报告"无法覆盖代码"清单
 */

// base: 输入维度非 3/4 维, 覆盖 GetShapeAttrsInfo 维度数校验分支
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_InvalidDimCount)
{
    gert::StorageShape xShape = {{64, 64}, {64, 64}};
    gert::StorageShape yShape = {{32, 32}, {32, 32}};
    gert::StorageShape argmaxShape = {{32, 32}, {32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

// base: 输入 dtype 非法(仅支持 float/float16/bfloat16), 覆盖 dtype 校验分支
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_InvalidDtype)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_INT32;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

// base: indices 与 y shape 不一致, 覆盖 shape 一致性校验分支
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_IndicesShapeMismatch)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 31, 32}, {1, 1, 31, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

// base: 输出 H 维为 0, 覆盖输出 H/W 维必须大于 0 的校验分支
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_OutputDimZero)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 0, 1}, {1, 1, 0, 1}};
    gert::StorageShape argmaxShape = {{1, 1, 0, 1}, {1, 1, 0, 1}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

// base: pads 超过 ksize 一半, 覆盖 pads 合法性校验分支
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_PadsExceedHalfKernel)
{
    gert::StorageShape xShape = {{1, 1, 64, 64}, {1, 1, 64, 64}};
    gert::StorageShape yShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 1, 32, 32}, {1, 1, 32, 32}};
    std::vector<int64_t> ksize = {3, 3};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {2, 2};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_FAILED, 0);
}

/*
 * base: 索引 dtype 属性取非法值(非 3/9), 覆盖 switch default 分支(按 INT32 处理),
 * tiling 正常成功并由 gather 模板处理
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_IndexDtypeDefault)
{
    gert::StorageShape xShape = {{2, 2, 34, 34}, {2, 2, 34, 34}};
    gert::StorageShape yShape = {{2, 2, 17, 17}, {2, 2, 17, 17}};
    gert::StorageShape argmaxShape = {{2, 2, 17, 17}, {2, 2, 17, 17}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 5;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 300001);
}

/*
 * base: 3维输入 + NCHW, 覆盖 base 的 CHW 格式解析分支(inputFormat 重写为 CHW, nInput=1, batches=1*C),
 * tiling 由 gather 模板正常处理
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Base_ThreeDimChwFormat)
{
    gert::StorageShape xShape = {{3, 32, 32}, {3, 32, 32}};
    gert::StorageShape yShape = {{3, 16, 16}, {3, 16, 16}};
    gert::StorageShape argmaxShape = {{3, 16, 16}, {3, 16, 16}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 300001);
}

/*
 * gather: N*C=200 > 总核数(64), TrySplitNC 第一次按 CeilDiv(200,64)=4 切分不满足核数要求,
 * highAxisInner=1 时满足, 触发 BinarySearch 在 NC 轴搜索最优切分(结果 highAxisInner=3, blockDim=34)
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Gather_TrySplitNC_BinarySearch)
{
    gert::StorageShape xShape = {{4, 50, 16, 16}, {4, 50, 16, 16}};
    gert::StorageShape yShape = {{4, 50, 8, 8}, {4, 50, 8, 8}};
    gert::StorageShape argmaxShape = {{4, 50, 8, 8}, {4, 50, 8, 8}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 300001);
}

/*
 * gather: N*C=4 且输出 32*32, TrySplitNC 核数不满足, TrySplitH 以 hOutputInner=1 满足条件,
 * 触发 BinarySearch 在 H 轴搜索最优切分(结果 hOutputInner=2, blockDim=64)
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Gather_TrySplitH_BinarySearch)
{
    gert::StorageShape xShape = {{1, 4, 33, 33}, {1, 4, 33, 33}};
    gert::StorageShape yShape = {{1, 4, 32, 32}, {1, 4, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 4, 32, 32}, {1, 4, 32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 300001);
}

/*
 * gather: N*C=1, 输出 8*64, TrySplitNC/TrySplitH 核数均不满足, TrySplitW 以 wOutputInner=1 满足条件,
 * 触发 BinarySearch 在 W 轴搜索最优切分(结果 wOutputInner=9, blockDim=64)
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Gather_TrySplitW_BinarySearch)
{
    gert::StorageShape xShape = {{1, 1, 9, 65}, {1, 1, 9, 65}};
    gert::StorageShape yShape = {{1, 1, 8, 64}, {1, 1, 8, 64}};
    gert::StorageShape argmaxShape = {{1, 1, 8, 64}, {1, 1, 8, 64}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 300001);
}

/*
 * gather: ceil_mode=true 且 pads=0 时输出反推输入尺寸与实际输入不一致((16-1)*2+2=32 != 33),
 * 覆盖 InitializationVars 中 isPad 置 1 分支, tiling key 为带 padding 的 300002
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Gather_CeilMode_SetPadFlag)
{
    gert::StorageShape xShape = {{2, 2, 33, 33}, {2, 2, 33, 33}};
    gert::StorageShape yShape = {{2, 2, 16, 16}, {2, 2, 16, 16}};
    gert::StorageShape argmaxShape = {{2, 2, 16, 16}, {2, 2, 16, 16}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    ExecuteExtraTestCase(xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode,
                         data_format, true, ge::GRAPH_SUCCESS, 300002);
}

/*
 * gather: 直接实例化 gather 模板执行 DoTiling 后调用 IsMeetUBSize。
 * IsMeetUBSize 在 -O2 下被内联到全部调用点(TrySplitNC/H/W/BinarySearch), 其 out-of-line 实现无法
 * 通过 tiling 流水线触达; UT 编译带 -fno-access-control(仓库内 quant_batch_matmul_v3 等 UT 已有
 * 直接调用私有接口的先例), 此处直接调用并断言最终切分满足 UB 约束
 */
TEST_F(MaxPoolWithArgmaxV3TilingExtra, MaxPoolWithArgmaxV3Tiling_Gather_IsMeetUBSize_Direct)
{
    gert::StorageShape xShape = {{1, 4, 33, 33}, {1, 4, 33, 33}};
    gert::StorageShape yShape = {{1, 4, 32, 32}, {1, 4, 32, 32}};
    gert::StorageShape argmaxShape = {{1, 4, 32, 32}, {1, 4, 32, 32}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};

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

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("MaxPoolWithArgmaxV3")
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<const void*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    optiling::MaxPoolWithArgmaxV3GatherTiling gatherTiling(tiling_context);
    EXPECT_EQ(gatherTiling.DoTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 300001ULL);
    EXPECT_TRUE(gatherTiling.IsMeetUBSize());
}
