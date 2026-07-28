/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/l2_normalize_grad_tiling.h"

using namespace std;
using namespace ge;

class L2NormalizeGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "L2NormalizeGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "L2NormalizeGradTiling TearDown" << std::endl; }
};

namespace {
constexpr const char* kOpType = "L2NormalizeGrad";

// 构造一次 tiling 调用；tilingKey 仅在返回 GRAPH_SUCCESS 时被写入。
// 四个 dtype 分别对应 x / y / dy / dx，便于构造 dtype 不一致的反向用例。
// 由维度列表构造 StorageShape（origin 与 storage 一致）。
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

ge::graphStatus RunTiling(gert::StorageShape& xShape, gert::StorageShape& yShape, gert::StorageShape& dyShape,
                          gert::StorageShape& dxShape, const std::vector<int64_t>& dim, uint64_t& tilingKey,
                          ge::DataType xDt = ge::DT_FLOAT, ge::DataType yDt = ge::DT_FLOAT,
                          ge::DataType dyDt = ge::DT_FLOAT, ge::DataType dxDt = ge::DT_FLOAT,
                          ge::Format fmt = ge::FORMAT_ND)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": 64}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::L2NormalizeGradCompileInfo compile_info;

    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    if (op_impl == nullptr || op_impl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = op_impl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &yShape, &dyShape})
                      .OutputShapes({&dxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDt, fmt, fmt)
                      .NodeInputTd(1, yDt, fmt, fmt)
                      .NodeInputTd(2, dyDt, fmt, fmt)
                      .NodeOutputTd(0, dxDt, fmt, fmt)
                      .NodeAttrs({{"dim", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dim)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-4f)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr || tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tiling_func(tiling_context);
    if (ret == ge::GRAPH_SUCCESS) {
        tilingKey = tiling_context->GetTilingKey();
    }
    return ret;
}

// 四个张量同形同 dtype 的常规调用封装。
ge::graphStatus RunTilingSameShape(const std::vector<int64_t>& shape, const std::vector<int64_t>& dim,
                                   uint64_t& tilingKey, ge::DataType dt = ge::DT_FLOAT)
{
    gert::StorageShape s = MakeShape(shape);
    return RunTiling(s, s, s, s, dim, tilingKey, dt, dt, dt, dt);
}
} // namespace

TEST_F(L2NormalizeGradTiling, l2_normalize_grad_tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
    ASSERT_NE(op_impl->tiling_parse, nullptr);
}

// ---------------- 正向：模板选择 ----------------

// inner==1 且 D<=6144 -> full load
TEST_F(L2NormalizeGradTiling, tilingkey_full_load_7000)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

// inner==1 且 D>6144 -> split D
TEST_F(L2NormalizeGradTiling, tilingkey_split_d_7010)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8192}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7010U);
}

// inner>1（归约轴之后还有维度）-> strided
TEST_F(L2NormalizeGradTiling, tilingkey_strided_7020)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 64, 32}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7020U);
}

// 元素总数为 0 -> empty
TEST_F(L2NormalizeGradTiling, tilingkey_empty_8000)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({0, 512}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 8000U);
}

// fp16 走同一套模板选择
TEST_F(L2NormalizeGradTiling, tilingkey_full_load_fp16)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {1}, key, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

// ---------------- 正向：dim 取值 ----------------

// 负轴换算为正轴：-1 等价于最后一维
TEST_F(L2NormalizeGradTiling, dim_negative_axis_accepted)
{
    uint64_t keyNeg = 0;
    uint64_t keyPos = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {-1}, keyNeg), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingSameShape({32, 512}, {1}, keyPos), ge::GRAPH_SUCCESS);
    EXPECT_EQ(keyNeg, keyPos);
}

// dim=0（首轴），其后仍有维度 -> strided
TEST_F(L2NormalizeGradTiling, dim_first_axis_accepted)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {0}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7020U);
}

// dim 为空数组时取默认轴 1
TEST_F(L2NormalizeGradTiling, dim_empty_falls_back_to_default)
{
    uint64_t keyEmpty = 0;
    uint64_t keyOne = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {}, keyEmpty), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingSameShape({32, 512}, {1}, keyOne), ge::GRAPH_SUCCESS);
    EXPECT_EQ(keyEmpty, keyOne);
}

// ---------------- 反向：非法输入必须被拦截 ----------------

// dtype 非 fp16/fp32
TEST_F(L2NormalizeGradTiling, reject_invalid_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {1}, key, ge::DT_INT32), ge::GRAPH_FAILED);
}

// x 与 y 的 dtype 不一致
TEST_F(L2NormalizeGradTiling, reject_dtype_mismatch_between_inputs)
{
    uint64_t key = 0;
    gert::StorageShape s = MakeShape({32, 512});
    EXPECT_EQ(RunTiling(s, s, s, s, {1}, key, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT),
              ge::GRAPH_FAILED);
}

// 输出 dx 的 dtype 与输入不一致
TEST_F(L2NormalizeGradTiling, reject_dtype_mismatch_on_output)
{
    uint64_t key = 0;
    gert::StorageShape s = MakeShape({32, 512});
    EXPECT_EQ(RunTiling(s, s, s, s, {1}, key, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// x 与 dy 维数不同
TEST_F(L2NormalizeGradTiling, reject_rank_mismatch)
{
    uint64_t key = 0;
    gert::StorageShape x = MakeShape({32, 512});
    gert::StorageShape dy = MakeShape({32, 512, 1});
    EXPECT_EQ(RunTiling(x, x, dy, x, {1}, key), ge::GRAPH_FAILED);
}

// x 与 y 某一维大小不同
TEST_F(L2NormalizeGradTiling, reject_dim_size_mismatch)
{
    uint64_t key = 0;
    gert::StorageShape x = MakeShape({32, 512});
    gert::StorageShape y = MakeShape({32, 256});
    EXPECT_EQ(RunTiling(x, y, x, x, {1}, key), ge::GRAPH_FAILED);
}

// dim 传入多个轴（历史 5HD 的 [1,4] 形态）
TEST_F(L2NormalizeGradTiling, reject_multi_axis_dim)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16, 32, 16}, {1, 4}, key), ge::GRAPH_FAILED);
}

// dim 超出 [-x.dim(), x.dim()-1]
TEST_F(L2NormalizeGradTiling, reject_dim_out_of_range_positive)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {2}, key), ge::GRAPH_FAILED);
}

TEST_F(L2NormalizeGradTiling, reject_dim_out_of_range_negative)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {-3}, key), ge::GRAPH_FAILED);
}

// ---------------- 边界：NCHW 与 ND 布局等价 ----------------

// 4 维 NCHW 与 4 维 ND 内存排布相同，tiling 不因 format 标签拒绝。
TEST_F(L2NormalizeGradTiling, accept_nchw_same_layout_as_nd)
{
    uint64_t key = 0;
    gert::StorageShape s = MakeShape({2, 16, 4, 4});
    EXPECT_EQ(RunTiling(s, s, s, s, {1}, key, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_NCHW),
              ge::GRAPH_SUCCESS);
}
