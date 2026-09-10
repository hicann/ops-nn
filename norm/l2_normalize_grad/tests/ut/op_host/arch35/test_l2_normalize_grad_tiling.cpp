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

// 平台参数(默认取真机 ascend950:platform_config/Ascend950DT_950x.ini 的 ub_size=253952、
// CORE_NUM=64、vector_reg_width=256)。三个值可单独置 0,用于打 GetPlatformInfo 的异常拒收分支。
struct FakePlatform {
    int64_t ubSize = 253952;
    int64_t coreNum = 64;
    int64_t vecRegWidth = 256;
};

void BuildPlatformRes(const FakePlatform& fp, map<string, string>& soc_infos, map<string, string>& aicore_spec,
                      map<string, string>& intrinsics)
{
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                              "Intrinsic_fix_pipe_l0c2out": false,
                              "Intrinsic_data_move_l12ub": true,
                              "Intrinsic_data_move_l0c2ub": true,
                              "Intrinsic_data_move_out2l1_nd2nz": false,
                              "UB_SIZE": )" +
                                 std::to_string(fp.ubSize) +
                                 R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                              "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                              "CORE_NUM": )" +
                                 std::to_string(fp.coreNum) + R"(}})";
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    // GetPlatFormInfos 只透传固定白名单键,vector_reg_width 不在其中,须显式补齐,
    // 否则 PlatformAscendC::GetVecRegLen() 返回 0,tiling 取平台信息即失败。
    aicore_spec["vector_reg_width"] = std::to_string(fp.vecRegWidth);
}

// 把构造好的平台三张表挂到 context 上(顺序固定:SoCInfo -> AICoreSpec -> 核数 -> intrinsic)。
void ApplyPlatformRes(gert::TilingContext* ctx, map<string, string>& soc_infos, map<string, string>& aicore_spec,
                      map<string, string>& intrinsics)
{
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
}

ge::graphStatus RunTiling(gert::StorageShape& xShape, gert::StorageShape& yShape, gert::StorageShape& dyShape,
                          gert::StorageShape& dxShape, const std::vector<int64_t>& dim, uint64_t& tilingKey,
                          ge::DataType xDt = ge::DT_FLOAT, ge::DataType yDt = ge::DT_FLOAT,
                          ge::DataType dyDt = ge::DT_FLOAT, ge::DataType dxDt = ge::DT_FLOAT,
                          ge::Format fmt = ge::FORMAT_ND, const FakePlatform& fp = FakePlatform{})
{
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    BuildPlatformRes(fp, soc_infos, aicore_spec, intrinsics);

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
    ApplyPlatformRes(tiling_context, soc_infos, aicore_spec, intrinsics);

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

ge::graphStatus RunTilingOnPlatform(const FakePlatform& fp, uint64_t& tilingKey)
{
    gert::StorageShape s = MakeShape({32, 512});
    return RunTiling(s, s, s, s, {1}, tilingKey, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::FORMAT_ND,
                     fp);
}
} // namespace

// ---------------- 平台信息异常:GetPlatformInfo 的三条拒收分支 ----------------
// 真机不可达,但假平台可以精确注入 —— 从没被执行过的错误分支既可能是死代码,
// 也可能把支持面内的输入判成 GRAPH_FAILED,必须各打一次。

TEST_F(L2NormalizeGradTiling, platform_core_num_zero_rejected)
{
    uint64_t key = 0;
    FakePlatform fp;
    fp.coreNum = 0;
    EXPECT_EQ(RunTilingOnPlatform(fp, key), ge::GRAPH_FAILED);
}

TEST_F(L2NormalizeGradTiling, platform_ub_size_zero_rejected)
{
    uint64_t key = 0;
    FakePlatform fp;
    fp.ubSize = 0;
    EXPECT_EQ(RunTilingOnPlatform(fp, key), ge::GRAPH_FAILED);
}

TEST_F(L2NormalizeGradTiling, platform_vec_reg_len_zero_rejected)
{
    uint64_t key = 0;
    FakePlatform fp;
    fp.vecRegWidth = 0;
    EXPECT_EQ(RunTilingOnPlatform(fp, key), ge::GRAPH_FAILED);
}

TEST_F(L2NormalizeGradTiling, l2_normalize_grad_tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
    ASSERT_NE(op_impl->tiling_parse, nullptr);
}

// ---------------- 正向：模板选择 ----------------

// inner==1 且整行(对齐 1VL 后)装得进 ubFactor -> full load
// 阈值不是写死常量,由 DeriveUbFactor 从 ubSize 解出:ascend950(ub_size=253952, VL=64 fp32 lane)
// 解得 ubFactor=6080,判据 AlignUp(D,64) <= 6080,即 D <= 6080。
TEST_F(L2NormalizeGradTiling, tilingkey_full_load_7000)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

// inner==1 且整行装不下 -> split D
TEST_F(L2NormalizeGradTiling, tilingkey_split_d_7010)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8192}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7010U);
}

// 选路边界逐格:D=6080 是 full_load 能吃下的最大行宽,+1 就跨到 split_d。
// (泛化用例集此前按写死的 6144 预测选路,导致这一档一直落在 7010、7000 上边界零覆盖)
TEST_F(L2NormalizeGradTiling, tilingkey_boundary_full_load_max)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({2, 6080}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

TEST_F(L2NormalizeGradTiling, tilingkey_boundary_split_d_min)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({2, 6081}, {1}, key), ge::GRAPH_SUCCESS);
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

// dim 不传/传空 = 不归约(对齐 ascend910b 的 GE 通路: proto 默认 {} -> tbe.sum(x, []) 恒等)。
// 退化成每元素自成一组 -> outer=totalNum, D=1, inner=1 -> full load。
TEST_F(L2NormalizeGradTiling, dim_empty_means_no_reduction)
{
    uint64_t keyEmpty = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, {}, keyEmpty), ge::GRAPH_SUCCESS);
    EXPECT_EQ(keyEmpty, 7000U);
}

// rank==1 且不传 dim:A2 正常算(不归约),不可按“默认轴 1”拒收(issue #31 的真定性)。
TEST_F(L2NormalizeGradTiling, dim_empty_rank1_accepted)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({1024}, {}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

// 重复/乱序/正负混写:折算去重排序后与规范形式等价,不可拒收。
TEST_F(L2NormalizeGradTiling, dim_duplicated_and_unordered_equivalent)
{
    uint64_t keyDup = 0;
    uint64_t keyMix = 0;
    uint64_t keyRef = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {1, 1, 1}, keyDup), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {1, -2}, keyMix), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {1}, keyRef), ge::GRAPH_SUCCESS);
    EXPECT_EQ(keyDup, keyRef);
    EXPECT_EQ(keyMix, keyRef);
}

// 连续多轴:折成一根等效轴。{1,2} 于 [4,8,16] 上 inner==1 -> full load。
TEST_F(L2NormalizeGradTiling, dim_contiguous_multi_axis_accepted)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {1, 2}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

// 乱序传入的连续多轴与升序等价。
TEST_F(L2NormalizeGradTiling, dim_contiguous_multi_axis_unordered)
{
    uint64_t keyDesc = 0;
    uint64_t keyAsc = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {2, 1}, keyDesc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {1, 2}, keyAsc), ge::GRAPH_SUCCESS);
    EXPECT_EQ(keyDesc, keyAsc);
}

// 全轴归约(连续区间覆盖 0..rank-1)。
TEST_F(L2NormalizeGradTiling, dim_all_axes_accepted)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {0, 1, 2}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

// 连续多轴且尾部还有保留轴 -> strided。
TEST_F(L2NormalizeGradTiling, dim_contiguous_multi_axis_strided)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({2, 4, 8, 16}, {1, 2}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7020U);
}

// 归约段整段放不下 UB 时改走 7030 沿 D 分块(不再钳位突破预算,也不再拒收)。
TEST_F(L2NormalizeGradTiling, tilingkey_strided_split_7030)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({2, 40000, 2}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7030U);
}

// D 超过 DataCopyPad blockCount(uint16)上限时同样由 7030 承接,不得静默截断。
TEST_F(L2NormalizeGradTiling, strided_split_handles_d_over_uint16)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({1, 70000, 2}, {1}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7030U);
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

// 非连续轴集(历史 5HD 的 [1,4] 形态):当前不支持,须显式拒收。
TEST_F(L2NormalizeGradTiling, reject_non_contiguous_dim_5hd)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16, 32, 16}, {1, 4}, key), ge::GRAPH_FAILED);
}

// 非连续轴集:中间隔着一根保留轴。
TEST_F(L2NormalizeGradTiling, reject_non_contiguous_dim_gap)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {0, 2}, key), ge::GRAPH_FAILED);
}

// 折算后才出现的非连续:{-1, 1} 于 rank4 上是 {1, 3}。
TEST_F(L2NormalizeGradTiling, reject_non_contiguous_after_folding)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({2, 4, 8, 16}, {-1, 1}, key), ge::GRAPH_FAILED);
}

// dim 数组长度上限 20(去重后有效轴至多 rank 个,超长必为冗余)。
TEST_F(L2NormalizeGradTiling, dim_length_at_limit_accepted)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, std::vector<int64_t>(20, 1), key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 7000U);
}

TEST_F(L2NormalizeGradTiling, reject_dim_length_over_limit)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({32, 512}, std::vector<int64_t>(21, 1), key), ge::GRAPH_FAILED);
}

// 混合列表里只要有一个元素越界就整体拒收(不丢弃非法元素,对齐 A2)。
TEST_F(L2NormalizeGradTiling, reject_dim_mixed_valid_and_out_of_range)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingSameShape({4, 8, 16}, {1, 5}, key), ge::GRAPH_FAILED);
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
