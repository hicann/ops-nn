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
 * \file test_instance_norm_grad_tiling.cpp
 * \brief Tiling UT for InstanceNormGrad (arch35)。
 *        TilingKey = 模式基值 + dtype 偏移：full_load 101(fp32)/102(fp16)，recompute 301/302，empty 500。
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/instance_norm_grad_tiling_arch35.h"

using namespace std;
using namespace ge;

class InstanceNormGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InstanceNormGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "InstanceNormGradTiling TearDown" << std::endl; }
};

namespace {
constexpr const char* kOpType = "InstanceNormGrad";

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    return s;
}

// 五输入(dy/x/variance/mean/gamma) 三输出(pd_x/pd_gamma/pd_beta)。
// tilingKey 仅在返回 GRAPH_SUCCESS 时被写入。
ge::graphStatus RunTiling(const std::vector<int64_t>& dyDims, const std::vector<int64_t>& xDims,
                          const std::vector<int64_t>& varDims, const std::vector<int64_t>& meanDims,
                          const std::vector<int64_t>& gammaDims, uint64_t& tilingKey, ge::DataType dt = ge::DT_FLOAT,
                          ge::DataType xDt = ge::DT_FLOAT, ge::DataType pdxDt = ge::DT_FLOAT,
                          ge::DataType paramDt = ge::DT_UNDEFINED, ge::DataType pdgDt = ge::DT_UNDEFINED)
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
    // tiling 从 CompileInfo 取平台参数（不走 PlatformInfo），此处按 Ascend950 规格填充。
    optiling::InstanceNormGradCompileInfo compile_info;
    compile_info.totalCoreNum = 64;
    compile_info.sysWorkspaceSize = 16U * 1024U * 1024U;
    compile_info.ubSizePlatForm = 245760U;
    compile_info.vectorLen = 256U; // VRegSize，单位字节
    compile_info.blockSize = 32U;
    compile_info.isRegBase = true;

    // 未显式指定时，variance/mean/gamma 与 pd_gamma/pd_beta 跟随 dy 的 dtype。
    ge::DataType pDt = (paramDt == ge::DT_UNDEFINED) ? dt : paramDt;
    ge::DataType gDt = (pdgDt == ge::DT_UNDEFINED) ? dt : pdgDt;

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

    gert::StorageShape dy = MakeShape(dyDims);
    gert::StorageShape x = MakeShape(xDims);
    gert::StorageShape var = MakeShape(varDims);
    gert::StorageShape mean = MakeShape(meanDims);
    gert::StorageShape gamma = MakeShape(gammaDims);
    gert::StorageShape pdx = MakeShape(xDims);
    gert::StorageShape pdg = MakeShape(gammaDims);
    gert::StorageShape pdb = MakeShape(gammaDims);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&dy, &x, &var, &mean, &gamma})
                      .OutputShapes({&pdx, &pdg, &pdb})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(1, xDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(2, pDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(3, pDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeInputTd(4, pDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(0, pdxDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(1, gDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
                      .NodeOutputTd(2, gDt, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC)
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

// 形如 [N, D, H, W, C] 的合法一组输入。
ge::graphStatus RunTilingNDHWC(int64_t n, int64_t d, int64_t h, int64_t w, int64_t c, uint64_t& tilingKey,
                               ge::DataType dt = ge::DT_FLOAT)
{
    return RunTiling({n, d, h, w, c}, {n, d, h, w, c}, {n, 1, 1, 1, c}, {n, 1, 1, 1, c}, {c}, tilingKey, dt, dt, dt);
}
} // namespace

TEST_F(InstanceNormGradTiling, instance_norm_grad_tiling_registered)
{
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);
}

// ---------------- 正向：模式 × dtype ----------------

// 小 M，UB 放得下整块 -> full load (fp32)
TEST_F(InstanceNormGradTiling, tilingkey_full_load_fp32_101)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(2, 2, 4, 4, 32, key, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// 同上，fp16 偏移
TEST_F(InstanceNormGradTiling, tilingkey_full_load_fp16_102)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(2, 2, 4, 4, 32, key, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 102U);
}

// 大 M，UB 放不下 -> recompute (fp32)
TEST_F(InstanceNormGradTiling, tilingkey_recompute_fp32_301)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(1, 64, 128, 128, 64, key, ge::DT_FLOAT), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 301U);
}

// 大 M，fp16 -> recompute 偏移
TEST_F(InstanceNormGradTiling, tilingkey_recompute_fp16_302)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(1, 64, 128, 128, 64, key, ge::DT_FLOAT16), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 302U);
}

// x 为空 -> empty 路径
TEST_F(InstanceNormGradTiling, tilingkey_empty_500)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({0, 2, 4, 4, 32}, {0, 2, 4, 4, 32}, {0, 1, 1, 1, 32}, {0, 1, 1, 1, 32}, {32}, key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 500U);
}

// C 轴为 0：gamma 同样为空，仍须落 empty 路径。若判据带 gammaSize != 0，这里会掉进主 tiling
// 并在 BlockTiling 触发 CeilDiv(C_, 0) 除零。
TEST_F(InstanceNormGradTiling, tilingkey_empty_c0_500)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 1, 2, 2, 0}, {2, 1, 2, 2, 0}, {2, 1, 1, 1, 0}, {2, 1, 1, 1, 0}, {0}, key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 500U);
}

// ---------------- rank 覆盖：tiling 放宽到 dimNum >= 2（A2 仅 NDHWC 5D） ----------------

// rank 2：[N, C]，M = 1（中间维为空乘积）
TEST_F(InstanceNormGradTiling, rank2_full_load_101)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({8, 64}, {8, 64}, {8, 64}, {8, 64}, {64}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// rank 3：[N, M, C]
TEST_F(InstanceNormGradTiling, rank3_full_load_101)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({4, 16, 32}, {4, 16, 32}, {4, 1, 32}, {4, 1, 32}, {32}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// rank 4：[N, H, W, C]
TEST_F(InstanceNormGradTiling, rank4_full_load_101)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 8, 8, 32}, {2, 8, 8, 32}, {2, 1, 1, 32}, {2, 1, 1, 32}, {32}, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// rank 6：中间四维一起构成 M
TEST_F(InstanceNormGradTiling, rank6_full_load_101)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 2, 4, 4, 32}, {2, 2, 2, 4, 4, 32}, {2, 1, 1, 1, 1, 32}, {2, 1, 1, 1, 1, 32}, {32}, key),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// ---------------- 核数边界：UT 平台 CORE_NUM = 64 ----------------

// N 恰好等于核数
TEST_F(InstanceNormGradTiling, coreboundary_n_equals_corenum)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(64, 2, 4, 4, 32, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// N 比核数少 1：触发 C 轴切分（cTileNum > 1）占满空闲核
TEST_F(InstanceNormGradTiling, coreboundary_n_below_corenum)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(63, 2, 4, 4, 32, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// N 比核数多 1：尾核路径
TEST_F(InstanceNormGradTiling, coreboundary_n_above_corenum)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(65, 2, 4, 4, 32, key), ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 101U);
}

// ---------------- 反向：非法输入必须被拦截 ----------------

// dtype 非 fp16/fp32
TEST_F(InstanceNormGradTiling, reject_invalid_dtype)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTilingNDHWC(2, 2, 4, 4, 32, key, ge::DT_INT32), ge::GRAPH_FAILED);
}

// dy 与 x 的 dtype 不一致
TEST_F(InstanceNormGradTiling, reject_dtype_mismatch_dy_x)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key, ge::DT_FLOAT,
                        ge::DT_FLOAT16, ge::DT_FLOAT),
              ge::GRAPH_FAILED);
}

// pd_x 的 dtype 与输入不一致
TEST_F(InstanceNormGradTiling, reject_dtype_mismatch_output)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// dy 与 x 形状不同
TEST_F(InstanceNormGradTiling, reject_shape_mismatch_dy_x)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 8, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key),
              ge::GRAPH_FAILED);
}

// dy 维数不足 2
TEST_F(InstanceNormGradTiling, reject_rank_less_than_two)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({32}, {32}, {32}, {32}, {32}, key), ge::GRAPH_FAILED);
}

// gamma 元素数不等于 C
TEST_F(InstanceNormGradTiling, reject_gamma_size_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {16}, key),
              ge::GRAPH_FAILED);
}

// variance 元素数不等于 N*C
TEST_F(InstanceNormGradTiling, reject_variance_size_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {1, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key),
              ge::GRAPH_FAILED);
}

// mean 元素数不等于 N*C
TEST_F(InstanceNormGradTiling, reject_mean_size_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {1, 1, 1, 1, 32}, {32}, key),
              ge::GRAPH_FAILED);
}

// gamma/variance/mean 的 dtype 必须与 dy 一致：kernel 按 dy 的 dtype 强转读取，不一致会静默读错数据。
TEST_F(InstanceNormGradTiling, reject_param_dtype_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// pd_gamma/pd_beta 的 dtype 必须与 dy 一致。
TEST_F(InstanceNormGradTiling, reject_pdgamma_dtype_mismatch)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key, ge::DT_FLOAT,
                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_FLOAT16),
              ge::GRAPH_FAILED);
}

// 全 fp16 是合法组合（ascend910b 契约：所有张量同 dtype）。
TEST_F(InstanceNormGradTiling, accept_all_fp16_consistent)
{
    uint64_t key = 0;
    EXPECT_EQ(RunTiling({2, 2, 4, 4, 32}, {2, 2, 4, 4, 32}, {2, 1, 1, 1, 32}, {2, 1, 1, 1, 32}, {32}, key,
                        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 102U);
}
