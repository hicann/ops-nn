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
 * \file test_cla_gate_backward_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "any_value.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/cla_gate_backward_tiling_arch35.h"
#include "../../../../op_kernel/arch35/cla_gate_backward_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ClaGateBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ClaGateBackwardTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ClaGateBackwardTiling TearDown" << std::endl; }
};

namespace {

constexpr const char* kOpType = "ClaGateBackward";
constexpr int64_t kDefaultUbSize = 245760;
constexpr int64_t kDefaultCoreNum = 64;

int64_t CeilDivRef(int64_t a, int64_t b) { return b <= 0 ? 0 : (a + b - 1) / b; }

// 平台信息字符串：UB_SIZE / CORE_NUM 可按用例覆盖，用于触发 UB 预算与核数分支。
std::string MakeCompileInfoString(int64_t ubSize, int64_t coreNum)
{
    std::string info = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", )";
    info += R"("Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, )";
    info += R"("Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, )";
    info += R"("UB_SIZE": )" + std::to_string(ubSize) + R"(, "L2_SIZE": 33554432, "L1_SIZE": 524288, )";
    info += R"("L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": )";
    info += std::to_string(coreNum) + R"(}})";
    return info;
}

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (auto d : dims) {
        shape.MutableOriginShape().AppendDim(d);
        shape.MutableStorageShape().AppendDim(d);
    }
    return shape;
}

struct TilingOptions {
    ge::DataType dtype0 = ge::DT_BF16;
    ge::DataType dtype1 = ge::DT_BF16;
    ge::DataType dtype2 = ge::DT_BF16;
    ge::DataType dtype3 = ge::DT_BF16;
    ge::DataType dtype4 = ge::DT_BF16;
    bool setAttrs = true;
    std::string layout = "TND";
    int64_t ubSize = kDefaultUbSize;
    int64_t coreNum = kDefaultCoreNum;
};

// 驱动一次 arch35 tiling：5 路输入 + 4 路输出。输出 shape 不参与 tiling，直接复用输入。
ge::graphStatus RunTiling(const gert::StorageShape& gradShape, const gert::StorageShape& globalAttnShape,
                          const gert::StorageShape& localAttnShape, const gert::StorageShape& globalLogitsShape,
                          const gert::StorageShape& localLogitsShape, ClaGateBackwardTilingData& outTiling,
                          const TilingOptions& opt = TilingOptions())
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string compileInfoString = MakeCompileInfoString(opt.ubSize, opt.coreNum);
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    optiling::ClaGateBackwardCompileInfo compileInfo;
    compileInfo.coreNum = opt.coreNum;
    compileInfo.ubSize = opt.ubSize;

    auto param = gert::TilingData::CreateCap(4096);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspaceSizes = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::StorageShape grad = gradShape;
    gert::StorageShape globalAttn = globalAttnShape;
    gert::StorageShape localAttn = localAttnShape;
    gert::StorageShape globalLogits = globalLogitsShape;
    gert::StorageShape localLogits = localLogitsShape;

    gert::TilingContextFaker faker;
    faker.SetOpType(kOpType)
        .NodeIoNum(5, 4)
        .IrInstanceNum({1, 1, 1, 1, 1})
        .InputShapes({&grad, &globalAttn, &localAttn, &globalLogits, &localLogits})
        .OutputShapes({&globalAttn, &localAttn, &globalLogits, &localLogits})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, opt.dtype0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, opt.dtype1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, opt.dtype2, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, opt.dtype3, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(4, opt.dtype4, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, opt.dtype0, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, opt.dtype1, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, opt.dtype3, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, opt.dtype4, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(param.get())
        .Workspace(workspaceSizes);
    if (opt.setAttrs) {
        faker.NodeAttrs({{"input_attn_layout", Ops::NN::AnyValue::CreateFrom<std::string>(opt.layout)}});
    }
    auto holder = faker.Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
    if (tilingContext == nullptr || tilingContext->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto status = opImpl->tiling(tilingContext);
    if (status == ge::GRAPH_SUCCESS) {
        auto data = reinterpret_cast<const ClaGateBackwardTilingData*>(tilingContext->GetRawTilingData()->GetData());
        outTiling = *data;
    }
    return status;
}

// 标准 TND 输入的快捷封装：grad/global_attn/local_attn = [T, N, D]，logits = [T, N]。
ge::graphStatus RunTilingStandard(int64_t t, int64_t n, int64_t d, ClaGateBackwardTilingData& outTiling,
                                  const TilingOptions& opt = TilingOptions())
{
    return RunTiling(MakeShape({t, n, d}), MakeShape({t, n, d}), MakeShape({t, n, d}), MakeShape({t, n}),
                     MakeShape({t, n}), outTiling, opt);
}

// 校验 tiling 结果的内部一致性（不依赖 ReduceSum tmp 的具体数值）。
void ExpectTilingConsistent(const ClaGateBackwardTilingData& d)
{
    EXPECT_GT(d.usedCoreNum, 0);
    EXPECT_LE(d.usedCoreNum, kDefaultCoreNum);
    EXPECT_EQ(d.baseCoreHeads, d.totalHeads / d.usedCoreNum);
    EXPECT_EQ(d.extraCoreCount, d.totalHeads % d.usedCoreNum);

    const int64_t headCoreHeads = d.baseCoreHeads + (d.extraCoreCount > 0 ? 1 : 0);
    const int64_t tailCoreHeads = d.baseCoreHeads;
    EXPECT_GT(d.batch, 0);
    EXPECT_LE(d.batch, headCoreHeads); // batch 不超过单核最大 TN 数

    EXPECT_EQ(d.headCoreLoopCount, CeilDivRef(headCoreHeads, d.batch));
    EXPECT_EQ(d.headCoreHeadsPerLoop, d.headCoreLoopCount > 0 ? CeilDivRef(headCoreHeads, d.headCoreLoopCount) : 0);
    EXPECT_EQ(d.tailCoreLoopCount, CeilDivRef(tailCoreHeads, d.batch));
    EXPECT_EQ(d.tailCoreHeadsPerLoop, d.tailCoreLoopCount > 0 ? CeilDivRef(tailCoreHeads, d.tailCoreLoopCount) : 0);
    EXPECT_GE(d.reduceTmpSize, 0);
}

} // namespace

// ---------------- 注册 ----------------

TEST_F(ClaGateBackwardTiling, tiling_registered)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
}

// tiling_parse：正常上下文返回成功。
TEST_F(ClaGateBackwardTiling, tiling_parse_success)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(kOpType);
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);

    std::string compileInfoString = MakeCompileInfoString(kDefaultUbSize, kDefaultCoreNum);
    optiling::ClaGateBackwardCompileInfo compileInfo;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    auto holder = gert::KernelRunContextFaker()
                      .KernelIONum(1, 1)
                      .Inputs({const_cast<char*>(compileInfoString.c_str()), reinterpret_cast<void*>(&platformInfo)})
                      .Outputs({&compileInfo})
                      .Build();
    ASSERT_NE(holder.GetContext<gert::KernelContext>(), nullptr);
    EXPECT_EQ(opImpl->tiling_parse(holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

// ---------------- 正向：常规核间 / 核内切分 ----------------

// T=8, N=64, D=256：大规模 shape，用满 64 核，batch 被 coreHeadsMax 截断为 8。
TEST_F(ClaGateBackwardTiling, bf16_tnd_basic)
{
    ClaGateBackwardTilingData tiling{};
    ASSERT_EQ(RunTilingStandard(8, 64, 256, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.headNum, 64);
    EXPECT_EQ(tiling.headDim, 256);
    EXPECT_EQ(tiling.totalHeads, 8 * 64);
    EXPECT_EQ(tiling.usedCoreNum, 64);
    EXPECT_EQ(tiling.baseCoreHeads, 8);
    EXPECT_EQ(tiling.extraCoreCount, 0);
    EXPECT_EQ(tiling.batch, 8);
    EXPECT_EQ(tiling.headCoreLoopCount, 1);
    EXPECT_EQ(tiling.headCoreHeadsPerLoop, 8);
    EXPECT_EQ(tiling.tailCoreLoopCount, 1);
    EXPECT_EQ(tiling.tailCoreHeadsPerLoop, 8);
    ExpectTilingConsistent(tiling);
}

// T=4, N=128, D=128，FP16：核数被 coresByCopy 收缩到 32。
TEST_F(ClaGateBackwardTiling, fp16_head128)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.dtype0 = opt.dtype1 = opt.dtype2 = opt.dtype3 = opt.dtype4 = ge::DT_FLOAT16;
    ASSERT_EQ(RunTilingStandard(4, 128, 128, tiling, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.headNum, 128);
    EXPECT_EQ(tiling.headDim, 128);
    EXPECT_EQ(tiling.totalHeads, 4 * 128);
    EXPECT_EQ(tiling.usedCoreNum, 32);
    EXPECT_EQ(tiling.baseCoreHeads, 16);
    EXPECT_EQ(tiling.extraCoreCount, 0);
    EXPECT_EQ(tiling.batch, 16);
    ExpectTilingConsistent(tiling);
}

// grad_merged 仅支持 [T, N, D]：传 [T, N*D] 等价 view（rank=2）应被拦截。
TEST_F(ClaGateBackwardTiling, reject_grad_merged_2d_view)
{
    ClaGateBackwardTilingData tiling{};
    ASSERT_EQ(RunTiling(MakeShape({8, 16384}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64}),
                        MakeShape({8, 64}), tiling),
              ge::GRAPH_FAILED);
}

// gate logits 仅支持 [T, N] 二维：[T, N, 1] 应被拦截。
TEST_F(ClaGateBackwardTiling, gate_logits_3d_rejected)
{
    ClaGateBackwardTilingData tiling{};
    ASSERT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}),
                        MakeShape({8, 64, 1}), MakeShape({8, 64, 1}), tiling),
              ge::GRAPH_FAILED);
}

// input_attn_layout 缺省（未挂属性）时按 "TND" 处理。
TEST_F(ClaGateBackwardTiling, attrs_omitted_default_tnd)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.setAttrs = false;
    ASSERT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalHeads, 8 * 64);
    ExpectTilingConsistent(tiling);
}

// 小 shape：T=1, N=8, D=128，核数收缩到 1。
TEST_F(ClaGateBackwardTiling, small_shape_shrink_cores)
{
    ClaGateBackwardTilingData tiling{};
    ASSERT_EQ(RunTilingStandard(1, 8, 128, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalHeads, 8);
    EXPECT_EQ(tiling.usedCoreNum, 1);
    EXPECT_EQ(tiling.baseCoreHeads, 8);
    EXPECT_EQ(tiling.extraCoreCount, 0);
    EXPECT_EQ(tiling.batch, 8);
    ExpectTilingConsistent(tiling);
}

// 核数用满且有余数：T=257, N=4, D=128 -> usedCoreNum=64, base=16, extra=4（头核 17 / 尾核 16）。
TEST_F(ClaGateBackwardTiling, extra_core_count_d128)
{
    ClaGateBackwardTilingData tiling{};
    ASSERT_EQ(RunTilingStandard(257, 4, 128, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalHeads, 257 * 4);
    EXPECT_EQ(tiling.usedCoreNum, 64);
    EXPECT_EQ(tiling.baseCoreHeads, 16);
    EXPECT_EQ(tiling.extraCoreCount, 4);
    EXPECT_EQ(tiling.batch, 17);
    EXPECT_EQ(tiling.headCoreLoopCount, 1);
    EXPECT_EQ(tiling.headCoreHeadsPerLoop, 17);
    EXPECT_EQ(tiling.tailCoreLoopCount, 1);
    EXPECT_EQ(tiling.tailCoreHeadsPerLoop, 16);
    ExpectTilingConsistent(tiling);
}

// 头核 / 尾核循环数不同：缩小 UB 预算使 batch 落到 1，每核循环次数即单核 TN 数。
// T=257, N=4, D=256 -> usedCoreNum=64, base=16, extra=4（头核 17 / 尾核 16）。
TEST_F(ClaGateBackwardTiling, head_tail_core_loop_diff)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.ubSize = 12000; // D=256 时 perBatch=7192，budget=12000 只够 batch=1
    ASSERT_EQ(RunTilingStandard(257, 4, 256, tiling, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.totalHeads, 257 * 4);
    EXPECT_EQ(tiling.usedCoreNum, 64);
    EXPECT_EQ(tiling.baseCoreHeads, 16);
    EXPECT_EQ(tiling.extraCoreCount, 4);
    EXPECT_EQ(tiling.batch, 1);
    EXPECT_EQ(tiling.headCoreLoopCount, 17);
    EXPECT_EQ(tiling.headCoreHeadsPerLoop, 1);
    EXPECT_EQ(tiling.tailCoreLoopCount, 16);
    EXPECT_EQ(tiling.tailCoreHeadsPerLoop, 1);
    ExpectTilingConsistent(tiling);
}

// 极大 UB 预算：batch 不再被 UB 复核削减（仍受 coreHeadsMax 截断）。
TEST_F(ClaGateBackwardTiling, large_ub_budget)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.ubSize = 1048576;
    ASSERT_EQ(RunTilingStandard(128, 64, 256, tiling, opt), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling.usedCoreNum, 64);
    EXPECT_EQ(tiling.baseCoreHeads, 128);
    EXPECT_EQ(tiling.batch, 128);
    EXPECT_EQ(tiling.headCoreLoopCount, 1);
    EXPECT_EQ(tiling.headCoreHeadsPerLoop, 128);
    ExpectTilingConsistent(tiling);
}

// ---------------- 反向：dtype / 属性 ----------------

TEST_F(ClaGateBackwardTiling, reject_invalid_dtype)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.dtype0 = ge::DT_FLOAT;
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_dtype_mismatch_between_inputs)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.dtype1 = ge::DT_FLOAT16; // 与 grad_merged 的 BF16 不一致
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_unsupported_layout)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.layout = "BSN";
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_FAILED);
}

// ---------------- 反向：shape 校验 ----------------

TEST_F(ClaGateBackwardTiling, reject_global_attn_not_3d)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 16384}), MakeShape({8, 16384}), MakeShape({8, 16384}), MakeShape({8, 64}),
                        MakeShape({8, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_local_attn_not_3d)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 16384}), MakeShape({8, 64}),
                        MakeShape({8, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_attn_shape_mismatch)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64, 128}), MakeShape({8, 64}),
                        MakeShape({8, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_invalid_head_dim)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingStandard(8, 64, 128, tiling), ge::GRAPH_SUCCESS);
    EXPECT_EQ(RunTilingStandard(8, 64, 64, tiling), ge::GRAPH_FAILED); // 仅支持 D=128/256
}

TEST_F(ClaGateBackwardTiling, reject_dynamic_negative)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({-1, 64, 256}), MakeShape({-1, 64, 256}), MakeShape({-1, 64, 256}),
                        MakeShape({-1, 64}), MakeShape({-1, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_grad_rank_too_low)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({131072}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64}),
                        MakeShape({8, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_grad_rank_too_high)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({1, 1, 1, 1, 1, 1, 1, 64, 256}), MakeShape({1, 64, 256}), MakeShape({1, 64, 256}),
                        MakeShape({1, 64}), MakeShape({1, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_grad_elems_mismatch)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 128}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64}),
                        MakeShape({8, 64}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_logits_shape_mismatch_between_inputs)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64}),
                        MakeShape({8, 32}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_logits_rank_invalid)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({512}),
                        MakeShape({512}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_logits_3d_lastdim_invalid)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}),
                        MakeShape({8, 64, 2}), MakeShape({8, 64, 2}), tiling),
              ge::GRAPH_FAILED);
}

TEST_F(ClaGateBackwardTiling, reject_logits_tn_mismatch)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 64, 256}), MakeShape({8, 32}),
                        MakeShape({8, 32}), tiling),
              ge::GRAPH_FAILED);
}

// ---------------- 反向：空 Tensor / 平台 / UB 预算 ----------------

// 空 Tensor：totalHeads == 0 直接拦截（本算子不支持空 Tensor）。
TEST_F(ClaGateBackwardTiling, reject_empty_tensor)
{
    ClaGateBackwardTilingData tiling{};
    EXPECT_EQ(RunTiling(MakeShape({0, 64, 256}), MakeShape({0, 64, 256}), MakeShape({0, 64, 256}), MakeShape({0, 64}),
                        MakeShape({0, 64}), tiling),
              ge::GRAPH_FAILED);
}

// 核数 <= 0：GetPlatformInfo 失败。
TEST_F(ClaGateBackwardTiling, reject_core_num_zero)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.coreNum = 0;
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_FAILED);
}

// UB 减去预留后 <= 0 直接拦截。
TEST_F(ClaGateBackwardTiling, reject_ub_budget_not_positive)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.ubSize = 4096; // ubBudget = 4096 - 8192 < 0
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_FAILED);
}

// ubBudget > 0 但连 batch=1 都放不下，UB 复核后 batch <= 0，拦截。
TEST_F(ClaGateBackwardTiling, reject_batch_zero_after_solve)
{
    ClaGateBackwardTilingData tiling{};
    TilingOptions opt;
    opt.ubSize = 8000; // ubBudget > 0，初算 batch=1，但扣除对齐与 Reduce 临时空间后放不下 batch=1
    EXPECT_EQ(RunTilingStandard(8, 64, 256, tiling, opt), ge::GRAPH_FAILED);
}
