/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 */

/*!
 * \file test_apply_adam_with_amsgrad_v2_tiling.cpp
 * \brief Tiling UT for ApplyAdamWithAmsgradV2 operator (arch35 / ascend950).
 *
 * Schema:
 *   Inputs  (11): var, m, v, vhat, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad
 *     - var / m / v / vhat / grad: share var.shape, dtype fixed FP32 (op narrowed to FP32-only)
 *     - beta1_power / beta2_power / lr / beta1 / beta2 / epsilon: scalar tensors (shape == {1})
 *   Outputs (4): var_out, m_out, v_out, vhat_out
 *   Attrs   (0)
 *
 * TilingKey encoding (def 驱动 dtype: dtype 走构建系统注入的 DTYPE_VAR 宏，非 tiling key):
 *   host: ASCENDC_TPL_SEL_PARAM(context, APPLY_ADAM_WITH_AMSGRAD_V2_TPL_SCH_MODE_0)
 *   tiling_key 仅剩单值占位轴 schMode∈{0}，dtype 不再编码进 tiling key，故 host tiling key
 *   恒为 0（schMode=0）。op 已收窄为 fp32-only，def DataType 列表仅展开 1 个 kernel 变体
 *   （float），op_host 层不可见。
 *
 * Tiling internals (covers ApplyAdamWithAmsgradV2TilingFunc helpers):
 *   - GetShapeAttrsInfo: var/m/v/vhat/grad shape-size equality + dtype whitelist
 *   - Empty tensor fast path (totalNum==0): BlockDim=1, blockFactor=ubFactor=0
 *   - GetShapeAttrsInfo: m/v/vhat/grad dtype must equal var dtype (heterogeneous -> FAILED)
 *   - bytesPerElem: fp32=72; UB split + 32B alignment
 *
 * Test coverage (op narrowed to FP32-only; fp16/bf16 var dtype now rejected):
 *   T01  FP32 small shape (single-core retreat boundary)
 *   T02  FP32 medium shape (multi-core full split)
 *   T03  FP32 large shape (multi-core stress)
 *   T04  FP16 var medium shape -> FAILED (dtype whitelist rejects fp16)
 *   T05  FP16 var small shape  -> FAILED (dtype whitelist rejects fp16)
 *   T06  BF16 var medium shape -> FAILED (dtype whitelist rejects bf16)
 *   T07  BF16 var large shape  -> FAILED (dtype whitelist rejects bf16)
 *   T08  Empty tensor totalNum=0 -- FP32 (blockFactor=ubFactor=0)
 *   T09  BF16 var empty tensor -> FAILED (dtype check precedes empty fast path)
 *   T10  Scalar shape {1} (lowest non-empty bound)
 *   T11  m.shape mismatch -> FAILED
 *   T12  grad.shape mismatch -> FAILED
 *   T13  vhat.shape mismatch -> FAILED
 *   T14  v.shape mismatch -> FAILED
 *   T15  Rank-8 boundary shape (high-dim path) -> SUCCESS
 *   T16  Unsupported dtype (INT32) -> FAILED
 *   T17  Heterogeneous dtype (var=FP32, m=FP16) -> FAILED (m/v/vhat/grad vs var consistency)
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../../op_kernel/arch35/apply_adam_with_amsgrad_v2_tiling_data.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestApplyAdamWithAmsgradV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TestApplyAdamWithAmsgradV2Tiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestApplyAdamWithAmsgradV2Tiling TearDown" << std::endl; }
};

// TilingKey constant: def 驱动 dtype 后 tiling key 仅编码占位轴 schMode=0。
// op 已收窄为 fp32-only 单变体，host tiling key 恒为 0。
static constexpr uint64_t TILING_KEY_FP32 = 0; // schMode=0

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion)
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

struct ApplyAdamWithAmsgradV2UtCompileInfo {};

struct ApplyAdamWithAmsgradV2TilingResult {
    ge::graphStatus status;
    uint64_t tilingKey;
    int64_t totalNum;
    int64_t blockFactor;
    int64_t ubFactor;
};

// Build a TilingContext for ApplyAdamWithAmsgradV2 with the 11-input / 4-output
// schema and run the registered tiling function. Returns the resulting tiling
// status, tiling key and (when status == SUCCESS) the parsed TilingData fields.
//
// var/m/v/vhat/grad share tensorDtype; the 6 scalar inputs are always fp32
// (matches Op def: scalars locked to DT_FLOAT, ValueDepend(REQUIRED, TILING)).
static ApplyAdamWithAmsgradV2TilingResult DoApplyAdamWithAmsgradV2TilingCase(
    const std::initializer_list<int64_t>& varShape, const std::initializer_list<int64_t>& mShape,
    const std::initializer_list<int64_t>& vShape, const std::initializer_list<int64_t>& vhatShape,
    const std::initializer_list<int64_t>& gradShape, ge::DataType tensorDtype, ge::Format inputFormat)
{
    ApplyAdamWithAmsgradV2TilingResult result{ge::GRAPH_FAILED, 0, 0, 0, 0};

    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("ApplyAdamWithAmsgradV2");
    EXPECT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) == nullptr) {
        return result;
    }

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    EXPECT_NE(param, nullptr);
    if (param == nullptr) {
        return result;
    }

    std::initializer_list<int64_t> scalarShape = {1};
    gert::StorageShape varStorage = {varShape, varShape};
    gert::StorageShape mStorage = {mShape, mShape};
    gert::StorageShape vStorage = {vShape, vShape};
    gert::StorageShape vhatStorage = {vhatShape, vhatShape};
    gert::StorageShape scalarStorage = {scalarShape, scalarShape};
    gert::StorageShape gradStorage = {gradShape, gradShape};

    ApplyAdamWithAmsgradV2UtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&varStorage, &mStorage, &vStorage, &vhatStorage, &scalarStorage, &scalarStorage,
                                    &scalarStorage, &scalarStorage, &scalarStorage, &scalarStorage, &gradStorage})
                      .OutputShapes({&varStorage, &mStorage, &vStorage, &vhatStorage})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, tensorDtype, inputFormat, inputFormat)  // var
                      .NodeInputTd(1, tensorDtype, inputFormat, inputFormat)  // m
                      .NodeInputTd(2, tensorDtype, inputFormat, inputFormat)  // v
                      .NodeInputTd(3, tensorDtype, inputFormat, inputFormat)  // vhat
                      .NodeInputTd(4, ge::DT_FLOAT, inputFormat, inputFormat) // beta1_power
                      .NodeInputTd(5, ge::DT_FLOAT, inputFormat, inputFormat) // beta2_power
                      .NodeInputTd(6, ge::DT_FLOAT, inputFormat, inputFormat) // lr
                      .NodeInputTd(7, ge::DT_FLOAT, inputFormat, inputFormat) // beta1
                      .NodeInputTd(8, ge::DT_FLOAT, inputFormat, inputFormat) // beta2
                      .NodeInputTd(9, ge::DT_FLOAT, inputFormat, inputFormat) // epsilon
                      .NodeInputTd(10, tensorDtype, inputFormat, inputFormat) // grad
                      .NodeOutputTd(0, tensorDtype, inputFormat, inputFormat) // var_out
                      .NodeOutputTd(1, tensorDtype, inputFormat, inputFormat) // m_out
                      .NodeOutputTd(2, tensorDtype, inputFormat, inputFormat) // v_out
                      .NodeOutputTd(3, tensorDtype, inputFormat, inputFormat) // vhat_out
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    result.status = tiling_func(tiling_context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.tilingKey = tiling_context->GetTilingKey();
        auto rawTilingData = tiling_context->GetRawTilingData();
        if (rawTilingData != nullptr && rawTilingData->GetDataSize() >= sizeof(ApplyAdamWithAmsgradV2TilingData)) {
            const auto* td = reinterpret_cast<const ApplyAdamWithAmsgradV2TilingData*>(rawTilingData->GetData());
            result.totalNum = td->totalNum;
            result.blockFactor = td->blockFactor;
            result.ubFactor = td->ubFactor;
        }
    }
    return result;
}

// Convenience wrapper: build a "valid" context where var/m/v/vhat/grad share the
// same shape and the 6 scalars are shape {1}. Returns the tiling result.
static ApplyAdamWithAmsgradV2TilingResult DoApplyAdamWithAmsgradV2ValidCase(
    const std::initializer_list<int64_t>& tensorShape, ge::DataType tensorDtype)
{
    return DoApplyAdamWithAmsgradV2TilingCase(tensorShape, tensorShape, tensorShape, tensorShape, tensorShape,
                                              tensorDtype, ge::FORMAT_ND);
}

// Heterogeneous-dtype builder: var/m/v/vhat/grad share tensorShape but each may
// carry an independent dtype. Used to exercise the m/v/vhat/grad-vs-var dtype
// consistency check in GetShapeAttrsInfo (heterogeneous dtype must FAIL).
static ge::graphStatus DoApplyAdamWithAmsgradV2HeteroDtypeCase(const std::initializer_list<int64_t>& tensorShape,
                                                               ge::DataType varDtype, ge::DataType mDtype,
                                                               ge::DataType vDtype, ge::DataType vhatDtype,
                                                               ge::DataType gradDtype)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("ApplyAdamWithAmsgradV2");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = opImpl->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    EXPECT_NE(param, nullptr);
    if (param == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::initializer_list<int64_t> scalarShape = {1};
    gert::StorageShape tensorStorage = {tensorShape, tensorShape};
    gert::StorageShape scalarStorage = {scalarShape, scalarShape};

    ApplyAdamWithAmsgradV2UtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(11, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&tensorStorage, &tensorStorage, &tensorStorage, &tensorStorage, &scalarStorage,
                                    &scalarStorage, &scalarStorage, &scalarStorage, &scalarStorage, &scalarStorage,
                                    &tensorStorage})
                      .OutputShapes({&tensorStorage, &tensorStorage, &tensorStorage, &tensorStorage})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)     // var
                      .NodeInputTd(1, mDtype, ge::FORMAT_ND, ge::FORMAT_ND)       // m
                      .NodeInputTd(2, vDtype, ge::FORMAT_ND, ge::FORMAT_ND)       // v
                      .NodeInputTd(3, vhatDtype, ge::FORMAT_ND, ge::FORMAT_ND)    // vhat
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // beta1_power
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // beta2_power
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // lr
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // beta1
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // beta2
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // epsilon
                      .NodeInputTd(10, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)   // grad
                      .NodeOutputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)    // var_out
                      .NodeOutputTd(1, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)    // m_out
                      .NodeOutputTd(2, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)    // v_out
                      .NodeOutputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)    // vhat_out
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    return tiling_func(tiling_context);
}

// =====================================================================
// T01 FP32 small shape (totalNum <= ubFactor boundary)
// Covers ComputeBlockAndUbSplit single-core fallback path (blockFactor
// stays 32B aligned).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_fp32_small)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({256}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, TILING_KEY_FP32);
    EXPECT_EQ(result.totalNum, 256);
    // blockFactor must be 32B aligned (=8 elements for fp32)
    EXPECT_GT(result.blockFactor, 0);
    EXPECT_GE(result.ubFactor, 8);
}

// =====================================================================
// T02 FP32 medium shape (multi-core full split)
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_fp32_medium)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({64, 1024}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, TILING_KEY_FP32);
    EXPECT_EQ(result.totalNum, 64L * 1024);
    EXPECT_GT(result.blockFactor, 0);
    EXPECT_GT(result.ubFactor, 0);
}

// =====================================================================
// T03 FP32 large shape (multi-core stress)
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_fp32_large)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({1024, 1024}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, TILING_KEY_FP32);
    EXPECT_EQ(result.totalNum, 1024L * 1024);
}

// =====================================================================
// T04 FP16 var medium shape -> FAILED (op narrowed to FP32-only; dtype
// whitelist in GetShapeAttrsInfo now rejects fp16).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_fp16_rejected)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({64, 1024}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T05 FP16 var small shape -> FAILED (dtype whitelist rejects fp16).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_fp16_small_rejected)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({128}, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T06 BF16 var medium shape -> FAILED (dtype whitelist rejects bf16).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_bf16_rejected)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({64, 1024}, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T07 BF16 var large shape -> FAILED (dtype whitelist rejects bf16).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_bf16_large_rejected)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({1024, 1024}, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T08 Empty Tensor (totalNum == 0) FP32 path.
// Covers empty fast path: BlockDim=1, blockFactor=ubFactor=0. TilingKey
// still set per dtype.
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_empty_tensor_fp32)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({0}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, TILING_KEY_FP32);
    EXPECT_EQ(result.totalNum, 0);
    EXPECT_EQ(result.blockFactor, 0);
    EXPECT_EQ(result.ubFactor, 0);
}

// =====================================================================
// T09 BF16 var empty tensor -> FAILED. The dtype whitelist check in
// GetShapeAttrsInfo runs before the empty (totalNum==0) fast path, so an
// unsupported var dtype is rejected even for an empty tensor.
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_bf16_empty_rejected)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({0}, ge::DT_BF16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T10 Scalar shape {1} (lowest non-empty bound, exercise EnsureNotScalar
// and small-shape fallback).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_scalar_shape_1)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({1}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, TILING_KEY_FP32);
    EXPECT_EQ(result.totalNum, 1);
    EXPECT_GE(result.blockFactor, 1);
    EXPECT_GE(result.ubFactor, 1);
}

// =====================================================================
// T11 Shape mismatch: m.shape != var.shape -> FAILED
// Covers GetShapeAttrsInfo shape-size equality check.
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_m_shape_mismatch)
{
    auto result = DoApplyAdamWithAmsgradV2TilingCase(
        /*var=*/{2048}, /*m=*/{1024}, /*v=*/{2048}, /*vhat=*/{2048}, /*grad=*/{2048}, ge::DT_FLOAT, ge::FORMAT_ND);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T12 Shape mismatch: grad.shape != var.shape -> FAILED
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_grad_shape_mismatch)
{
    auto result = DoApplyAdamWithAmsgradV2TilingCase(
        /*var=*/{2048}, /*m=*/{2048}, /*v=*/{2048}, /*vhat=*/{2048}, /*grad=*/{1024}, ge::DT_FLOAT, ge::FORMAT_ND);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T13 Shape mismatch: vhat.shape != var.shape -> FAILED
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_vhat_shape_mismatch)
{
    auto result = DoApplyAdamWithAmsgradV2TilingCase(
        /*var=*/{2048}, /*m=*/{2048}, /*v=*/{2048}, /*vhat=*/{1024}, /*grad=*/{2048}, ge::DT_FLOAT, ge::FORMAT_ND);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T14 Shape mismatch: v.shape != var.shape -> FAILED
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_v_shape_mismatch)
{
    auto result = DoApplyAdamWithAmsgradV2TilingCase(
        /*var=*/{2048}, /*m=*/{2048}, /*v=*/{1024}, /*vhat=*/{2048}, /*grad=*/{2048}, ge::DT_FLOAT, ge::FORMAT_ND);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T15 8D shape (rank == 8 boundary, README declares rank in [1, 8])
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_rank_8_boundary)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_FLOAT);
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, TILING_KEY_FP32);
    EXPECT_EQ(result.totalNum, 256); // 2^8
}

// =====================================================================
// T16 Unsupported dtype (DT_INT32) -> FAILED.
// Covers the dtype whitelist check in GetShapeAttrsInfo
// (only float32 supported after FP32-only narrowing).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_unsupported_dtype_int32)
{
    auto result = DoApplyAdamWithAmsgradV2ValidCase({2048}, ge::DT_INT32);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

// =====================================================================
// T17 Heterogeneous dtype: var=FP32 but m=FP16 -> FAILED.
// Covers the m/v/vhat/grad-vs-var dtype consistency check in
// GetShapeAttrsInfo (kernel is single DTYPE_X template; heterogeneous
// dtype would mis-read bytes at wrong word width).
// =====================================================================
TEST_F(TestApplyAdamWithAmsgradV2Tiling, apply_adam_amsgrad_v2_hetero_dtype_m_mismatch)
{
    auto status = DoApplyAdamWithAmsgradV2HeteroDtypeCase({2048}, /*var=*/ge::DT_FLOAT, /*m=*/ge::DT_FLOAT16,
                                                          /*v=*/ge::DT_FLOAT, /*vhat=*/ge::DT_FLOAT,
                                                          /*grad=*/ge::DT_FLOAT);
    EXPECT_EQ(status, ge::GRAPH_FAILED);
}
