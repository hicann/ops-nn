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
 * \file gru_grad_tiling.cpp
 * \brief
 */

#include "gru_grad_tiling.h"
#include <iostream>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {

const int64_t GATES_NUM = 3;
const int64_t AIV_DOUBLE = 2;
const int64_t FP32_BYTES = 4;
const int64_t INPUT_DIM_NUM = 3;
const int64_t DEFAULT_UB_RESERVE_SIZE = 1024;
const int64_t DEFAULT_ALIGNED_FP16 = 16;
const int64_t DEFAULT_ALIGNED_FP32 = 8;
const int64_t DEFAULT_BUFFER_SPACE = -1;
const int64_t DEFAULT_SPLIT_FACTOR = 4;
const int64_t DEFAULT_REDUCE_N_LIMIT = 128;
const int64_t DEFAULT_COPY_FACTOR_FP32 = 4;
const int64_t DEFAULT_COPY_FACTOR_FP16 = 6;
const int64_t DEFAULT_ELEMENTS_PER_PART = 4096;

const int64_t INPUT_X = 0, INPUT_WI = 1, INPUT_WH = 2, INPUT_H0 = 3;
const int64_t INPUT_H = 4, INPUT_R = 5, INPUT_Z = 6, INPUT_N = 7, INPUT_HN = 8, INPUT_DY = 9, INPUT_DH = 10,
              INPUT_BS = 11;
const int64_t OUT_DX = 0, OUT_DHP = 1, OUT_DWI = 2, OUT_DWH = 3, OUT_DBI = 4, OUT_DBH = 5;
const int64_t T_IDX = 0, B_IDX = 1, S_IDX = 2;

const std::vector<std::string> SUPPORT_DIRECTION = {"UNIDIRECTIONAL", "REDIRECTIONAL"};

bool GruGradTiling::IsCapable() { return true; }

ge::graphStatus GruGradTiling::GetPlatformInfo()
{
    OP_LOGD(nodeName_, "GruGradTiling GetPlatformInfo.");
    compileInfo_ = static_cast<const GruGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);

    sysAicCoreNum_ = static_cast<int64_t>(compileInfo_->aicCoreNum);
    sysAivCoreNum_ = static_cast<int64_t>(compileInfo_->aivCoreNum);
    tilingData_.ubSize = compileInfo_->ubSizePlatForm;

    auto dataType = context_->GetInputDesc(INPUT_X)->GetDataType();
    inputDSize_ = dataType == ge::DT_FLOAT ? FP32_BYTES : FP32_BYTES / AIV_DOUBLE;
    alignedPara_ = dataType == ge::DT_FLOAT ? DEFAULT_ALIGNED_FP32 : DEFAULT_ALIGNED_FP16;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTiling::GetShapeAttrsInfo()
{
    nodeName_ = context_->GetNodeName();
    auto xIn = context_->GetInputShape(INPUT_X);
    OP_CHECK_IF(xIn == nullptr, OP_LOGE(nodeName_, "x nullptr"), return ge::GRAPH_FAILED);
    auto xS = xIn->GetStorageShape();
    OP_CHECK_IF(xS.GetDimNum() != INPUT_DIM_NUM, OP_LOGE(nodeName_, "x must 3D"), return ge::GRAPH_FAILED);

    auto h0In = context_->GetInputShape(INPUT_H0);
    OP_CHECK_IF(h0In == nullptr, OP_LOGE(nodeName_, "h0 nullptr"), return ge::GRAPH_FAILED);
    auto h0S = h0In->GetStorageShape();

    auto wiIn = context_->GetInputShape(INPUT_WI);
    OP_CHECK_IF(wiIn == nullptr, OP_LOGE(nodeName_, "wi nullptr"), return ge::GRAPH_FAILED);

    auto whIn = context_->GetInputShape(INPUT_WH);
    OP_CHECK_IF(whIn == nullptr, OP_LOGE(nodeName_, "wh nullptr"), return ge::GRAPH_FAILED);

    tilingData_.timeStep = xS.GetDim(T_IDX);
    tilingData_.batch = xS.GetDim(B_IDX);
    tilingData_.inputSize = xS.GetDim(S_IDX);
    tilingData_.hiddenSize = h0S.GetDim(S_IDX);

    OP_TILING_CHECK(tilingData_.inputSize <= 0 || tilingData_.hiddenSize <= 0 || tilingData_.timeStep <= 0 ||
                        tilingData_.batch <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "shape<=0"), return ge::GRAPH_FAILED);

    auto bsS = context_->GetOptionalInputShape(INPUT_BS);
    auto bsD = context_->GetOptionalInputDesc(INPUT_BS);
    tilingData_.isSeqLength = (bsD && bsS && bsS->GetStorageShape().GetDimNum() != 0) ? 1 : 0;

    OP_TILING_CHECK(!CheckParamsShape(), VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "shape fail"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckAttr(), VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "attr fail"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTiling::DoOpTiling()
{
    context_->SetScheduleMode(1);

    int64_t elementsPerPart = DEFAULT_ELEMENTS_PER_PART;

    int64_t D_aligned = (tilingData_.hiddenSize + alignedPara_ - 1) / alignedPara_ * alignedPara_;

    int64_t B_per_core = (tilingData_.batch + sysAivCoreNum_ - 1) / sysAivCoreNum_;

    int64_t B_tile_max = elementsPerPart / D_aligned;
    int64_t B_tile = 0;
    int64_t B_tile_tail = 0;
    int64_t B_tile_cnt = 0;
    int64_t D_tile = 0;
    int64_t D_tile_tail = 0;
    int64_t D_tile_cnt = 0;

    if (B_tile_max >= 1) {
        B_tile = B_tile_max < B_per_core ? B_tile_max : B_per_core;
        B_tile_cnt = (B_per_core + B_tile - 1) / B_tile;
        B_tile_tail = B_per_core - (B_tile_cnt - 1) * B_tile;
        D_tile = D_aligned;
        D_tile_cnt = 1;
        D_tile_tail = D_aligned;
    } else {
        B_tile = 1;
        B_tile_cnt = B_per_core;
        B_tile_tail = 1;
        D_tile = (elementsPerPart / alignedPara_) * alignedPara_;
        D_tile_cnt = (D_aligned + D_tile - 1) / D_tile;
        D_tile_tail = D_aligned - (D_tile_cnt - 1) * D_tile;
    }

    tilingData_.singleCoreM = B_tile;
    tilingData_.singleCoreMTail = B_tile_tail;
    tilingData_.singleCoreN = D_tile;
    tilingData_.singleCoreNTail = D_tile_tail;
    tilingData_.baseN = elementsPerPart;
    tilingData_.baseM = 0;
    tilingData_.mCnt = B_tile_cnt;
    tilingData_.nCnt = D_tile_cnt;

    GetMatmulTiling();
    ReduceBlockCalculate();
    SplitDxhBlockCalculate();
    ConcatXhBlockCalculate();

    tilingData_.inputSizeAligned = (tilingData_.inputSize + alignedPara_ - 1) / alignedPara_ * alignedPara_;
    tilingData_.hiddenSizeAligned = D_aligned;
    tilingData_.oneLineAligned = tilingData_.inputSizeAligned + tilingData_.hiddenSizeAligned;

    SetTilingData();
    context_->SetBlockDim(sysAicCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }
ge::graphStatus GruGradTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus GruGradTiling::GetWorkspaceSize()
{
    int64_t TB = tilingData_.timeStep * tilingData_.batch;
    int64_t H = tilingData_.hiddenSize;
    int64_t r1 = TB * GATES_NUM * H;
    int64_t r2 = TB * GATES_NUM * H;
    int64_t r3 = TB * H;
    int64_t r4 = TB * tilingData_.inputSize;
    int64_t r5 = tilingData_.batch * H;
    int64_t r6 = tilingData_.batch * H;
    int64_t ws1 = (r1 + r2 + r3 + r4 + r5 + r6) * inputDSize_;

    size_t* ws = context_->GetWorkspaceSizes(1);
    ws[0] = ws1 + compileInfo_->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

bool GruGradTiling::ValidateInputShape(int idx, const std::vector<int64_t>& e)
{
    auto in = context_->GetInputShape(idx);
    if (!in)
        return false;
    auto s = in->GetStorageShape();
    if (s.GetDimNum() != (int64_t)e.size())
        return false;
    for (size_t i = 0; i < e.size(); i++)
        if (e[i] != s.GetDim(i))
            return false;
    return true;
}

bool GruGradTiling::ValidateOutputShape(int idx, const std::vector<int64_t>& e)
{
    auto out = context_->GetOutputShape(idx);
    if (!out)
        return false;
    auto s = out->GetStorageShape();
    if (s.GetDimNum() != (int64_t)e.size())
        return false;
    for (size_t i = 0; i < e.size(); i++)
        if (e[i] != s.GetDim(i))
            return false;
    return true;
}

bool GruGradTiling::CheckParamsShape()
{
    std::vector<int64_t> inD = {tilingData_.timeStep, tilingData_.batch, tilingData_.inputSize};
    std::vector<int64_t> ihD = {1, tilingData_.batch, tilingData_.hiddenSize};
    std::vector<int64_t> hidD = {tilingData_.timeStep, tilingData_.batch, tilingData_.hiddenSize};
    std::vector<int64_t> wiD = {GATES_NUM * tilingData_.hiddenSize, tilingData_.inputSize};
    std::vector<int64_t> whD = {GATES_NUM * tilingData_.hiddenSize, tilingData_.hiddenSize};
    std::vector<int64_t> biD = {GATES_NUM * tilingData_.hiddenSize};
    std::vector<int64_t> bhD = {GATES_NUM * tilingData_.hiddenSize};

    OP_LOGI(nodeName_, "CheckParamsShape: T=%ld B=%ld I=%ld H=%ld", tilingData_.timeStep, tilingData_.batch,
            tilingData_.inputSize, tilingData_.hiddenSize);

    bool ret = true;
    auto ci = [&](int i, auto& e, const char* n) {
        bool ok = ValidateInputShape(i, e);
        auto in = context_->GetInputShape(i);
        if (in) {
            auto s = in->GetStorageShape();
            std::string actual;
            for (size_t d = 0; d < s.GetDimNum(); d++)
                actual += std::to_string(s.GetDim(d)) + (d + 1 < s.GetDimNum() ? "," : "");
            std::string exp;
            for (size_t d = 0; d < e.size(); d++)
                exp += std::to_string(e[d]) + (d + 1 < e.size() ? "," : "");
            OP_LOGI(nodeName_, "CheckInput[%d] %s expected=[%s] actual=[%s] %s", i, n, exp.c_str(), actual.c_str(),
                    ok ? "PASS" : "FAIL");
        } else {
            OP_LOGI(nodeName_, "CheckInput[%d] %s shape is nullptr, FAIL", i, n);
        }
        return ok;
    };
    auto co = [&](int i, auto& e, const char* n) {
        bool ok = ValidateOutputShape(i, e);
        auto out = context_->GetOutputShape(i);
        if (out) {
            auto s = out->GetStorageShape();
            std::string actual;
            for (size_t d = 0; d < s.GetDimNum(); d++)
                actual += std::to_string(s.GetDim(d)) + (d + 1 < s.GetDimNum() ? "," : "");
            std::string exp;
            for (size_t d = 0; d < e.size(); d++)
                exp += std::to_string(e[d]) + (d + 1 < e.size() ? "," : "");
            OP_LOGI(nodeName_, "CheckOutput[%d] %s expected=[%s] actual=[%s] %s", i, n, exp.c_str(), actual.c_str(),
                    ok ? "PASS" : "FAIL");
        } else {
            OP_LOGI(nodeName_, "CheckOutput[%d] %s shape is nullptr, FAIL", i, n);
        }
        return ok;
    };

    ret = ci(INPUT_WI, wiD, "w_input") && ret;
    ret = ci(INPUT_WH, whD, "w_hidden") && ret;
    ret = ci(INPUT_H0, ihD, "init_h") && ret;
    ret = ci(INPUT_H, hidD, "h") && ret;
    ret = ci(INPUT_R, hidD, "r") && ret;
    ret = ci(INPUT_Z, hidD, "z") && ret;
    ret = ci(INPUT_N, hidD, "n") && ret;
    ret = ci(INPUT_HN, hidD, "hn") && ret;
    ret = ci(INPUT_DY, hidD, "dy") && ret;
    ret = ci(INPUT_DH, ihD, "dh") && ret;
    ret = co(OUT_DX, inD, "dx") && ret;
    ret = co(OUT_DHP, ihD, "dh_prev") && ret;
    ret = co(OUT_DWI, wiD, "dw_input") && ret;
    ret = co(OUT_DWH, whD, "dw_hidden") && ret;
    ret = co(OUT_DBI, biD, "db_input") && ret;
    ret = co(OUT_DBH, bhD, "db_hidden") && ret;
    if (tilingData_.isSeqLength) {
        std::vector<int64_t> bsD = {tilingData_.timeStep};
        ret = ci(INPUT_BS, bsD, "batch_sz") && ret;
    }
    return ret;
}

bool GruGradTiling::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* dir = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(std::find(SUPPORT_DIRECTION.begin(), SUPPORT_DIRECTION.end(), dir) == SUPPORT_DIRECTION.end(),
                OP_LOGE(nodeName_, "direction not supported"), return false);
    tilingData_.direction = (std::string(dir) == "REDIRECTIONAL") ? 1 : 0;

    const bool* hasBiasAttr = attrs->GetAttrPointer<bool>(1);
    tilingData_.isBias = (hasBiasAttr != nullptr && *hasBiasAttr) ? 1 : 0;
    return true;
}

void GruGradTiling::GetMatmulTiling()
{
    auto geDataType = context_->GetInputDesc(INPUT_X)->GetDataType();
    auto mmDataType = static_cast<matmul_tiling::DataType>(geDataType);

    // ========== MM1: dgateMM — 计算 grad_h_prev = d_gh × w_hh ==========
    // A: d_gh      [B, 3H]
    // B: w_hh      [3H, H]
    // C: grad_h_prev [B, H]
    {
        matmul_tiling::MultiCoreMatmulTiling dgateMM;
        auto ret = dgateMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetAType fail."), return);
        ret = dgateMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetBType fail."), return);
        ret = dgateMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetCType fail."), return);

        ret = dgateMM.SetDim(sysAicCoreNum_);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetDim fail."), return);
        // M=B, N=H, K=3H
        ret = dgateMM.SetOrgShape(tilingData_.batch, tilingData_.hiddenSize, tilingData_.hiddenSize * GATES_NUM);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetOrgShape fail."), return);
        ret = dgateMM.SetShape(tilingData_.batch, tilingData_.hiddenSize, tilingData_.hiddenSize * GATES_NUM);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetShape fail."), return);
        ret = dgateMM.SetBufferSpace(DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetBufferSpace fail."), return);

        ret = dgateMM.GetTiling(tilingData_.dgateMMParam);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM GetTiling fail."), return);
    }

    // ========== MM2a: dwIhMM — dw_ih = d_gi^T × x  [3H,T*B]×[T*B, I]=[3H,I] ==========
    {
        matmul_tiling::MultiCoreMatmulTiling dwIhMM;
        auto ret = dwIhMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, true);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetAType fail."), return);
        ret = dwIhMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetBType fail."), return);
        ret = dwIhMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetCType fail."), return);
        ret = dwIhMM.SetDim(sysAicCoreNum_);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetDim fail."), return);
        ret = dwIhMM.SetOrgShape(tilingData_.hiddenSize * GATES_NUM, tilingData_.inputSize,
                                 tilingData_.batch * tilingData_.timeStep); // M=I, N=3H, K=T*B (x^T @ d_gi)
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetOrgShape fail."), return);
        ret = dwIhMM.SetShape(tilingData_.hiddenSize * GATES_NUM, tilingData_.inputSize,
                              tilingData_.batch * tilingData_.timeStep);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetShape fail."), return);
        ret = dwIhMM.SetBufferSpace(DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetBufferSpace fail."), return);
        ret = dwIhMM.GetTiling(tilingData_.dwIhMMParam);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM GetTiling fail."), return);
    }

    // ========== MM2b: dwHhMM — dw_hh = h^T × d_gh  [H,T*B]×[T*B,3H]=[H,3H] ==========
    {
        matmul_tiling::MultiCoreMatmulTiling dwHhMM;
        auto ret = dwHhMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, true);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetAType fail."), return);
        ret = dwHhMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetBType fail."), return);
        ret = dwHhMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetCType fail."), return);
        ret = dwHhMM.SetDim(sysAicCoreNum_);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetDim fail."), return);
        ret = dwHhMM.SetOrgShape(tilingData_.hiddenSize * GATES_NUM, tilingData_.hiddenSize,
                                 tilingData_.batch * tilingData_.timeStep);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetOrgShape fail."), return);
        ret = dwHhMM.SetShape(tilingData_.hiddenSize * GATES_NUM, tilingData_.hiddenSize,
                              tilingData_.batch * tilingData_.timeStep);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetShape fail."), return);
        ret = dwHhMM.SetBufferSpace(DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetBufferSpace fail."), return);
        ret = dwHhMM.GetTiling(tilingData_.dwHhMMParam);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM GetTiling fail."), return);
    }

    // ========== MM3: dxMM — dx = d_gi × w_input ==========
    // A: d_gi_all [T*B, 3H]
    // B: w_input  [3H, I]
    // C: dx       [T*B, I]
    {
        matmul_tiling::MultiCoreMatmulTiling dxMM;
        auto ret = dxMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetAType fail."), return);
        ret = dxMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetBType fail."), return);
        ret = dxMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetCType fail."), return);

        ret = dxMM.SetDim(sysAicCoreNum_);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetDim fail."), return);
        // M=T*B, N=I, K=3H
        ret = dxMM.SetOrgShape(tilingData_.batch * tilingData_.timeStep, tilingData_.inputSize,
                               tilingData_.hiddenSize * GATES_NUM);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetOrgShape fail."), return);
        ret = dxMM.SetShape(tilingData_.batch * tilingData_.timeStep, tilingData_.inputSize,
                            tilingData_.hiddenSize * GATES_NUM);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetShape fail."), return);
        ret = dxMM.SetBufferSpace(DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE, DEFAULT_BUFFER_SPACE);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetBufferSpace fail."), return);

        ret = dxMM.GetTiling(tilingData_.dxMMParam);
        OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM GetTiling fail."), return);
    }
}

void GruGradTiling::ReduceBlockCalculate()
{
    tilingData_.baseReduceN = Ops::Base::CeilDiv(Ops::Base::CeilDiv(tilingData_.hiddenSize * GATES_NUM, sysAivCoreNum_),
                                                 alignedPara_) *
                              alignedPara_;
    tilingData_.baseReduceN = tilingData_.baseReduceN > DEFAULT_REDUCE_N_LIMIT ? DEFAULT_REDUCE_N_LIMIT :
                                                                                 tilingData_.baseReduceN;
    tilingData_.maxReduceNumOnce = (tilingData_.ubSize - DEFAULT_UB_RESERVE_SIZE) / FP32_BYTES /
                                   tilingData_.baseReduceN;
    tilingData_.reduceBlockSize = tilingData_.timeStep * tilingData_.batch;
    int64_t bn = Ops::Base::CeilDiv(tilingData_.hiddenSize * GATES_NUM, tilingData_.baseReduceN);
    tilingData_.nReduceCnt = sysAivCoreNum_ < bn ? sysAivCoreNum_ : bn;
    int64_t pc = Ops::Base::CeilDiv(bn, tilingData_.nReduceCnt);
    tilingData_.nReduceCnt = Ops::Base::CeilDiv(bn, pc);
    tilingData_.singleCoreReduceN = pc * tilingData_.baseReduceN;
    tilingData_.singleCoreReduceNTail = tilingData_.hiddenSize * GATES_NUM -
                                        (tilingData_.nReduceCnt - 1) * tilingData_.singleCoreReduceN;
}

CutBatchTiling GruGradTiling::CalculateCutBatchTiling(int64_t ub, int64_t al, int64_t act, int64_t ml, int64_t b)
{
    CutBatchTiling r;
    r.copyMLines = ub > al ? ub / al : 1;
    r.copyMLines = r.copyMLines < ml ? r.copyMLines : ml;
    r.taskNum = Ops::Base::CeilDiv(b, r.copyMLines);
    r.copyMLinesTail = b - (r.taskNum - 1) * r.copyMLines;
    r.copyNLength = Ops::Base::CeilAlign(Ops::Base::CeilDiv(ub, r.copyMLines), al);
    r.nLoop = Ops::Base::CeilDiv(act, r.copyNLength);
    r.copyNLengthTail = act - (r.nLoop - 1) * r.copyNLength;
    r.splitTaskPerCore = r.taskNum / sysAivCoreNum_;
    r.splitPreCore = r.taskNum % sysAivCoreNum_;
    return r;
}

void GruGradTiling::SplitDxhBlockCalculate()
{
    int64_t f = DEFAULT_SPLIT_FACTOR;
    int64_t ub = Ops::Base::CeilAlign((tilingData_.ubSize - DEFAULT_UB_RESERVE_SIZE) / f, alignedPara_);
    tilingData_.inputSizeAligned = Ops::Base::CeilAlign(tilingData_.inputSize, alignedPara_);
    tilingData_.hiddenSizeAligned = Ops::Base::CeilAlign(tilingData_.hiddenSize, alignedPara_);
    tilingData_.oneLineAligned = tilingData_.inputSizeAligned + tilingData_.hiddenSizeAligned;
    int64_t ml = Ops::Base::CeilDiv(tilingData_.batch, sysAivCoreNum_);
    if (ub > tilingData_.oneLineAligned && tilingData_.isSeqLength == 0) {
        dxhInputParam_ = dxhHiddenParam_ = CalculateCutBatchTiling(
            ub, tilingData_.oneLineAligned, tilingData_.inputSize + tilingData_.hiddenSize, ml, tilingData_.batch);
    } else if (tilingData_.isSeqLength == 0) {
        f = inputDSize_ == FP32_BYTES ? DEFAULT_COPY_FACTOR_FP32 : DEFAULT_COPY_FACTOR_FP16;
        ub = Ops::Base::CeilAlign((tilingData_.ubSize - DEFAULT_UB_RESERVE_SIZE) / f, alignedPara_);
        dxhInputParam_ = CalculateCutBatchTiling(ub, tilingData_.inputSizeAligned, tilingData_.inputSize, ml,
                                                 tilingData_.batch);
        dxhHiddenParam_ = CalculateCutBatchTiling(ub, tilingData_.hiddenSizeAligned, tilingData_.hiddenSize, ml,
                                                  tilingData_.batch);
    }
}

void GruGradTiling::ConcatXhBlockCalculate()
{
    int64_t f = inputDSize_ == FP32_BYTES ? DEFAULT_COPY_FACTOR_FP32 : DEFAULT_COPY_FACTOR_FP16;
    int64_t ub = Ops::Base::CeilDiv((tilingData_.ubSize - DEFAULT_UB_RESERVE_SIZE) / f, alignedPara_) * alignedPara_;
    int64_t ml = Ops::Base::CeilDiv(tilingData_.batch, sysAivCoreNum_);
    if (ub > tilingData_.oneLineAligned) {
        xhInputParam_ = xhHiddenParam_ = CalculateCutBatchTiling(
            ub, tilingData_.oneLineAligned, tilingData_.inputSize + tilingData_.hiddenSize, ml, tilingData_.batch);
    } else {
        xhHiddenParam_ = CalculateCutBatchTiling(ub, tilingData_.hiddenSizeAligned, tilingData_.hiddenSize, ml,
                                                 tilingData_.batch);
        int64_t mli = Ops::Base::CeilDiv(tilingData_.batch * tilingData_.timeStep, sysAivCoreNum_);
        xhInputParam_ = CalculateCutBatchTiling(ub, tilingData_.inputSizeAligned, tilingData_.inputSize, mli,
                                                tilingData_.batch * tilingData_.timeStep);
    }
}

void GruGradTiling::SetTilingData()
{
    tilingData_.dxhInputTiling = dxhInputParam_;
    tilingData_.dxhHiddenTiling = dxhHiddenParam_;
    tilingData_.xhInputTiling = xhInputParam_;
    tilingData_.xhHiddenTiling = xhHiddenParam_;

    auto* buf = context_->GetTilingData<GruGradTilingData>();
    *buf = tilingData_;
    context_->GetRawTilingData()->SetDataSize(sizeof(GruGradTilingData));
}

static ge::graphStatus TilingFunc4GruGrad(gert::TilingContext* ctx)
{
    GruGradTiling obj(ctx);
    auto ret = obj.DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(ctx->GetNodeName(), "GruGradTiling failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForGruGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "TilingPrepareForGruGrad start.");
    auto ci = context->GetCompiledInfo<GruGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    fe::PlatFormInfos* pi = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, pi);
    auto plat = platform_ascendc::PlatformAscendC(pi);
    ci->aicCoreNum = plat.GetCoreNumAic();
    OP_CHECK_IF(ci->aicCoreNum <= 0, OP_LOGE(context->GetNodeName(), "aicCore<=0"), return ge::GRAPH_FAILED);
    ci->aivCoreNum = ci->aicCoreNum * AIV_DOUBLE;
    OP_CHECK_IF(ci->aivCoreNum <= 0, OP_LOGE(context->GetNodeName(), "aivCore<=0"), return ge::GRAPH_FAILED);
    uint64_t ub = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ci->ubSizePlatForm = static_cast<int64_t>(ub);
    ci->sysWorkspaceSize = static_cast<int64_t>(plat.GetLibApiWorkSpaceSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GruGrad).Tiling(TilingFunc4GruGrad).TilingParse<GruGradCompileInfo>(TilingPrepareForGruGrad);

} // namespace optiling
