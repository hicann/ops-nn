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
 * \file gru_grad_tiling_arch35.cpp
 * \brief GRU 反向 Tiling（ascend950/dav_3510）实现。
 *
 * 要点：
 *  1. IMPL_OP_OPTILING(GruGrad) 注册点在本文件（随 ascend950 构建编译），运行时按
 *     Ops::Base::IsRegbaseSocVersion 分派；Prepare 统一走基线头文件的 inline 实现。
 *  2. ClampDepth：四路 MM GetTiling 后对 depthA1/depthB1 钳制为 2（字段语义不变量）。
 *  3. GetWorkspaceSize：基线段（dGhGm/dGiGm [TS,3H] + hPrevWs [TS,H] + dhPrevWs/dhFromH [B,H]，
 *     输入 dtype）之后追加扩展段 dGiT/dGhT [TS,3H]（输入 dtype）+ dhPrevF32 [B,H] +
 *     dwIF32 [3H,I] + dwHF32 [3H,H] + dxF32 [TS,I]（fp32），段序与 kernel arch35/gru_grad.h
 *     的 InitWorkBuffers 逐段对齐；总 workspace 含 GetLibApiWorkSpaceSize 系统段。
 */

#include "gru_grad_tiling_arch35.h"
#include <cstdint>
#include <iostream>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "tiling/tiling_api.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {

const int64_t GG_GATES_NUM = 3;
const int64_t GG_AIV_DOUBLE = 2;
const int64_t GG_FP32_BYTES = 4;
const int64_t GG_INPUT_DIM_NUM = 3;
const int64_t GG_INPUT_DIM_NUM_COMPACT = 2;
const int64_t GG_DEFAULT_UB_RESERVE_SIZE = 1024;
const int64_t GG_DEFAULT_ALIGNED_FP16 = 16;
const int64_t GG_DEFAULT_ALIGNED_FP32 = 8;
const int64_t GG_DEFAULT_BUFFER_SPACE = -1;
const int64_t GG_DEFAULT_SPLIT_FACTOR = 4;
const int64_t GG_DEFAULT_REDUCE_N_LIMIT = 128;
const int64_t GG_DEFAULT_COPY_FACTOR_FP32 = 4;
const int64_t GG_DEFAULT_COPY_FACTOR_FP16 = 6;
const int64_t GG_DEFAULT_ELEMENTS_PER_PART = 4096;
const int64_t GG_MMAD_FRACT_ROWS = 16; // Mmad 分形整装的越界读垫片行数（与 kernel CUBE_BLOCK 同源）
const int64_t GG_WNZ_MIN_HIDDEN = 512; // wNz 预暂存域门：H 下限
const int64_t GG_WNZ_MIN_TIMESTEP = 8; // wNz 预暂存域门：T 下限（预装摊销需足量步数）
const int64_t GG_S1_MIN_HIDDEN = 1024; // K 切分域门：3H/kChunk >= 核数 的 H 下限
const int64_t GG_S1_MAX_BATCH = 512;   // K 切分域门：部分和段 [核数,B,H] 的 B 上限
// 四路 MM tiling 的 L1 双缓冲深度上限（大 H TPipe 槽位 65>64 规避）
const int64_t GRU_GRAD_ARCH35_MAX_DEPTH = 2;

const int64_t GG_INPUT_X = 0, GG_INPUT_WI = 1, GG_INPUT_WH = 2, GG_INPUT_H0 = 3;
const int64_t GG_INPUT_H = 4, GG_INPUT_R = 5, GG_INPUT_Z = 6, GG_INPUT_N = 7, GG_INPUT_HN = 8, GG_INPUT_DY = 9,
              GG_INPUT_DH = 10, GG_INPUT_BS = 11;
const int64_t GG_OUT_DX = 0, GG_OUT_DHP = 1, GG_OUT_DWI = 2, GG_OUT_DWH = 3, GG_OUT_DBI = 4, GG_OUT_DBH = 5;
const int64_t GG_T_IDX = 0, GG_B_IDX = 1, GG_S_IDX = 2;

const std::vector<std::string> GG_SUPPORT_DIRECTION = {"UNIDIRECTIONAL", "REDIRECTIONAL"};

bool GruGradTilingArch35::IsCapable() { return true; }

ge::graphStatus GruGradTilingArch35::GetPlatformInfo()
{
    OP_LOGD(nodeName_, "GruGradTiling GetPlatformInfo.");
    compileInfo_ = static_cast<const GruGradCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo_);

    sysAicCoreNum_ = static_cast<int64_t>(compileInfo_->aicCoreNum);
    OP_LOGD(nodeName_, "sysAicCoreNum=%ld.", static_cast<long>(sysAicCoreNum_));
    sysAivCoreNum_ = static_cast<int64_t>(compileInfo_->aivCoreNum);
    tilingData_.ubSize = compileInfo_->ubSizePlatForm;

    auto dataType = context_->GetInputDesc(GG_INPUT_X)->GetDataType();
    inputDSize_ = dataType == ge::DT_FLOAT ? GG_FP32_BYTES : GG_FP32_BYTES / GG_AIV_DOUBLE;
    alignedPara_ = dataType == ge::DT_FLOAT ? GG_DEFAULT_ALIGNED_FP32 : GG_DEFAULT_ALIGNED_FP16;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::GetShapeAttrsInfo()
{
    nodeName_ = context_->GetNodeName();
    auto xIn = context_->GetInputShape(GG_INPUT_X);
    OP_CHECK_IF(xIn == nullptr, OP_LOGE(nodeName_, "x nullptr"), return ge::GRAPH_FAILED);
    auto xS = xIn->GetStorageShape();

    auto h0In = context_->GetInputShape(GG_INPUT_H0);
    OP_CHECK_IF(h0In == nullptr, OP_LOGE(nodeName_, "h0 nullptr"), return ge::GRAPH_FAILED);
    auto h0S = h0In->GetStorageShape();

    auto wiIn = context_->GetInputShape(GG_INPUT_WI);
    OP_CHECK_IF(wiIn == nullptr, OP_LOGE(nodeName_, "wi nullptr"), return ge::GRAPH_FAILED);

    auto whIn = context_->GetInputShape(GG_INPUT_WH);
    OP_CHECK_IF(whIn == nullptr, OP_LOGE(nodeName_, "wh nullptr"), return ge::GRAPH_FAILED);

    auto bsS = context_->GetOptionalInputShape(GG_INPUT_BS);
    auto bsD = context_->GetOptionalInputDesc(GG_INPUT_BS);
    tilingData_.isSeqLength = (bsD && bsS && bsS->GetStorageShape().GetDimNum() != 0) ? 1 : 0;

    if (tilingData_.isSeqLength) {
        OP_CHECK_IF(xS.GetDimNum() != GG_INPUT_DIM_NUM_COMPACT, OP_LOGE(nodeName_, "x must 2D for seq_length mode"),
                    return ge::GRAPH_FAILED);
        tilingData_.totalSteps = xS.GetDim(GG_T_IDX);  // dim 0
        tilingData_.inputSize = xS.GetDim(GG_B_IDX);   // dim 1 = I (2D: [totalSteps, I])
        tilingData_.hiddenSize = h0S.GetDim(GG_S_IDX); // h0 is [1, B, H]
        tilingData_.batch = h0S.GetDim(GG_B_IDX);      // from init_h dim 1
        tilingData_.timeStep = bsS->GetStorageShape().GetDim(0);
    } else {
        OP_CHECK_IF(xS.GetDimNum() != GG_INPUT_DIM_NUM, OP_LOGE(nodeName_, "x must 3D for fixed-length mode"),
                    return ge::GRAPH_FAILED);
        tilingData_.timeStep = xS.GetDim(GG_T_IDX);
        tilingData_.batch = xS.GetDim(GG_B_IDX);
        tilingData_.inputSize = xS.GetDim(GG_S_IDX);
        tilingData_.hiddenSize = h0S.GetDim(GG_S_IDX);
        tilingData_.totalSteps = tilingData_.timeStep * tilingData_.batch;
    }

    OP_TILING_CHECK(tilingData_.inputSize <= 0 || tilingData_.hiddenSize <= 0 || tilingData_.timeStep <= 0 ||
                        tilingData_.batch <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "shape<=0"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CheckParamsShape(), VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "shape fail"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(!CheckAttr(), VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "attr fail"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::DoOpTiling()
{
    // 本 kernel 为基础 API 手驱实现（无 KFC），调度模式不参与握手/同步路径，不设置 ScheduleMode。

    int64_t elementsPerPart = GG_DEFAULT_ELEMENTS_PER_PART;

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

    OP_TILING_CHECK(GetMatmulTiling() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "matmul tiling failed"), return ge::GRAPH_FAILED);
    ReduceBlockCalculate();
    SplitDxhBlockCalculate();
    ConcatXhBlockCalculate();

    tilingData_.inputSizeAligned = (tilingData_.inputSize + alignedPara_ - 1) / alignedPara_ * alignedPara_;
    tilingData_.hiddenSizeAligned = D_aligned;
    tilingData_.oneLineAligned = tilingData_.inputSizeAligned + tilingData_.hiddenSizeAligned;

    // 巨型 shape 驻留失效时 w_hh 每步 ND2NZ 变换为 dgate 串行路径的主要开销——
    // 预暂存 NZ 像(一次)后每步线性装载。host 域门(粗): 非变长+H>=512+T>=8;
    // kernel 侧复核(plan 对齐/驻留失效), 不满足则弃用(段位闲置无害)。
    // 环境变量可强制关闭。
    tilingData_.wNzEnabled = (getenv("GG_WNZ_DISABLE") == nullptr && tilingData_.isSeqLength == 0 &&
                              tilingData_.hiddenSize >= GG_WNZ_MIN_HIDDEN &&
                              tilingData_.timeStep >= GG_WNZ_MIN_TIMESTEP) ?
                                 1 :
                                 0;
    // StreamK 式 dgate K 维切分——提升 billed MMAD 利用率, 跨簇部分和
    // 规约回 [B,H]。需 wNz 域(B 侧线性装载) + H>=1024(3H/kChunk >= 28 核)。
    tilingData_.s1Enabled = (getenv("GG_S1_DISABLE") == nullptr && tilingData_.wNzEnabled == 1 &&
                             tilingData_.hiddenSize >= GG_S1_MIN_HIDDEN && tilingData_.batch <= GG_S1_MAX_BATCH) ?
                                1 :
                                0;

    SetTilingData();
    context_->SetBlockDim(sysAicCoreNum_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }
ge::graphStatus GruGradTilingArch35::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus GruGradTilingArch35::GetWorkspaceSize()
{
    // 段序与 kernel gru_grad.h InitWorkBuffers 逐段一致（字节级契约，勿调整段序/元素数）：
    //  前五段（输入 dtype）：dGhGm/dGiGm [TS,3H] + hPrevWs [TS,H] + dhPrevWs/dhFromH 各 [B,H]
    //    （原 xRevWs [TS,I] 段已删除：kernel 零消费，与大 shape OOM 风险相关）
    //  扩展段：dGiT/dGhT 各 [TS,3H]（输入 dtype）+ dhPrevF32 [B,H] + dwIF32 [3H,I] +
    //    dwHF32 [3H,H] + dxF32 [TS,I]（fp32）
    int64_t TS = tilingData_.totalSteps; // 不定长 = sum(batch_sizes), 定长 = T*B
    int64_t H = tilingData_.hiddenSize;
    int64_t B = tilingData_.batch;
    int64_t I = tilingData_.inputSize;
    // 乘积链运算前守卫（T·B 守卫仅覆盖 totalSteps 本身）：TS·3H、TS·H、TS·I、B·H、3H·I、
    // 3H·H、aicoreCnt·B·H 及求和可回绕——回绕小正值致 workspace 欠预留+kernel 越界寻址。
    // 逐因子做除法上界检查（各维已保证 >0）。
    OP_TILING_CHECK(H > INT64_MAX / GG_GATES_NUM || TS > INT64_MAX / (GG_GATES_NUM * H) || TS > INT64_MAX / H ||
                        TS > INT64_MAX / I || B > INT64_MAX / H || H > INT64_MAX / I || H > INT64_MAX / H ||
                        sysAicCoreNum_ > INT64_MAX / (B * H),
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "workspace size product overflow"),
                    return ge::GRAPH_FAILED);
    int64_t r1 = TS * GG_GATES_NUM * H; // dGhGm
    int64_t r2 = TS * GG_GATES_NUM * H; // dGiGm
    int64_t r3 = TS * H;                // hPrevWsGm
    int64_t r5 = B * H;                 // dhPrevWsGm (per-batch, independent of time)
    int64_t r6 = B * H;                 // dhFromHGm
    int64_t r7 = TS * I;                // xT（dwIh^T GEMM 的 A 侧转置物化 [I,TS]）
    int64_t r8 = TS * H;                // hPrevT（dwHh^T GEMM 的 A 侧转置物化 [H,TS]）
    // Mmad 16 行分形整装的 GM 读越界落入段尾垫片（dGh/dGi +16*3H、dhPrevWs +16*H），
    // 不越过 workspace 末段；kernel InitWorkBuffers 分段偏移不含垫（段间读越界进下段头，
    // 数据面无害；写路径无越界）。
    const int64_t padDgate = GG_MMAD_FRACT_ROWS * H * GG_GATES_NUM;
    const int64_t padDhPrev = GG_MMAD_FRACT_ROWS * H;
    r1 += padDgate;
    r2 += padDgate;
    r5 += padDhPrev;
    // xRevWs [TS,I] 段已删除：kernel 零消费（deprecated 段位），与 kernel InitWorkBuffers 同步移除
    int64_t ws1 = (r1 + r2 + r3 + r5 + r6 + r7 + r8) * inputDSize_;
    int64_t ws2 = (B * H + GG_GATES_NUM * H * I + GG_GATES_NUM * H * H + TS * I) *
                  GG_FP32_BYTES; // dhPrevF32/dwIF32/dwHF32/dxF32
    // wNz 段(输入 dtype, dxF32 之后)——w_hh [3H,H] 的 NZ 像(分形布局, 容量=3H*H 对齐域)
    int64_t r9 = tilingData_.wNzEnabled ? (GG_GATES_NUM * H * H) : 0;
    int64_t ws3 = r9 * inputDSize_;
    // dgate K 切分部分和段(fp32)——[aicoreCnt, B, H], 跨簇规约后回投 dhPrevF32。
    // 域门: wNz 域 + H>=1024(保证 chunk 数>=核数, 空切片簇不存在) + 非变长 + T>=8。
    int64_t r10 = tilingData_.s1Enabled ? (sysAicCoreNum_ * B * H) : 0;
    int64_t ws4 = r10 * GG_FP32_BYTES;

    size_t* ws = context_->GetWorkspaceSizes(1);
    // 系统段（GetLibApiWorkSpaceSize）：本 kernel 无 KFC、不调 GetSysWorkSpacePtr，预留无害。
    ws[0] = ws1 + ws2 + ws3 + ws4 + compileInfo_->sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

bool GruGradTilingArch35::ValidateInputShape(int idx, const std::vector<int64_t>& e)
{
    auto in = context_->GetInputShape(idx);
    if (!in)
        return false;
    auto s = in->GetStorageShape();
    if (s.GetDimNum() != static_cast<int64_t>(e.size()))
        return false;
    for (size_t i = 0; i < e.size(); i++)
        if (e[i] != s.GetDim(i))
            return false;
    return true;
}

bool GruGradTilingArch35::ValidateOutputShape(int idx, const std::vector<int64_t>& e)
{
    auto out = context_->GetOutputShape(idx);
    if (!out)
        return false;
    auto s = out->GetStorageShape();
    if (s.GetDimNum() != static_cast<int64_t>(e.size()))
        return false;
    for (size_t i = 0; i < e.size(); i++)
        if (e[i] != s.GetDim(i))
            return false;
    return true;
}

void GruGradTilingArch35::LogShapeCheck(bool isInput, int index, const std::vector<int64_t>& expected, const char* name,
                                        bool ok)
{
    auto shape = isInput ? context_->GetInputShape(index) : context_->GetOutputShape(index);
    if (shape) {
        auto s = shape->GetStorageShape();
        std::string actual;
        for (size_t d = 0; d < s.GetDimNum(); d++)
            actual += std::to_string(s.GetDim(d)) + (d + 1 < s.GetDimNum() ? "," : "");
        std::string exp;
        for (size_t d = 0; d < expected.size(); d++)
            exp += std::to_string(expected[d]) + (d + 1 < expected.size() ? "," : "");
        OP_LOGI(nodeName_, "Check%s[%d] %s expected=[%s] actual=[%s] %s", isInput ? "Input" : "Output", index, name,
                exp.c_str(), actual.c_str(), ok ? "PASS" : "FAIL");
    } else {
        OP_LOGI(nodeName_, "Check%s[%d] %s shape is nullptr, FAIL", isInput ? "Input" : "Output", index, name);
    }
}

bool GruGradTilingArch35::ValidateAndLogShape(bool isInput, int idx, const std::vector<int64_t>& expected,
                                              const char* name)
{
    bool ok = isInput ? ValidateInputShape(idx, expected) : ValidateOutputShape(idx, expected);
    LogShapeCheck(isInput, idx, expected, name, ok);
    return ok;
}

void GruGradTilingArch35::LogTCubeTiling(const char* nm, const AscendC::tiling::TCubeTiling& q)
{
    OP_LOGD(nodeName_, "%s M=%ld N=%ld Ka=%ld sM=%ld sN=%ld bM=%ld bN=%ld bK=%ld usedC=%ld.", nm,
            static_cast<long>(q.M), static_cast<long>(q.N), static_cast<long>(q.Ka), static_cast<long>(q.singleCoreM),
            static_cast<long>(q.singleCoreN), static_cast<long>(q.baseM), static_cast<long>(q.baseN),
            static_cast<long>(q.baseK), static_cast<long>(q.usedCoreNum));
}

bool GruGradTilingArch35::CheckParamsShape()
{
    int64_t TS = tilingData_.totalSteps;
    int64_t B = tilingData_.batch;
    int64_t H = tilingData_.hiddenSize;
    int64_t I = tilingData_.inputSize;

    std::vector<int64_t> dxD = tilingData_.isSeqLength ? std::vector<int64_t>{TS, I} :
                                                         std::vector<int64_t>{tilingData_.timeStep, B, I};
    std::vector<int64_t> ihD = {1, B, H};
    std::vector<int64_t> hidD = tilingData_.isSeqLength ? std::vector<int64_t>{TS, H} :
                                                          std::vector<int64_t>{tilingData_.timeStep, B, H};
    std::vector<int64_t> wiD = {GG_GATES_NUM * H, I};
    std::vector<int64_t> whD = {GG_GATES_NUM * H, H};
    std::vector<int64_t> biD = {GG_GATES_NUM * H};
    std::vector<int64_t> bhD = {GG_GATES_NUM * H};

    OP_LOGI(nodeName_, "CheckParamsShape: T=%ld B=%ld I=%ld H=%ld totalSteps=%ld isSeqLength=%ld", tilingData_.timeStep,
            B, I, H, TS, tilingData_.isSeqLength);

    bool ret = true;
    ret = ValidateAndLogShape(true, GG_INPUT_WI, wiD, "w_input") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_WH, whD, "w_hidden") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_H0, ihD, "init_h") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_H, hidD, "h") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_R, hidD, "r") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_Z, hidD, "z") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_N, hidD, "n") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_HN, hidD, "hn") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_DY, hidD, "dy") && ret;
    ret = ValidateAndLogShape(true, GG_INPUT_DH, ihD, "dh") && ret;
    ret = ValidateAndLogShape(false, GG_OUT_DX, dxD, "dx") && ret;
    ret = ValidateAndLogShape(false, GG_OUT_DHP, ihD, "dh_prev") && ret;
    ret = ValidateAndLogShape(false, GG_OUT_DWI, wiD, "dw_input") && ret;
    ret = ValidateAndLogShape(false, GG_OUT_DWH, whD, "dw_hidden") && ret;
    ret = ValidateAndLogShape(false, GG_OUT_DBI, biD, "db_input") && ret;
    ret = ValidateAndLogShape(false, GG_OUT_DBH, bhD, "db_hidden") && ret;
    if (tilingData_.isSeqLength) {
        std::vector<int64_t> bsD = {tilingData_.timeStep};
        ret = ValidateAndLogShape(true, GG_INPUT_BS, bsD, "batch_sz") && ret;
    }
    return ret;
}

bool GruGradTilingArch35::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* dir = attrs->GetAttrPointer<char>(0);
    OP_CHECK_IF(std::find(GG_SUPPORT_DIRECTION.begin(), GG_SUPPORT_DIRECTION.end(), dir) == GG_SUPPORT_DIRECTION.end(),
                OP_LOGE(nodeName_, "direction not supported"), return false);
    tilingData_.direction = (std::string(dir) == "REDIRECTIONAL") ? 1 : 0;

    const bool* hasBiasAttr = attrs->GetAttrPointer<bool>(1);
    tilingData_.isBias = (hasBiasAttr != nullptr && *hasBiasAttr) ? 1 : 0;
    return true;
}

void GruGradTilingArch35::ClampDepth(AscendC::tiling::TCubeTiling& param)
{
    // 大 H 时按默认 depth 规划会申请超过 TPipe 64 槽位上限的 UB 队列，
    // 钳制为 2 规避。本 kernel 为基础 API 裸路径（手驱 Mmad/Fixpipe，无 TPipe 槽位消费），
    // 钳制保留为 TCubeTiling 字段语义不变量。
    // 注意：TilingData 内嵌的 TCubeTiling 是 AscendC::tiling 命名空间的公有字段形态
    // （host/kernel 布局一致）；host 侧另有 optiling::TCubeTiling（访问器形态），二者同名不同型，
    // 未限定名在 optiling 命名空间内会解析到后者，必须显式限定。
    const int32_t depth = static_cast<int32_t>(GRU_GRAD_ARCH35_MAX_DEPTH);
    param.depthA1 = param.depthA1 > depth ? depth : param.depthA1;
    param.depthB1 = param.depthB1 > depth ? depth : param.depthB1;
}

ge::graphStatus GruGradTilingArch35::GetMatmulTiling()
{
    auto geDataType = context_->GetInputDesc(GG_INPUT_X)->GetDataType();
    auto mmDataType = static_cast<matmul_tiling::DataType>(geDataType);
    // matmul_tiling 库 SetShape/SetOrgShape 形参为 int32：batch/TS 截断为负或回绕小值时
    // GEMM 计划按错误 M/K 生成（或 helper 静默失败、零值 tiling 仍报成功→kernel 除 0）。
    // 元数据合法即可达，补运算前域校验（拒绝路径保护，EB 级输入本就无法物化）。
    OP_TILING_CHECK(tilingData_.batch > INT32_MAX || tilingData_.totalSteps > INT32_MAX ||
                        tilingData_.inputSize > INT32_MAX || tilingData_.hiddenSize > INT32_MAX,
                    VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "shape exceeds int32 matmul tiling domain"),
                    return ge::GRAPH_FAILED);
    if (GetDgateMMTiling(mmDataType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetDwIhMMTiling(mmDataType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetDwHhMMTiling(mmDataType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetDxMMTiling(mmDataType) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::GetDgateMMTiling(matmul_tiling::DataType mmDataType)
{
    // ========== MM1: dgateMM — grad_h_prev = d_gh × w_hh ==========
    // A: d_gh [B, 3H], B: w_hh [3H, H], C: grad_h_prev [B, H]
    matmul_tiling::MultiCoreMatmulTiling dgateMM;
    auto ret = dgateMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = dgateMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = dgateMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetCType fail."),
                    return ge::GRAPH_FAILED);
    ret = dgateMM.SetDim(sysAicCoreNum_);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetDim fail."),
                    return ge::GRAPH_FAILED);
    // M=B (最大batch), N=H, K=3H
    ret = dgateMM.SetOrgShape(tilingData_.batch, tilingData_.hiddenSize, tilingData_.hiddenSize * GG_GATES_NUM);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dgateMM.SetShape(tilingData_.batch, tilingData_.hiddenSize, tilingData_.hiddenSize * GG_GATES_NUM);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dgateMM.SetBufferSpace(GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);
    ret = dgateMM.GetTiling(tilingData_.dgateMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dgateMM GetTiling fail."),
                    return ge::GRAPH_FAILED);
    ClampDepth(tilingData_.dgateMMParam);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::GetDwIhMMTiling(matmul_tiling::DataType mmDataType)
{
    // ========== MM2a: dwIhMM — dw_ih = d_gi^T × x [3H,TS]×[TS,I]=[3H,I] ==========
    // （优化栈 kernel 侧 TransA 由 AIV TransposeGiGhT 物化承担，A 存储为 dGiT [3H,TS]，此处仍按
    //   TransA 语义产 tiling，singleCoreM/baseM 等分块字段语义不变。）
    int64_t TS = tilingData_.totalSteps; // 不定长 = sum(batch_sizes), 定长 = T*B
    matmul_tiling::MultiCoreMatmulTiling dwIhMM;
    auto ret = dwIhMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, true);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetCType fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.SetDim(sysAicCoreNum_);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetDim fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.SetOrgShape(tilingData_.hiddenSize * GG_GATES_NUM, tilingData_.inputSize, TS);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.SetShape(tilingData_.hiddenSize * GG_GATES_NUM, tilingData_.inputSize, TS);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.SetBufferSpace(GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);
    ret = dwIhMM.GetTiling(tilingData_.dwIhMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwIhMM GetTiling fail."),
                    return ge::GRAPH_FAILED);
    ClampDepth(tilingData_.dwIhMMParam);
    auto& q = tilingData_.dwIhMMParam;
    OP_LOGD(nodeName_, "dwIhMM M=%ld N=%ld Ka=%ld Kb=%ld sM=%ld sN=%ld sK=%ld bM=%ld bN=%ld bK=%ld usedCores=%ld.",
            static_cast<long>(q.M), static_cast<long>(q.N), static_cast<long>(q.Ka), static_cast<long>(q.Kb),
            static_cast<long>(q.singleCoreM), static_cast<long>(q.singleCoreN), static_cast<long>(q.singleCoreK),
            static_cast<long>(q.baseM), static_cast<long>(q.baseN), static_cast<long>(q.baseK),
            static_cast<long>(q.usedCoreNum));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::GetDwHhMMTiling(matmul_tiling::DataType mmDataType)
{
    // ========== MM2b: dwHhMM — dw_hh = h^T × d_gh [H,TS]×[TS,3H]=[H,3H] ==========
    int64_t TS = tilingData_.totalSteps; // 不定长 = sum(batch_sizes), 定长 = T*B
    matmul_tiling::MultiCoreMatmulTiling dwHhMM;
    auto ret = dwHhMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, true);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetCType fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.SetDim(sysAicCoreNum_);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetDim fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.SetOrgShape(tilingData_.hiddenSize * GG_GATES_NUM, tilingData_.hiddenSize, TS);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.SetShape(tilingData_.hiddenSize * GG_GATES_NUM, tilingData_.hiddenSize, TS);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.SetBufferSpace(GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);
    ret = dwHhMM.GetTiling(tilingData_.dwHhMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dwHhMM GetTiling fail."),
                    return ge::GRAPH_FAILED);
    ClampDepth(tilingData_.dwHhMMParam);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GruGradTilingArch35::GetDxMMTiling(matmul_tiling::DataType mmDataType)
{
    // ========== MM3: dxMM — dx = d_gi × w_input ==========
    int64_t TS = tilingData_.totalSteps; // 不定长 = sum(batch_sizes), 定长 = T*B
    matmul_tiling::MultiCoreMatmulTiling dxMM;
    auto ret = dxMM.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetAType fail."),
                    return ge::GRAPH_FAILED);
    ret = dxMM.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetBType fail."),
                    return ge::GRAPH_FAILED);
    ret = dxMM.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetCType fail."),
                    return ge::GRAPH_FAILED);
    ret = dxMM.SetDim(sysAicCoreNum_);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetDim fail."),
                    return ge::GRAPH_FAILED);
    // M=TS, N=I, K=3H
    ret = dxMM.SetOrgShape(TS, tilingData_.inputSize, tilingData_.hiddenSize * GG_GATES_NUM);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetOrgShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dxMM.SetShape(TS, tilingData_.inputSize, tilingData_.hiddenSize * GG_GATES_NUM);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetShape fail."),
                    return ge::GRAPH_FAILED);
    ret = dxMM.SetBufferSpace(GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE, GG_DEFAULT_BUFFER_SPACE);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM SetBufferSpace fail."),
                    return ge::GRAPH_FAILED);
    ret = dxMM.GetTiling(tilingData_.dxMMParam);
    OP_TILING_CHECK(ret == -1, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "dxMM GetTiling fail."),
                    return ge::GRAPH_FAILED);
    ClampDepth(tilingData_.dxMMParam);
    LogTCubeTiling("dwIh", tilingData_.dwIhMMParam);
    LogTCubeTiling("dwHh", tilingData_.dwHhMMParam);
    LogTCubeTiling("dgate", tilingData_.dgateMMParam);
    LogTCubeTiling("dx", tilingData_.dxMMParam);
    return ge::GRAPH_SUCCESS;
}

void GruGradTilingArch35::ReduceBlockCalculate()
{
    tilingData_.baseReduceN = Ops::Base::CeilDiv(
                                  Ops::Base::CeilDiv(tilingData_.hiddenSize * GG_GATES_NUM, sysAivCoreNum_),
                                  alignedPara_) *
                              alignedPara_;
    tilingData_.baseReduceN = tilingData_.baseReduceN > GG_DEFAULT_REDUCE_N_LIMIT ? GG_DEFAULT_REDUCE_N_LIMIT :
                                                                                    tilingData_.baseReduceN;
    tilingData_.maxReduceNumOnce = (tilingData_.ubSize - GG_DEFAULT_UB_RESERVE_SIZE) / GG_FP32_BYTES /
                                   tilingData_.baseReduceN;
    tilingData_.reduceBlockSize = tilingData_.totalSteps;
    int64_t bn = Ops::Base::CeilDiv(tilingData_.hiddenSize * GG_GATES_NUM, tilingData_.baseReduceN);
    tilingData_.nReduceCnt = sysAivCoreNum_ < bn ? sysAivCoreNum_ : bn;
    int64_t pc = Ops::Base::CeilDiv(bn, tilingData_.nReduceCnt);
    tilingData_.nReduceCnt = Ops::Base::CeilDiv(bn, pc);
    tilingData_.singleCoreReduceN = pc * tilingData_.baseReduceN;
    tilingData_.singleCoreReduceNTail = tilingData_.hiddenSize * GG_GATES_NUM -
                                        (tilingData_.nReduceCnt - 1) * tilingData_.singleCoreReduceN;
}

CutBatchTiling GruGradTilingArch35::CalculateCutBatchTiling(int64_t ub, int64_t al, int64_t act, int64_t ml, int64_t b)
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

void GruGradTilingArch35::SplitDxhBlockCalculate()
{
    int64_t f = GG_DEFAULT_SPLIT_FACTOR;
    int64_t ub = Ops::Base::CeilAlign((tilingData_.ubSize - GG_DEFAULT_UB_RESERVE_SIZE) / f, alignedPara_);
    tilingData_.inputSizeAligned = Ops::Base::CeilAlign(tilingData_.inputSize, alignedPara_);
    tilingData_.hiddenSizeAligned = Ops::Base::CeilAlign(tilingData_.hiddenSize, alignedPara_);
    tilingData_.oneLineAligned = tilingData_.inputSizeAligned + tilingData_.hiddenSizeAligned;
    int64_t ml = Ops::Base::CeilDiv(tilingData_.batch, sysAivCoreNum_);
    if (ub > tilingData_.oneLineAligned && tilingData_.isSeqLength == 0) {
        dxhInputParam_ = dxhHiddenParam_ = CalculateCutBatchTiling(
            ub, tilingData_.oneLineAligned, tilingData_.inputSize + tilingData_.hiddenSize, ml, tilingData_.batch);
    } else if (tilingData_.isSeqLength == 0) {
        f = inputDSize_ == GG_FP32_BYTES ? GG_DEFAULT_COPY_FACTOR_FP32 : GG_DEFAULT_COPY_FACTOR_FP16;
        ub = Ops::Base::CeilAlign((tilingData_.ubSize - GG_DEFAULT_UB_RESERVE_SIZE) / f, alignedPara_);
        dxhInputParam_ = CalculateCutBatchTiling(ub, tilingData_.inputSizeAligned, tilingData_.inputSize, ml,
                                                 tilingData_.batch);
        dxhHiddenParam_ = CalculateCutBatchTiling(ub, tilingData_.hiddenSizeAligned, tilingData_.hiddenSize, ml,
                                                  tilingData_.batch);
    }
}

void GruGradTilingArch35::ConcatXhBlockCalculate()
{
    int64_t f = inputDSize_ == GG_FP32_BYTES ? GG_DEFAULT_COPY_FACTOR_FP32 : GG_DEFAULT_COPY_FACTOR_FP16;
    int64_t ub = Ops::Base::CeilDiv((tilingData_.ubSize - GG_DEFAULT_UB_RESERVE_SIZE) / f, alignedPara_) * alignedPara_;
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

void GruGradTilingArch35::SetTilingData()
{
    tilingData_.dxhInputTiling = dxhInputParam_;
    tilingData_.dxhHiddenTiling = dxhHiddenParam_;
    tilingData_.xhInputTiling = xhInputParam_;
    tilingData_.xhHiddenTiling = xhHiddenParam_;

    auto* buf = context_->GetTilingData<GruGradTilingData>();
    *buf = tilingData_;
    context_->GetRawTilingData()->SetDataSize(sizeof(GruGradTilingData));
}

// TilingFunc4GruGrad：由注册点按 SocVersion 分派调用。
static ge::graphStatus TilingFunc4GruGrad(gert::TilingContext* ctx)
{
    GruGradTilingArch35 obj(ctx);
    auto ret = obj.DoTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(ctx->GetNodeName(), "GruGradTiling failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 优化栈（ascend950）注册点：本目录 tiling 仅随对应 SoC 构建编译（add_tiling_sources 按
// TILING_DIR glob），与基线目录互斥。TilingPrepare 共用基线头文件 inline 实现。
IMPL_OP_OPTILING(GruGrad).Tiling(TilingFunc4GruGrad).TilingParse<GruGradCompileInfo>(TilingPrepareForGruGrad);

} // namespace optiling
