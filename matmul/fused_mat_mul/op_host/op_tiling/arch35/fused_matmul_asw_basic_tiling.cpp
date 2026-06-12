/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_matmul_asw_basic_tiling.cpp
 * \brief
 */
#include "fused_matmul_asw_basic_tiling.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;
constexpr uint64_t SMALL_N_LIMIT = 16UL;
constexpr uint64_t SMALL_N_BASE_M_MEDIUM_K_MIN = 128UL;
constexpr uint64_t SMALL_N_BASE_M_SMALL_K_MIN = 2048UL;
constexpr uint64_t SMALL_N_BASE_M_LARGE = 1024UL;
constexpr uint64_t SMALL_N_BASE_M_MEDIUM = 512UL;
constexpr uint64_t SMALL_N_BASE_M_SMALL = 256UL;
constexpr uint64_t ADD_MUL_SMALL_N_LIMIT = 32UL;
constexpr uint64_t ADD_MUL_K_LIMIT = 4096UL;
constexpr uint64_t ADD_MUL_BASE_M_LARGE = 1024UL;
constexpr uint64_t ADD_MUL_BASE_M_SMALL = 512UL;

inline bool IsSmallNShape(const MatMulV3Args& args, string opType)
{
    if(opType == "add" || opType == "mul"){
        return args.nValue <= ADD_MUL_SMALL_N_LIMIT && args.kValue <= ADD_MUL_K_LIMIT;
    }
    return args.nValue < SMALL_N_LIMIT;
}

inline uint64_t GetSmallNBaseM(const MatMulV3Args& args, string opType)
{
    if(opType == "add" || opType == "mul"){
        if(args.aType == ge::DT_FLOAT){
            return ADD_MUL_BASE_M_SMALL;
        }
        return ADD_MUL_BASE_M_LARGE;
    }
    if (args.kValue >= SMALL_N_BASE_M_SMALL_K_MIN) {
        return SMALL_N_BASE_M_SMALL;
    }
    if (args.kValue >= SMALL_N_BASE_M_MEDIUM_K_MIN) {
        return SMALL_N_BASE_M_MEDIUM;
    }
    return SMALL_N_BASE_M_LARGE;
}

inline uint64_t GetSmallNBaseK(const MatMulV3Args& args, uint64_t baseM, uint64_t baseN)
{
    uint64_t baseKAlignValue = BASIC_BLOCK_SIZE_16;
    if (!args.isATrans && args.isBTrans && args.aType == ge::DT_FLOAT) {
        baseKAlignValue = BLOCK_BYTE_SIZE / args.aDtypeSize;
    }
    uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(args.kValue), baseKAlignValue);
    uint64_t kValueMax = ops::FloorAlign(
        ops::FloorDiv(L0A_SIZE_2 / DB_SIZE / args.aDtypeSize, std::max(baseM, baseN)), baseKAlignValue);
    return std::min(kValueAlign, kValueMax);
}
// ------------------------------ CheckBatch -------------------------------------------//
bool CheckBatchDefault(const MatMulV3Args& args)
{
    if (args.batchInfo->batchC > 1) {
        OP_LOGD(args.opName, "FusedMatMulAswBasicApiTiling does not support multiple batches");
        return false;
    }
    return true;
}

bool CheckBatchDav3510(const MatMulV3Args& args)
{
    if (args.batchInfo->batchC > 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            args.opName, "y", "",
            Ops::NN::FormatString("Batch-axis of y cannot be greater than 1").c_str());
        return false;
    }
    return true;
}

using CheckBatchFunc = bool (*)(const MatMulV3Args&);

const static std::map<NpuArch, CheckBatchFunc> CheckBatchFuncMap = {
    {NpuArch::DAV_3510, CheckBatchDav3510},
};
} // namespace

namespace optiling {
namespace fused_matmul {
using namespace strategy;
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulAswBasicApiTiling, DAV_3510, BASIC_ASWT_INHERITED_FROM_MMV3);
MM_REGISTER_TILING_TEMPLATE(FusedMatMul, FusedMatMulAswBasicApiTiling, DAV_RESV, BASIC_ASWT_INHERITED_FROM_MMV3);

bool FusedMatMulAswBasicApiTiling::CheckBatch() const
{
    auto iter = (CheckBatchFuncMap.find(compileInfo_.npuArch) == CheckBatchFuncMap.end()) ?
                    CheckBatchDefault :
                    CheckBatchFuncMap.at(compileInfo_.npuArch);
    return iter(args_);
}

bool FusedMatMulAswBasicApiTiling::IsCapable()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (!CheckBatch()) {
        return false;
    }
    if (args_.bFormat != ge::FORMAT_ND || args_.aFormat != ge::FORMAT_ND) {
        OP_LOGD(args_.opName, "ND is the only supported format for basic api");
        return false;
    }
    const bool isGeluOp = (opType == "gelu_erf" || opType == "gelu_tanh");
    if (opType == "add" || opType == "mul" || isGeluOp) {
        // AIV epilogue ops require one AIC paired with two AIVs.
        if (compileInfo_.aivNum != (compileInfo_.aicNum * NUM_TWO)) {
            OP_LOGD(args_.opName, "FusedMatMul aswt model only support aivNum == aicNum *2");
            return false;
        }
    }
    OP_LOGI(args_.opName, "FusedMatMul tiling enable state basic api");
    return true;
}

void FusedMatMulAswBasicApiTiling::UpdateSmallNTailInfo(uint64_t tailCnt)
{
    runInfo_.tailInfo.mCnt = 1UL;
    runInfo_.tailInfo.nCnt = 1UL;
    runInfo_.mBaseTailSplitCnt = 1UL;
    runInfo_.nBaseTailSplitCnt = 1UL;
    runInfo_.tailInfo.mTailMain = 0UL;
    runInfo_.tailInfo.nTailMain = 0UL;
    if (tailCnt == 0UL) {
        return;
    }

    // Spread tail tiles across M and N until the remaining work can occupy available AICs.
    while ((runInfo_.tailInfo.mCnt + 1UL) * runInfo_.tailInfo.nCnt * tailCnt <= compileInfo_.aicNum) {
        runInfo_.tailInfo.mCnt += 1UL;
        uint64_t nSubTileSize = runInfo_.baseN / (runInfo_.tailInfo.nCnt + 1UL);
        if (nSubTileSize >= ADD_MUL_SMALL_N_LIMIT &&
            runInfo_.tailInfo.mCnt * (runInfo_.tailInfo.nCnt + 1UL) * tailCnt <= compileInfo_.aicNum) {
            runInfo_.tailInfo.nCnt += 1UL;
        }
    }
}

void FusedMatMulAswBasicApiTiling::AdjustSmallNTiling(string opType)
{
    const uint64_t targetBaseM = GetSmallNBaseM(args_, opType);
    const uint64_t tmpBaseN = ops::CeilAlign(static_cast<uint64_t>(args_.nValue), BASIC_BLOCK_SIZE_16);
    const uint64_t maxBaseM = compileInfo_.l0CSize / DB_SIZE / std::max(tmpBaseN, BASIC_BLOCK_SIZE_16) / DATA_SIZE_FP32;
    if (maxBaseM < targetBaseM) {
        return;
    }

    const uint64_t tmpBaseM = std::min(ops::CeilAlign(static_cast<uint64_t>(args_.mValue), BASIC_BLOCK_SIZE_16), targetBaseM);
    const uint64_t baseK = GetSmallNBaseK(args_, tmpBaseM, tmpBaseN);
    if (baseK == 0UL) {
        return;
    }

    // Small-N GELU keeps N in one aligned tile and stretches M to improve AIC utilization.
    runInfo_.baseM = tmpBaseM;
    runInfo_.baseN = tmpBaseN;
    runInfo_.dbL0C = DB_SIZE;
    runInfo_.baseK = baseK;

    const uint64_t tileCnt = ops::CeilDiv(args_.mValue, runInfo_.baseM) * ops::CeilDiv(args_.nValue, runInfo_.baseN);
    const uint64_t tailCnt = tileCnt <= compileInfo_.aicNum ? 0UL : tileCnt % compileInfo_.aicNum;
    UpdateSmallNTailInfo(tailCnt);
    runInfo_.usedCoreNum = std::min(tileCnt, compileInfo_.aicNum);
}

void FusedMatMulAswBasicApiTiling::UpdateAswDepth()
{
    uint64_t remainSizeForAL1BL1 =
        args_.hasBias ? (compileInfo_.l1Size - BIAS_TABLE_NUM * DATA_SIZE_FP32) : compileInfo_.l1Size;
    // A/B share the remaining L1 space, so keep stepKa and stepKb identical for ASWT.
    runInfo_.stepKa =
        remainSizeForAL1BL1 / NUM_TWO / ((runInfo_.baseM + runInfo_.baseN) * runInfo_.baseK) / args_.aDtypeSize;
    runInfo_.stepKb = runInfo_.stepKa;
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
    runInfo_.depthB1 = runInfo_.stepKb * DB_SIZE;
}

void FusedMatMulAswBasicApiTiling::UpdateBFullLoadDepth()
{
    uint64_t kAligned = ops::CeilAlign(static_cast<uint64_t>(args_.kValue), BASIC_BLOCK_SIZE_16);
    uint64_t nAligned = ops::CeilAlign(static_cast<uint64_t>(args_.nValue), BASIC_BLOCK_SIZE_16);
    uint64_t bL1Size = kAligned * nAligned * args_.bDtypeSize;
    uint64_t baseBiasSize = args_.hasBias ?
        nAligned * GetSizeByDataType(args_.biasType) : 0;

    runInfo_.stepKb = ops::CeilDiv(static_cast<uint64_t>(args_.kValue), runInfo_.baseK);
    uint64_t stepN = ops::CeilDiv(static_cast<uint64_t>(args_.nValue), runInfo_.baseN);
    runInfo_.depthB1 = stepN * runInfo_.stepKb;

    if (compileInfo_.l1Size <= bL1Size + baseBiasSize) {
        return;
    }
    uint64_t remainL1 = compileInfo_.l1Size - bL1Size - baseBiasSize;
    uint64_t aPerStep = runInfo_.baseM * runInfo_.baseK * args_.aDtypeSize;
    if (aPerStep == 0) {
        return;
    }
    runInfo_.stepKa = remainL1 / DB_SIZE / aPerStep;
    runInfo_.stepKa = std::max(runInfo_.stepKa, 1UL);
    runInfo_.depthA1 = runInfo_.stepKa * DB_SIZE;
    uint64_t aL14Buffer = runInfo_.baseK * runInfo_.stepKa * runInfo_.baseM *
                          args_.aDtypeSize * BASIC_L1_BUFFER_NUM;
    runInfo_.l1BufferNum = aL14Buffer + bL1Size + baseBiasSize * BASIC_L1_BUFFER_NUM > compileInfo_.l1Size ?
                           DB_SIZE : BASIC_L1_BUFFER_NUM;

    runInfo_.dbL0C = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 * DB_SIZE <=
                     compileInfo_.l0CSize ? DB_SIZE : 1UL;
    runInfo_.mixInfo.ubDB = runInfo_.baseM * runInfo_.baseN * DATA_SIZE_FP32 <=
                            compileInfo_.ubSize ? DB_SIZE : 1UL;

    runInfo_.singleCoreM = runInfo_.baseM;
    runInfo_.singleCoreN = args_.nValue;
}

ge::graphStatus FusedMatMulAswBasicApiTiling::DoOpTiling() {
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    const bool isGeluOp = (opType == "gelu_erf" || opType == "gelu_tanh");
    if (isGeluOp) {
        // AIV epilogue ops share ASWT tiling, but use different kernel epilogues.
        OP_TILING_CHECK(
            (MatMulV3AswTiling::DoOpTiling() != ge::GRAPH_SUCCESS),
            CUBE_INNER_ERR_REPORT(args_.opName, "Do MatMul AswTiling failed in FusedMatMul."), return ge::GRAPH_FAILED);
        if (isGeluOp) {
            if (IsSmallNShape(args_, opType)) {
                // For small-N GELU, increase baseM to reduce data movement bound.
                AdjustSmallNTiling(opType);
            }
            // gelu 暂时只支持Basic模板
            fullLoad_ = MatMulV3FullLoad::NONE_FULL_LOAD;
            l0C2Out_ = MatMulV3L0C2Out::ON_THE_FLY;
        }
        UpdateAswDepth();
        return ge::GRAPH_SUCCESS;
    }
    // 16cast32 "" 等支持基础API全载模板
    ge::graphStatus status = MatMulV3BasicAswtTiling::DoOpTiling();
    if (l0C2Out_ == MatMulV3L0C2Out::ND_FIXPIPE_1_1) {
        l0C2Out_ = MatMulV3L0C2Out::ON_THE_FLY;
    }
    if ((opType == "add" || opType == "mul") && IsSmallNShape(args_, opType) && fullLoad_ == MatMulV3FullLoad::B_FULL_LOAD) {
        AdjustSmallNTiling(opType);
        UpdateBFullLoadDepth();
    }
    return status;
}
uint64_t FusedMatMulAswBasicApiTiling::GetTilingKey() const
{
    MatMulV3TilingKey tmp = MatMulV3TilingKey();
    MatMulV3TilingKey& tilingKey = tilingKeyObj == nullptr ? tmp : *tilingKeyObj;
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if (opType == "gelu_erf" || opType == "gelu_tanh") {
        // gelu 操作当前仅支持BASIC模板
        return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
            .SetFullLoad(MatMulV3FullLoad::NONE_FULL_LOAD)
            .SetModel(MatMulV3Model::BASIC)
            .SetL0C2Out(MatMulV3L0C2Out::ON_THE_FLY)
            .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
            .GetTilingKey();
    }
    return tilingKey.SetTrans(args_.isATrans, args_.isBTrans)
        .SetFullLoad(fullLoad_)
        .SetModel(MatMulV3Model::BASIC)
        .SetL0C2Out(l0C2Out_)
        .SetApiLevel(MatMulV3ApiLevel::BASIC_LEVEL)
        .GetTilingKey();
}

ge::graphStatus FusedMatMulAswBasicApiTiling::PostTiling()
{
    ge::graphStatus ret = MatMulV3BasicAswtTiling::PostTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    if ((opType == "gelu_erf" || opType == "gelu_tanh") && IsSmallNShape(args_, opType)) {
        size_t* workspaces = context_->GetWorkspaceSizes(1);
        OP_TILING_CHECK(workspaces == nullptr,
            CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspace is nullptr"), return ge::GRAPH_FAILED);
        // Kernel indexes the GELU GM workspace with the output matrix offset, so reserve the full output area.
        workspaces[0] += static_cast<size_t>(args_.mValue * args_.nValue * DATA_SIZE_FP32);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace fused_matmul
} // namespace optiling
