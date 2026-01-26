/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_sliding_window_tiling.cc
 * \brief
 */
#include "common/op_host/op_tiling/tiling_type.h"
#include "quant_batch_matmul_v3_checker_base.h"
#include "quant_batch_matmul_v3_checker.h"
#include "quant_batch_matmul_v3_checker_for_mmads8s4.h"
#include "op_cache_tiling.h"
#include "log/log.h"
#include "common/inc/error_util.h"
#include "adaptive_sliding_window_tiling.h"

using Ops::NN::MathUtil;
using namespace QuantBatchMatmulV3Arch35TilingKey;

namespace {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_REDUCE_BLOCK = 32;
constexpr uint64_t MX_GROUP_SIZE = 32;
constexpr uint64_t MXFP_DIVISOR_SIZE = 64;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint64_t PER_BLOCK_SIZE = 128;
constexpr uint64_t PER_BLOCK_BASE_SIZE_256 = 256;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_1024 = 1024;
constexpr uint64_t BASIC_BLOCK_SIZE_32 = 32;
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t DOUBLE_CORE_NUM = 2;
constexpr uint32_t DB_SIZE = 2;

constexpr uint32_t SLIDING_WINOW_SHORTER_EDGE = 4;
constexpr uint32_t SLIDING_WINOW_LONGER_EDGE = 8;
constexpr uint32_t UB_ALIGN_SIZE = 32;
constexpr uint64_t MTE2_MIN_LOAD_SIZE_V100 = 32768;
constexpr uint64_t L2_ALIGN_SIZE = 128;

constexpr uint32_t SCALER_FACTOR_MAX = 127;
constexpr uint32_t SCALER_FACTOR_MIN = 1;
constexpr uint32_t SCALER_FACTOR_DEFAULT = 1;
constexpr uint32_t SCALER_FACTOR_B_BIT = 8;
constexpr uint32_t SCALER_FACTOR_M_BIT = 16;
constexpr uint32_t SCALER_FACTOR_N_BIT = 24;

constexpr uint64_t BASEM_BASEN_RATIO = 2;
constexpr uint64_t BASEK_LIMIT = 4095;

constexpr uint64_t AFULLLOAD_SINGLE_CORE_A_SCALER = 2;
constexpr uint32_t DATA_SIZE_L0C = 4;
constexpr uint32_t VEC_CORE_GROUP_NUM = 2;

constexpr uint64_t WINDOW_LEN = 4;                // v100 的滑窗大小
constexpr uint64_t MTE2_CACHELINE_SIZE = 256;     // MTE2 CACHELINE大小
constexpr uint64_t LOAD_BALANCE_THRESHOLD = 1792; // 执行外轴负载均衡优化时，M轴或N轴的最小尺寸阈值

// supportMmadS8S4平台
constexpr uint64_t MIN_CARRY_DATA_SIZE_32K = 32 * 1024UL;
constexpr uint64_t FULL_LOAD_DATA_SIZE_64K = 64 * 1024UL;
constexpr uint64_t CACHE_LINE_512B = 512UL;
constexpr uint32_t MAX_STEPK_With_BL1_FULL = 8U;
} // namespace

namespace optiling {

AdaptiveSlidingWindowTiling::AdaptiveSlidingWindowTiling(gert::TilingContext* context)
    : QuantBatchMatmulV3TilingBase(context, false), tilingData_(tilingDataSelf_)
{
    Reset();
}

AdaptiveSlidingWindowTiling::AdaptiveSlidingWindowTiling(
    gert::TilingContext* context, DequantBmm::QuantBatchMatmulV3TilingDataParams* out)
    : QuantBatchMatmulV3TilingBase(context, true), tilingData_(*out)
{
    Reset();
    InitCompileInfo();
    inputParams_.Reset();
}

void AdaptiveSlidingWindowTiling::Reset()
{
    isBf16Opt_ = false;
    isUbQuant_ = false;

    if (!isTilingOut_) {
        tilingData_ = DequantBmm::QuantBatchMatmulV3TilingDataParams();
        OP_TILING_CHECK(
            memset_s(
                context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                context_->GetRawTilingData()->GetCapacity()) != EOK,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to clear tiling data"), return);
    }
}

void AdaptiveSlidingWindowTiling::LoadBalanceDataReset()
{
    adaptiveWin_.mBaseTailSplitCnt = 1UL;
    adaptiveWin_.nBaseTailSplitCnt = 1UL;
    adaptiveWin_.mTailMain = 0UL;
    adaptiveWin_.nTailMain = 0UL;
}

bool AdaptiveSlidingWindowTiling::CheckDtype() const
{
    QuantBatchMatmulV3CheckerBase* qmmV3Checker = nullptr;
    if (compileInfo_.supportMmadS8S4) {
        qmmV3Checker = new (std::nothrow) QuantBatchMatmulV3Checker4MmadS8S4(context_, inputParams_);
    } else {
        qmmV3Checker = new (std::nothrow) QuantBatchMatmulV3Checker(context_, inputParams_);
    }
    OP_TILING_CHECK(
        qmmV3Checker == nullptr, CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to instantiate qmmV3Checker"),
        return false);
    bool res = qmmV3Checker->CheckDtype();
    delete qmmV3Checker;
    OP_TILING_CHECK(!res, CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype fail"), return false);
    return true;
}

bool AdaptiveSlidingWindowTiling::CheckShape(
    const std::vector<gert::Shape*>& mandtoryShape, const gert::StorageShape* biasShape,
    const gert::StorageShape* pertokenShape, const std::vector<int64_t>& dimValueOfMKN) const
{
    QuantBatchMatmulV3CheckerBase *qmmV3Checker = nullptr;
    if (compileInfo_.supportMmadS8S4) {
         qmmV3Checker = new (std::nothrow) QuantBatchMatmulV3Checker4MmadS8S4(context_, inputParams_);
    } else {
         qmmV3Checker = new (std::nothrow) QuantBatchMatmulV3Checker(context_, inputParams_);
    }
    OP_TILING_CHECK(qmmV3Checker == nullptr,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "failed to instantiate qmmV3Checker"), return false);
    bool res = qmmV3Checker->CheckShape(mandtoryShape, biasShape, pertokenShape, dimValueOfMKN);
    delete qmmV3Checker;
    OP_TILING_CHECK(!res, CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckShape fail"), return false);
    return true;
}

ge::graphStatus AdaptiveSlidingWindowTiling::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "SetPlatformInfo fail");
    if (aicoreParams_.aicNum == 0UL || aicoreParams_.l1Size == 0UL || aicoreParams_.l0cSize == 0UL) {
        OP_LOGE(
            inputParams_.opName, "coreNum/L1Size/L0cSize should not be 0. coreNum: %lu, L1Size: %lu, L0cSize: %lu",
            aicoreParams_.aicNum, aicoreParams_.l1Size, aicoreParams_.l0cSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    tilingDataSize_ = sizeof(DequantBmm::QuantBatchMatmulV3TilingDataParams);
    return QuantBatchMatmulV3TilingBase::GetShapeAttrsInfo();
}

ge::graphStatus AdaptiveSlidingWindowTiling::DoOpTiling()
{
    OP_LOGD(inputParams_.opName, "DoOpTiling of adaptive sliding window tiling strategy.");
    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(inputParams_.opName, "DoOpTiling fail");
        return ge::GRAPH_FAILED;
    }
    LoadBalanceDataReset();
    if (!OptimizeEdgeBasicBlock()) {
        OP_LOGE(inputParams_.opName, "OptimizeEdgeBasicBlock fail");
        return ge::GRAPH_FAILED;
    }
    SetBf16Compat();
    CalL1Tiling();
    if (inputParams_.isPertoken) {
        CalcUbTiling();
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::DoLibApiTiling()
{
    QuantBatchMatMulV3TilingUtil::SetBasicLibApiTiling(inputParams_, basicTiling_, tilingData_);
    if (inputParams_.isMxPerGroup) {
        tilingData_.matmulTiling.mxTypePara =
            (SCALER_FACTOR_MIN << SCALER_FACTOR_N_BIT) + (SCALER_FACTOR_MIN << SCALER_FACTOR_M_BIT);
        if (basicTiling_.scaleFactorA >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorA <= SCALER_FACTOR_MAX &&
            basicTiling_.scaleFactorB >= SCALER_FACTOR_MIN && basicTiling_.scaleFactorB <= SCALER_FACTOR_MAX) {
            tilingData_.matmulTiling.mxTypePara +=
                (basicTiling_.scaleFactorB << SCALER_FACTOR_B_BIT) + basicTiling_.scaleFactorA;
        } else {
            tilingData_.matmulTiling.mxTypePara +=
                (SCALER_FACTOR_DEFAULT << SCALER_FACTOR_B_BIT) + SCALER_FACTOR_DEFAULT;
        }
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t AdaptiveSlidingWindowTiling::GetBiasMode() const
{
    return QuantBatchMatMulV3TilingUtil::GetBiasMode(inputParams_);
}

uint64_t AdaptiveSlidingWindowTiling::GetKernelType() const
{
    return QuantBatchMatMulV3TilingUtil::GetKernelType(inputParams_, basicTiling_, isBf16Mix_, isAFullLoad_,
                                                       isBFullLoad_, isABFullLoad_);
}

uint64_t AdaptiveSlidingWindowTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(
        static_cast<uint64_t>(inputParams_.transA), static_cast<uint64_t>(inputParams_.transB), GetBiasMode(),
        GetKernelType());
}

ge::graphStatus AdaptiveSlidingWindowTiling::GetWorkspaceSize()
{
    workspaceSize_ = inputParams_.libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::PostTiling()
{
    OP_TILING_CHECK(
        tilingDataSize_ % sizeof(uint64_t) != 0UL,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Tiling data size[%zu] is not aligned to 8.", tilingDataSize_),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(&tilingData_), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AdaptiveSlidingWindowTiling::CalcUbTiling()
{
    uint64_t ubSize = static_cast<uint64_t>(aicoreParams_.ubSize);
    basicTiling_.ubCalcN = basicTiling_.baseN;
    // src(int32) + scale(fp32) + pertoken(fp32) + out(fp16/bf16), in and out need double buffer
    // int16_t reprersent bf16, input src + output dst + veccalc dequant api
    uint64_t ubCalc =
        static_cast<uint64_t>(DOUBLE_BUFFER_NUM * (sizeof(int32_t) + ge::GetSizeByDataType(inputParams_.cDtype))) *
        static_cast<uint64_t>(basicTiling_.ubCalcN);
    // input: scale perchannel
    if (!inputParams_.isPerTensor) {
        ubSize -= ge::GetSizeByDataType(inputParams_.scaleDtype) * basicTiling_.ubCalcN;
    }
    // input: pertokenScale fp32
    ubCalc += DOUBLE_BUFFER_NUM * ge::GetSizeByDataType(inputParams_.perTokenScaleDtype);
    // ub buffer size need 32 align
    ubSize -= DOUBLE_BUFFER_NUM * (UB_ALIGN_SIZE - ge::GetSizeByDataType(inputParams_.perTokenScaleDtype));
    basicTiling_.ubCalcM = static_cast<uint32_t>(ubSize / ubCalc);
    basicTiling_.ubCalcM = std::min(
        std::min(basicTiling_.ubCalcM, ops::CeilDiv(basicTiling_.baseM, VEC_CORE_GROUP_NUM)),
        ops::CeilDiv(static_cast<uint32_t>(inputParams_.mSize), VEC_CORE_GROUP_NUM));
    return ge::GRAPH_SUCCESS;
}

bool AdaptiveSlidingWindowTiling::AnalyseSlidingWinInfo()
{
    if (!CalcBasicBlock()) {
        OP_LOGE(inputParams_.opName, "inappropriate basicBlock");
        return false;
    }
    adaptiveWin_.mBlockCnt = ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    adaptiveWin_.totalBlockCnt = adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = inputParams_.mSize - (adaptiveWin_.mBlockCnt - 1UL) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = inputParams_.nSize - (adaptiveWin_.nBlockCnt - 1UL) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, aicoreParams_.aicNum);
    adaptiveWin_.tailWinBlockCnt = adaptiveWin_.totalBlockCnt % aicoreParams_.aicNum;
    IsABFullLoad();
    if (isABFullLoad_) {
        adaptiveWin_.useTailWinLogic = false;
        return true;
    }
    IsAFullLoad();
    IsBFullLoad();
    if (adaptiveWin_.useTailWinLogic) {
        if (isAFullLoad_) {
            CalcTailBasicBlockAfullLoad();
        } else if (isBFullLoad_) {
            CalcTailBasicBlockBfullLoad();
        } else {
            CalcTailBasicBlock();
        }
    }
    return true;
}

bool AdaptiveSlidingWindowTiling::CalcBasicBlock()
{
    // baseM/baseN/baseK对齐原则
    // 原则1：L1上x1、x2都是NZ排布，转置可选          须满足内轴32B对齐，外轴16对齐
    // 原则2：L0上x1是Zz排布，x2是Zn排布，非转置      须满足baseM、baseN关于16对齐，baseK关于32B对齐
    // 理论上需要同时满足原则1和原则2,但全量化的x1Dtype可取{8bit, 4bit}, x2Dtype可取{8bit, 4bit}
    // sizeof(x1Dtype) <= 1, sizeof(x2Dtype) <= 1
    // baseM满足原则1时，可能是关于32B对齐或关于16对齐，间接满足原则2
    // baseN满足原则1时，可能是关于32B对齐或关于16对齐，间接满足原则2
    // baseK满足32B对齐时，同时满足原则1和原则2
    if (inputParams_.isPerBlock) {
        // when m or n is not divisible by 128, baseM or baseN is 128,
        // to prevent the division of tail blocks into two vector cores from generating a non-factor of 128.
        // when m <= 128, baseM is 128.
        // only one of baseM and baseN can be 256, with baseM taking priority.
        if (inputParams_.mSize <= PER_BLOCK_SIZE || inputParams_.mSize % PER_BLOCK_SIZE != 0UL) {
            adaptiveWin_.baseM = PER_BLOCK_SIZE;
            if (inputParams_.nSize % PER_BLOCK_SIZE == 0UL) {
                adaptiveWin_.baseN = PER_BLOCK_BASE_SIZE_256;
            } else {
                adaptiveWin_.baseN = PER_BLOCK_SIZE;
            }
        } else {
            adaptiveWin_.baseM = PER_BLOCK_BASE_SIZE_256;
            adaptiveWin_.baseN = PER_BLOCK_SIZE;
        }
        adaptiveWin_.baseK = PER_BLOCK_SIZE;
    } else {
        adaptiveWin_.baseM = std::min(inputParams_.mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
        adaptiveWin_.baseM =
            !inputParams_.transA ?
                ops::CeilAlign(adaptiveWin_.baseM, CUBE_BLOCK) :
                ops::CeilAlign(adaptiveWin_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.aDtype));
        adaptiveWin_.baseN = std::min(inputParams_.nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
        adaptiveWin_.baseN =
            inputParams_.transB ?
                ops::CeilAlign(adaptiveWin_.baseN, CUBE_BLOCK) :
                ops::CeilAlign(adaptiveWin_.baseN, GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype));
        uint64_t baseKDefaultSize =
            static_cast<uint64_t>(GetShapeWithDataType(BASIC_BLOCK_SIZE_128, inputParams_.aDtype));

        adaptiveWin_.baseK = ops::CeilAlign(
            std::min(baseKDefaultSize, inputParams_.kSize),
            inputParams_.isMxPerGroup ? MXFP_DIVISOR_SIZE :
                                        GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype));

        // supportMmadS8S4平台，当A8W4增量shape，在l0Size允许下，baseK可以超过BASIC_BLOCK_SIZE_128
        if (compileInfo_.supportMmadS8S4) {
            uint64_t basicBlockSizeA = static_cast<uint64_t>(BASIC_BLOCK_SIZE_128);
            uint64_t basicBlockSizeB = static_cast<uint64_t>(BASIC_BLOCK_SIZE_128);
            if (aicoreParams_.l0aSize / DB_SIZE / adaptiveWin_.baseM >= static_cast<uint64_t>(BASIC_BLOCK_SIZE_256)) {
                basicBlockSizeA = static_cast<uint64_t>(BASIC_BLOCK_SIZE_256);
            }
            if (aicoreParams_.l0bSize / DB_SIZE / adaptiveWin_.baseN >= static_cast<uint64_t>(BASIC_BLOCK_SIZE_256)) {
                basicBlockSizeB = static_cast<uint64_t>(BASIC_BLOCK_SIZE_256);
            }
            uint64_t minBaseK = std::min(std::min(GetShapeWithDataType(basicBlockSizeA, inputParams_.aDtype),
                                                  GetShapeWithDataType(basicBlockSizeB, inputParams_.bDtype)),
                                         inputParams_.kSize);
            uint64_t maxAlignSize = std::max(GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype),
                                             GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.bDtype));
            adaptiveWin_.baseK = ops::CeilAlign(minBaseK, maxAlignSize);
        }
    }
    uint64_t oriBlock =
        ops::CeilDiv(inputParams_.mSize, adaptiveWin_.baseM) * ops::CeilDiv(inputParams_.nSize, adaptiveWin_.baseN);
    bool isSmallBlock = oriBlock < aicoreParams_.aicNum;
    if (isSmallBlock && !inputParams_.isPerBlock) {
        if (compileInfo_.supportMmadS8S4) {
            AdjustBasicBlock4MmadS8S4(oriBlock);
        } else {
            AdjustBasicBlock();
        }
        OP_TILING_CHECK(adaptiveWin_.baseM == 0UL || adaptiveWin_.baseN == 0UL || adaptiveWin_.baseK == 0UL,
                        CUBE_INNER_ERR_REPORT(
                            inputParams_.opName,
                            "BaseM, baseN and baseK should be greater than 0, but baseM: %lu, baseN: %lu, baseK: %lu,",
                            adaptiveWin_.baseM, adaptiveWin_.baseN, adaptiveWin_.baseK),
                        return false);
    }
    return true;
}

void AdaptiveSlidingWindowTiling::AdjustBasicBlock()
{
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        inputParams_.transB ? CUBE_BLOCK : GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB) ?
                                 GetShapeWithDataType(BASIC_BLOCK_SIZE_32, inputParams_.aDtype) :
                                 GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    if (IsMxBackwardTrans()) {
        baseKAlignNum = GetShapeWithDataType(MXFP_DIVISOR_SIZE, inputParams_.aDtype);
    }
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = adaptiveWin_.baseM;
    uint64_t tempBaseN = adaptiveWin_.baseN;
    if (mMaxtile * nMaxtile >= aicoreParams_.aicNum || (!inputParams_.transA && inputParams_.transB)) {
        uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, adaptiveWin_.baseM);
        uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, adaptiveWin_.baseN);
        if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == CUBE_BLOCK)) {
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
            mCore = MathUtil::CeilDivision(inputParams_.mSize, tempBaseM);
            nCore = aicoreParams_.aicNum / mCore;
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
        } else {
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
            nCore = MathUtil::CeilDivision(inputParams_.nSize, tempBaseN);
            mCore = aicoreParams_.aicNum / nCore;
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
        }

        while (tempBaseN >= tempBaseM * BASEM_BASEN_RATIO && nCore < aicoreParams_.aicNum / NUM_HALF &&
               tempBaseN != baseNAlignNum) {
            nCore = nCore * DOUBLE_CORE_NUM;
            mCore = aicoreParams_.aicNum / nCore;
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
            mCore = MathUtil::CeilDivision(inputParams_.mSize, static_cast<uint64_t>(tempBaseM));
            nCore = MathUtil::CeilDivision(inputParams_.nSize, static_cast<uint64_t>(tempBaseN));
        }
        while (tempBaseM >= tempBaseN * BASEM_BASEN_RATIO && mCore < aicoreParams_.aicNum / NUM_HALF &&
               tempBaseM != baseMAlignNum) {
            mCore = mCore * DOUBLE_CORE_NUM;
            nCore = aicoreParams_.aicNum / mCore;
            tempBaseM = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, nCore), baseNAlignNum);
            mCore = MathUtil::CeilDivision(inputParams_.mSize, static_cast<uint64_t>(tempBaseM));
            nCore = MathUtil::CeilDivision(inputParams_.nSize, static_cast<uint64_t>(tempBaseN));
        }
        uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(inputParams_.kSize), baseKAlignNum);
        uint64_t kValueMax =
            GetShapeWithDataType(aicoreParams_.l0aSize / DB_SIZE, inputParams_.aDtype) / std::max(tempBaseM, tempBaseN);
        if (kValueMax >= baseKAlignNum) {
            adaptiveWin_.baseM = tempBaseM;
            adaptiveWin_.baseN = tempBaseN;
            kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
            adaptiveWin_.baseK = std::min(kValueAlign, kValueMax);
            adaptiveWin_.baseK = adaptiveWin_.baseK > BASEK_LIMIT ? adaptiveWin_.baseK / NUM_HALF : adaptiveWin_.baseK;
            adaptiveWin_.useTailWinLogic = false;
        }
    }
}

void AdaptiveSlidingWindowTiling::SetBf16Compat()
{
    bool isMix = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64 &&
                  inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0) &&
                 inputParams_.isPerChannel;
    bool isCompat = (inputParams_.scaleDtype != ge::DT_UINT64 && inputParams_.scaleDtype != ge::DT_INT64 &&
                     inputParams_.scaleDtype != ge::DT_FLOAT8_E8M0) &&
                    inputParams_.isPerTensor && inputParams_.aDtype == ge::DT_INT8 && inputParams_.hasBias &&
                    (inputParams_.biasDtype == ge::DT_FLOAT || inputParams_.biasDtype == ge::DT_BF16);
    isBf16Mix_ = isMix || isCompat;
}

bool AdaptiveSlidingWindowTiling::IsMxKOdd() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
           ops::CeilDiv(inputParams_.kSize, MX_GROUP_SIZE) % MXFP_MULTI_BASE_SIZE != 0;
}

bool AdaptiveSlidingWindowTiling::IsMxBackwardTrans() const
{
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 && (inputParams_.transA || !inputParams_.transB);
}

void AdaptiveSlidingWindowTiling::IsAFullLoad()
{
    if (inputParams_.batchA != 1UL || IsMxKOdd() || IsMxBackwardTrans() || isTilingOut_ || inputParams_.isPerBlock) {
        isAFullLoad_ = false;
        return;
    }

    singleCoreASizeWithFullLoad_ =
        adaptiveWin_.baseM *
        (inputParams_.transA
             ? GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.aDtype)
             : ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), CUBE_REDUCE_BLOCK));
    // supportMmadS8S4平台和david都是阈值256K
    constexpr uint64_t A_L1_LOAD_THRESHOLD = 256 * 1024UL;
    // supportMmadS8S4平台不需要判断mBlockCnt <= nBlockCnt，更大概率使能AFullLoad
    bool blockCntCmp = (adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt);
    if (compileInfo_.supportMmadS8S4) {
        blockCntCmp = true;
    }
    // 当mcnt为1/2/4, 并且coreNum是mcnt的整数倍时，ASW可保证所有核上的计算对M的依赖不换行;
    // coreNum必须大于等于mcnt, 否则A矩阵数据无法载入完全
    isAFullLoad_ =
        singleCoreASizeWithFullLoad_ <= A_L1_LOAD_THRESHOLD && blockCntCmp &&
        (((adaptiveWin_.mBlockCnt == 1UL || adaptiveWin_.mBlockCnt == 2UL || adaptiveWin_.mBlockCnt == 4UL) &&
          aicoreParams_.aicNum >= adaptiveWin_.mBlockCnt && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0) ||
         adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
}

bool AdaptiveSlidingWindowTiling::CheckBiasAndScale(uint64_t baseN, uint64_t dbL0c) const
{
    // bias int32(BT 4096B)对baseN的影响，不超过1024; 开DB不超过512
    // scale uint64(FB 4096B)目前对baseN无影响，api会对超512的scale再做tiling
    uint64_t maxBiasBaseN = dbL0c == 1UL ? BASIC_BLOCK_SIZE_1024 : BASIC_BLOCK_SIZE_512;
    // FB 2KB, api再切分，tbe tiling亦如此
    uint64_t maxScaleBaseN = dbL0c == 1UL ? BASIC_BLOCK_SIZE_512 : BASIC_BLOCK_SIZE_256;
    bool isBiasInvalid = inputParams_.hasBias && inputParams_.biasDtype != ge::DT_BF16 && baseN > maxBiasBaseN;
    bool isScaleInvalid = !isUbQuant_ && baseN > maxScaleBaseN;
    return !(isBiasInvalid || isScaleInvalid);
}

void AdaptiveSlidingWindowTiling::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(inputParams_.opName, "coreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;

    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = inputParams_.kSize;

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t totalL1Size = aicoreParams_.l1Size;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c =
        ((basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DB_SIZE <= aicoreParams_.l0cSize) &&
         CheckBiasAndScale(basicTiling_.baseN, DB_SIZE)) ?
            DB_SIZE :
            1U;
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0UL;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? basicTiling_.baseN * scaleDtypeSize : 0UL;
    uint64_t leftL1Size = totalL1Size - singleCoreBiasSize - singleCoreScaleSize;

    if (IsCalL1TilingDepth4MmadS8S4()) {
        CalL1TilingDepth4MmadS8S4(leftL1Size);
    } else if (isAFullLoad_) {
        CalL1TilingDepthAfullload(leftL1Size);
    } else {
        CalL1TilingDepthNotfullload(leftL1Size);
    }
}

void AdaptiveSlidingWindowTiling::CalL1TilingDepthAfullload(uint64_t leftL1Size)
{
    basicTiling_.stepKa = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    basicTiling_.depthA1 = basicTiling_.stepKa;

    uint64_t singleCoreASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.singleCoreK, inputParams_.aDtype);

    uint64_t leftL1SizeAfterFullA = leftL1Size - ops::CeilAlign(singleCoreASize, L1_ALIGN_SIZE);

    uint64_t basebSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

    if (inputParams_.isMxPerGroup) {
        uint64_t singleCoreScaleASize = GetSizeWithDataType(
            basicTiling_.singleCoreM *
                ops::CeilAlign(
                    ops::CeilDiv(static_cast<uint64_t>(basicTiling_.singleCoreK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE),
            inputParams_.perTokenScaleDtype);
        basicTiling_.scaleFactorA = 1U;
        leftL1SizeAfterFullA -= ops::CeilAlign(singleCoreScaleASize, L1_ALIGN_SIZE);
        basicTiling_.depthB1 = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
        basicTiling_.stepKb = basicTiling_.depthB1 == 1U ? basicTiling_.depthB1 : basicTiling_.depthB1 / DB_SIZE;
        basicTiling_.scaleFactorB = GetScaleFactorBAfullLoad(
            leftL1SizeAfterFullA - ops::CeilAlign(basicTiling_.depthB1 * basebSize, L1_ALIGN_SIZE));
    } else {
        basicTiling_.depthB1 = GetDepthB1AfullLoad(leftL1SizeAfterFullA);
    }

    basicTiling_.stepKb = basicTiling_.depthB1 == 1U ? basicTiling_.depthB1 : basicTiling_.depthB1 / DB_SIZE;
}

void AdaptiveSlidingWindowTiling::CalL1TilingDepthNotfullload(uint64_t leftL1Size)
{
    if (inputParams_.isMxPerGroup) {
        uint64_t baseASize = GetSizeWithDataType(basicTiling_.baseM * basicTiling_.baseK, inputParams_.aDtype);
        uint64_t baseBSize = GetSizeWithDataType(basicTiling_.baseN * basicTiling_.baseK, inputParams_.bDtype);

        uint64_t baseScaleASize = GetSizeWithDataType(
            ops::CeilAlign(
                ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) *
                basicTiling_.baseM,
            inputParams_.perTokenScaleDtype);
        uint64_t baseScaleBSize = GetSizeWithDataType(
            ops::CeilAlign(
                ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE) *
                basicTiling_.baseN,
            inputParams_.scaleDtype);
        uint64_t baseL1Size = baseASize + baseBSize + baseScaleASize + baseScaleBSize;
        uint64_t depthInit = GetDepthA1B1(leftL1Size, baseL1Size, 1UL);
        uint64_t leftL1SizeByDepthInit = leftL1Size - depthInit * (baseL1Size);
        uint64_t depthASec = GetDepthA1B1(leftL1SizeByDepthInit, (baseASize + baseScaleASize) * depthInit, depthInit);
        uint64_t depthBSec = GetDepthA1B1(leftL1SizeByDepthInit, (baseBSize + baseScaleBSize) * depthInit, depthInit);
        basicTiling_.depthA1 = std::max(depthASec, depthBSec);
        basicTiling_.depthB1 = basicTiling_.depthA1;
        if (basicTiling_.depthA1 * baseL1Size > leftL1Size) {
            basicTiling_.depthA1 = depthASec >= depthBSec ? depthASec : depthInit;
            basicTiling_.depthB1 = depthASec < depthBSec ? depthBSec : depthInit;
        }
        CalStepKs();
        CalScaleFactors(baseASize, baseBSize, baseScaleASize, baseScaleBSize);
    } else {
        // depthA1/B1恒>=2, 证明如下：
        // 已知baseM/N/K <= 256, sizeof(x1Dtype) <= 1, sizeof(x2Dtype) <=1
        // 则baseM * baseK * sizeof(x1Dtype) <= 16K, baseN * baseK * sizeof(x2Dtype) <= 16K
        // 又因为 L1 bias和scale所需最大空间分别是baseN * sizeof(biasDtype) = 0.5K, baseN * sizeof(scaleDtype) = 1K
        // 所以depth >= (L1Size - bias最大空间 - scale最大空间） / NUM_HALF / max(左基本块最大空间, 右基本块最大空间)
        //           >=  (L1Size - 1.5K - 1K) / 2 / 16K
        // 进一步计算，只要L1Size >= 66.5K, 则 depthA1/B1恒>=2，硬件规格显然满足
        basicTiling_.depthA1 =
            leftL1Size / NUM_HALF /
            GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM * basicTiling_.baseK), inputParams_.aDtype);
        basicTiling_.depthB1 =
            leftL1Size / NUM_HALF /
            GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN * basicTiling_.baseK), inputParams_.bDtype);
        CalStepKs();
    }
}

void AdaptiveSlidingWindowTiling::CalStepKs()
{
    basicTiling_.stepKa = basicTiling_.depthA1 / DB_SIZE;
    basicTiling_.stepKb = basicTiling_.depthB1 / DB_SIZE;

    if (static_cast<uint64_t>(basicTiling_.stepKa * basicTiling_.baseK) > inputParams_.kSize) {
        basicTiling_.stepKa = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (static_cast<uint64_t>(basicTiling_.stepKb * basicTiling_.baseK) > inputParams_.kSize) {
        basicTiling_.stepKb = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));
    }

    if (basicTiling_.stepKa > basicTiling_.stepKb) {
        basicTiling_.stepKa = basicTiling_.stepKa / basicTiling_.stepKb * basicTiling_.stepKb;
    }
    if (basicTiling_.stepKb > basicTiling_.stepKa) {
        basicTiling_.stepKb = basicTiling_.stepKb / basicTiling_.stepKa * basicTiling_.stepKa;
    }
    if (inputParams_.isPerBlock) {
        basicTiling_.stepKa =
            std::min(basicTiling_.stepKa, static_cast<uint32_t>(4)); // 限制stepKa最大为4, 防止issue queue阻塞
        basicTiling_.stepKb =
            std::min(basicTiling_.stepKb, static_cast<uint32_t>(4)); // 限制stepKb最大为4, 防止issue queue阻塞
    }
    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

void AdaptiveSlidingWindowTiling::CalScaleFactors(uint64_t baseASize, uint64_t baseBSize, uint64_t baseScaleASize,
                                                  uint64_t baseScaleBSize)
{
    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t baseBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;

    // 计算scaleFactorA, scaleFactorB
    // 来自K轴的约束
    uint32_t scaleFactorAMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V100 / baseScaleASize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorBMax =
        std::min(static_cast<uint32_t>(MTE2_MIN_LOAD_SIZE_V100 / baseScaleBSize), SCALER_FACTOR_MAX);
    uint32_t scaleFactorA = static_cast<uint32_t>(inputParams_.kSize) / (basicTiling_.stepKa * basicTiling_.baseK);
    uint32_t scaleFactorB = static_cast<uint32_t>(inputParams_.kSize) / (basicTiling_.stepKb * basicTiling_.baseK);
    basicTiling_.scaleFactorA = std::max(SCALER_FACTOR_MIN, scaleFactorA);
    basicTiling_.scaleFactorB = std::max(SCALER_FACTOR_MIN, scaleFactorB);
    basicTiling_.scaleFactorA = std::min(scaleFactorAMax, basicTiling_.scaleFactorA);
    basicTiling_.scaleFactorB = std::min(scaleFactorBMax, basicTiling_.scaleFactorB);

    // 来自L1 size 的约束
    uint64_t leftL1sie =
        aicoreParams_.l1Size - (basicTiling_.depthA1 * baseASize + basicTiling_.depthB1 * baseBSize + baseBiasSize);
    uint32_t scaleInit = static_cast<uint32_t>(
        leftL1sie / (basicTiling_.depthA1 * baseScaleASize + basicTiling_.depthB1 * baseScaleBSize));
    if (basicTiling_.scaleFactorA <= scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(basicTiling_.scaleFactorA) * basicTiling_.depthA1 * baseScaleASize);
        basicTiling_.scaleFactorB = std::min(
            static_cast<uint32_t>(leftL1sie / (basicTiling_.depthB1 * baseScaleBSize)), basicTiling_.scaleFactorB);
    } else if (basicTiling_.scaleFactorB <= scaleInit && basicTiling_.scaleFactorA > scaleInit) {
        leftL1sie -= (static_cast<uint64_t>(basicTiling_.scaleFactorB) * basicTiling_.depthB1 * baseScaleBSize);
        basicTiling_.scaleFactorA = std::min(
            static_cast<uint32_t>(leftL1sie / (basicTiling_.depthA1 * baseScaleASize)), basicTiling_.scaleFactorA);
    } else if (basicTiling_.scaleFactorA > scaleInit && basicTiling_.scaleFactorB > scaleInit) {
        leftL1sie -=
            (static_cast<uint64_t>(scaleInit) * basicTiling_.depthB1 * baseScaleBSize +
             static_cast<uint64_t>(scaleInit) * basicTiling_.depthA1 * baseScaleASize);
        uint32_t scaleASec = std::min(
            static_cast<uint32_t>(leftL1sie / (basicTiling_.depthA1 * baseScaleASize)),
            basicTiling_.scaleFactorA - scaleInit);
        uint32_t scaleBSec = std::min(
            static_cast<uint32_t>(leftL1sie / (basicTiling_.depthB1 * baseScaleBSize)),
            basicTiling_.scaleFactorB - scaleInit);
        basicTiling_.scaleFactorA = scaleASec >= scaleBSec ? (scaleASec + scaleInit) : scaleInit;
        basicTiling_.scaleFactorB = scaleASec < scaleBSec ? (scaleBSec + scaleInit) : scaleInit;
    }
}

uint64_t AdaptiveSlidingWindowTiling::GetDepthA1B1(uint64_t leftSize, uint64_t perDepthSize, uint64_t depthInit)
{
    if (depthInit > 1UL && perDepthSize > DB_SIZE * MTE2_MIN_LOAD_SIZE_V100) {
        return depthInit;
    }
    uint64_t depthScale = leftSize / perDepthSize;
    if (depthInit > 1UL) {
        uint64_t baseKSize = GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseK), inputParams_.aDtype);
        while ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
               (depthScale * baseKSize) > BASIC_BLOCK_SIZE_512) {
            depthScale -= 1UL;
        }
        if ((depthScale * baseKSize) % BASIC_BLOCK_SIZE_512 != 0UL &&
            (depthScale * baseKSize) >= BASIC_BLOCK_SIZE_256) {
            depthScale = BASIC_BLOCK_SIZE_256 / baseKSize;
        }
        depthScale = std::max(depthScale, static_cast<uint64_t>(1));
    } else {
        constexpr uint64_t index = 2; // 2: depth的值是2的幂
        depthScale = 1UL;
        while (depthScale * (perDepthSize) < leftSize) {
            depthScale *= index;
        }
        depthScale = depthScale == 1UL ? depthScale : depthScale / index;
    }
    return depthInit * depthScale;
}

uint64_t AdaptiveSlidingWindowTiling::GetDepthB1AfullLoad(uint64_t leftSize)
{
    // 内轴128B 对齐
    uint64_t stepKbBase = 1UL;
    if (inputParams_.transB) {
        uint64_t singleBaseKSize = GetSizeWithDataType(basicTiling_.baseK, inputParams_.bDtype);
        if (singleBaseKSize < L2_ALIGN_SIZE) {
            stepKbBase = ops::CeilDiv(L2_ALIGN_SIZE, singleBaseKSize);
        }
    }

    // 一次性的总搬运量不少于32KB，不超L1 内存
    uint64_t scaleBBaseK = 0UL;
    if (inputParams_.isMxPerGroup) {
        scaleBBaseK = ops::CeilAlign(
            ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    }

    uint64_t basebSize =
        GetSizeWithDataType(basicTiling_.baseN * (basicTiling_.baseK + scaleBBaseK) * stepKbBase, inputParams_.bDtype);
    uint64_t stepKbBaseScale = 1UL;
    if (leftSize >= MTE2_MIN_LOAD_SIZE_V100 * DB_SIZE) {
        stepKbBaseScale = ops::CeilDiv(MTE2_MIN_LOAD_SIZE_V100, basebSize);
    } else {
        stepKbBaseScale = ops::CeilDiv(leftSize / DB_SIZE, basebSize);
    }
    stepKbBase = stepKbBase * stepKbBaseScale;

    uint64_t refinedStepkb = 2UL; // stekpb为1时容易使mte1无法并行，因此在不超出l1 空间的情况下设置stepkb最小为2
    if (stepKbBase == 1UL && inputParams_.kSize > static_cast<uint64_t>(basicTiling_.baseK) &&
        leftSize > basebSize * refinedStepkb) {
        stepKbBase = refinedStepkb;
    }

    return stepKbBase * DB_SIZE;
}

uint64_t AdaptiveSlidingWindowTiling::GetScaleFactorBAfullLoad(uint64_t leftSize)
{
    uint64_t baseScaleBSize = GetSizeWithDataType(
        basicTiling_.baseN *
            ops::CeilAlign(
                ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE),
        inputParams_.scaleDtype);

    uint64_t scaleFactorbBase = 1UL;
    if (inputParams_.transB) {
        // k 是内轴的情况
        uint64_t singleScalebBasekSize = GetSizeWithDataType(
            ops::CeilAlign(
                ops::CeilDiv(static_cast<uint64_t>(basicTiling_.baseK), MX_GROUP_SIZE), MXFP_MULTI_BASE_SIZE),
            inputParams_.scaleDtype);
        if (singleScalebBasekSize < L2_ALIGN_SIZE) {
            scaleFactorbBase = ops::CeilDiv(L2_ALIGN_SIZE, singleScalebBasekSize);
        }
    }

    uint64_t scaleFactorbMaxFromK =
        inputParams_.kSize / static_cast<uint64_t>(basicTiling_.stepKb * basicTiling_.baseK);
    scaleFactorbMaxFromK = std::min(static_cast<uint64_t>(SCALER_FACTOR_MAX), scaleFactorbMaxFromK);
    scaleFactorbMaxFromK = std::max(static_cast<uint64_t>(SCALER_FACTOR_MIN), scaleFactorbMaxFromK);

    uint64_t scaleFactorB = 1;
    if (scaleFactorbBase > scaleFactorbMaxFromK) {
        // 一次性搬运所有的K 方向上所有的scaleB 也无法满足内轴128对齐,按照搬运量来计算scaleFactorb
        uint64_t scaleFactorBMax =
            std::min(MTE2_MIN_LOAD_SIZE_V100 * DB_SIZE, leftSize) / (baseScaleBSize * basicTiling_.depthB1);
        scaleFactorB = std::min(scaleFactorBMax, scaleFactorbMaxFromK);
    } else {
        // 在内轴128对齐的基础上，同时确保搬运量不小于32， 且不超L1 内存
        uint64_t scaleFactor = std::min(MTE2_MIN_LOAD_SIZE_V100 * DB_SIZE, leftSize) /
                               (baseScaleBSize * scaleFactorbBase * basicTiling_.depthB1);
        scaleFactorB = std::min(scaleFactor * scaleFactorbBase, scaleFactorbMaxFromK);
    }

    return scaleFactorB;
}

void AdaptiveSlidingWindowTiling::SetTilingData()
{
    QuantBatchMatMulV3TilingUtil::SetBasicTilingData(inputParams_, basicTiling_, tilingData_);
    tilingData_.adaptiveSlidingWin.mTailTile = adaptiveWin_.mTailTile;
    tilingData_.adaptiveSlidingWin.nTailTile = adaptiveWin_.nTailTile;
    tilingData_.adaptiveSlidingWin.mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_.adaptiveSlidingWin.mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_.adaptiveSlidingWin.nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);
}

uint32_t AdaptiveSlidingWindowTiling::CalUsedCoreNum()
{
    if (adaptiveWin_.totalWinCnt > 1UL || adaptiveWin_.tailWinBlockCnt == 0UL) {
        return aicoreParams_.aicNum;
    }

    return static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile);
}

uint32_t AdaptiveSlidingWindowTiling::CalUsedCoreNum(uint32_t mTile, uint32_t nTile)
{
    return mTile * nTile * static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt);
}

bool AdaptiveSlidingWindowTiling::IsInValidPerblockTailSplit(uint64_t splitCnt) const
{
    if (!inputParams_.isPerBlock) {
        return false;
    }

    return PER_BLOCK_SIZE % splitCnt != 0UL;
}

bool AdaptiveSlidingWindowTiling::IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (inputParams_.bFormat != ge::FORMAT_FRACTAL_NZ ||
        (!isAFullLoad_ && ((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
                           (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)))) {
        return false;
    }

    uint64_t tailN = adaptiveWin_.baseN / splitCnt;
    return tailN % GetShapeWithDataType(L1_ALIGN_SIZE, inputParams_.bDtype) != 0UL;
}

void AdaptiveSlidingWindowTiling::CalcTailBasicBlock()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    auto& preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto& secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;
    while (CalUsedCoreNum(preSplit + 1, secSplit) <= aicoreParams_.aicNum) {
        preSplit += 1UL;
        if (!IsInValidPerblockTailSplit(preSplit) && !IsInValidWeighNzTailSplit(preSplit, true)) {
            preSplitValid = preSplit;
        }

        if (CalUsedCoreNum(preSplit, secSplit + 1) <= aicoreParams_.aicNum) {
            secSplit += 1UL;
            if (!IsInValidPerblockTailSplit(secSplit) && !IsInValidWeighNzTailSplit(secSplit, false)) {
                secSplitValid = secSplit;
            }
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

void AdaptiveSlidingWindowTiling::CalcTailBasicBlockAfullLoad()
{
    adaptiveWin_.mTailTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t nTileValid = 1UL;
    constexpr uint64_t MIN_BASEN_PER_TILE = 16UL; // 尾窗口切分后N方向不少于16
    if (adaptiveWin_.tailWinBlockCnt != 0UL) {
        while (CalUsedCoreNum(adaptiveWin_.mTailTile, (nTile + 1UL)) <= aicoreParams_.aicNum &&
               adaptiveWin_.baseN / (nTile + 1UL) >= MIN_BASEN_PER_TILE) {
            nTile += 1UL;
            if (IsInValidWeighNzTailSplit(nTile, true)) {
                continue;
            }
            nTileValid = nTile;
        }
    }
    adaptiveWin_.nTailTile = nTileValid;
}

bool AdaptiveSlidingWindowTiling::GetOuterMAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    OP_TILING_CHECK(inputParams_.mSize == 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Input size of the M-axis is zero."),
                    return false);
    uint64_t mCnt = MathUtil::CeilDivision(inputParams_.mSize, adaptiveWin_.baseM);
    uint64_t mTailSize = inputParams_.mSize % adaptiveWin_.baseM;
    uint64_t baseTailCntMax = std::min((adaptiveWin_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, mCnt);
    uint64_t windowSize = std::min(WINDOW_LEN, mCnt);
    uint64_t mainWindowNum = mCnt / windowSize - 1UL;
    uint64_t tailWindowSize = mCnt - mainWindowNum * windowSize;
    uint64_t perfRes = (mainWindowNum + 1UL) * adaptiveWin_.baseM;
    uint64_t mergeWindowNum = 1UL;

    for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
         mergeLen += windowSize, ++mergeWindowNum) {
        uint64_t newTailMain = MathUtil::Align(
            MathUtil::CeilDivision((mergeLen * adaptiveWin_.baseM + mTailSize), mergeLen + 1UL), BASIC_BLOCK_SIZE_16);
        OP_TILING_CHECK(
            mainWindowNum + 1UL < mergeWindowNum,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                  "Subtraction underflow: mainWindowNum(%lu) + 1UL - mergeWindowNum(%lu).",
                                  mainWindowNum, mergeWindowNum),
            return false);
        uint64_t curPerf = (mainWindowNum + 1UL - mergeWindowNum) * adaptiveWin_.baseM + mergeWindowNum * newTailMain;
        if (curPerf <= perfRes) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
    return true;
}

uint64_t AdaptiveSlidingWindowTiling::CalculateCurrentPerf(uint64_t mergeLen, uint64_t nTail, uint64_t mCnt,
                                                           uint64_t nCnt, uint64_t& newTailMain)
{
    uint64_t totalWindows = MathUtil::CeilDivision(nCnt * mCnt, aicoreParams_.aicNum);
    newTailMain = MathUtil::Align(MathUtil::CeilDivision((mergeLen * adaptiveWin_.baseN + nTail), mergeLen + 1UL),
                                  BASIC_BLOCK_SIZE_16);
    OP_TILING_CHECK(adaptiveWin_.baseN < newTailMain,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Subtraction underflow: adaptiveWin_.baseN(%lu) - newTailMain(%lu).",
                                          adaptiveWin_.baseN, newTailMain),
                    return false);
    uint64_t newTailLast = mergeLen * (adaptiveWin_.baseN - newTailMain) + nTail;
    uint64_t newMainRound = 0UL;
    uint64_t newTailRound = 0UL;

    if (mergeLen < nCnt - 1UL) {
        newMainRound = MathUtil::CeilDivision(
            (nCnt - 1UL - mergeLen) * mCnt + (mergeLen + 1UL) * mCnt % aicoreParams_.aicNum, aicoreParams_.aicNum);
    }
    if (mergeLen > 0UL) {
        OP_TILING_CHECK(totalWindows < newMainRound,
                        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                              "Subtraction underflow: totalWindows(%lu) - newMainRound(%lu).",
                                              totalWindows, newMainRound),
                        return false);
        newTailRound =
            std::min(MathUtil::CeilDivision(mergeLen * mCnt + mCnt % aicoreParams_.aicNum, aicoreParams_.aicNum),
                     totalWindows - newMainRound);
    }

    OP_TILING_CHECK(
        totalWindows < newMainRound + newTailRound,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
                              "Subtraction underflow: totalWindows(%lu) - newMainRound(%lu) - newTailRound(%lu).",
                              totalWindows, newMainRound, newTailRound),
        return false);
    return newMainRound * adaptiveWin_.baseN + newTailRound * newTailMain +
           (totalWindows - newMainRound - newTailRound) * newTailLast;
}

bool AdaptiveSlidingWindowTiling::GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    uint64_t baseN = adaptiveWin_.baseN;
    uint64_t nCnt = MathUtil::CeilDivision(inputParams_.nSize, baseN);
    uint64_t mCnt = MathUtil::CeilDivision(inputParams_.mSize, adaptiveWin_.baseM);
    uint64_t nTail = inputParams_.nSize % baseN;
    uint64_t totalWindows = MathUtil::CeilDivision(nCnt * mCnt, aicoreParams_.aicNum);

    OP_TILING_CHECK(nCnt == 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName, "Subtraction underflow: nCnt(%lu) - 1UL and \
the divisor is zero: WINDOW_LEN %% nCnt.",
                                          nCnt),
                    return false);
    uint64_t mainWindows =
        MathUtil::CeilDivision((nCnt - 1UL) * mCnt + mCnt % aicoreParams_.aicNum, aicoreParams_.aicNum);

    OP_TILING_CHECK(aicoreParams_.aicNum == 0UL,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "The number of enabled Cube cores is 0."),
                    return false);
    if (nCnt * mCnt <= aicoreParams_.aicNum ||
        (mCnt % aicoreParams_.aicNum == 0UL && (nCnt % WINDOW_LEN == 0UL || WINDOW_LEN % nCnt == 0UL))) {
        mainWindows = totalWindows;
    }
    OP_TILING_CHECK(totalWindows < mainWindows,
                    CUBE_INNER_ERR_REPORT(inputParams_.opName,
                                          "Subtraction underflow: totalWindows(%lu) - mainWindows(%lu).",
                                          totalWindows, mainWindows),
                    return false);
    uint64_t tailWindows = totalWindows - mainWindows;
    uint64_t perfRes = mainWindows * baseN + tailWindows * nTail;

    uint64_t baseTailCntMax = std::min((baseN - nTail) / BASIC_BLOCK_SIZE_16, nCnt);
    for (uint64_t mergeLen = 1UL; mergeLen < baseTailCntMax; ++mergeLen) {
        uint64_t newTailMain = 0UL;
        uint64_t curPerf = CalculateCurrentPerf(mergeLen, nTail, mCnt, nCnt, newTailMain);
        if (curPerf < perfRes) {
            perfRes = curPerf;
            tailMain = newTailMain;
            baseTailSplitCnt = mergeLen + 1UL;
        }
    }
    return true;
}

bool AdaptiveSlidingWindowTiling::OptimizeEdgeBasicBlock()
{
    if (compileInfo_.supportMmadS8S4 || isAFullLoad_ || (inputParams_.transA && !inputParams_.transB) ||
        (inputParams_.isPerBlock && inputParams_.groupSizeM != 1UL)) {
        return true;
    }

    uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, adaptiveWin_.baseM);
    uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, adaptiveWin_.baseN);
    if (mCore == 1UL || nCore == 1UL) {
        return true;
    }

    uint64_t mBaseTail = static_cast<uint64_t>(inputParams_.mSize % adaptiveWin_.baseM);
    uint64_t nBaseTail = static_cast<uint64_t>(inputParams_.nSize % adaptiveWin_.baseN);
    bool isMxfp4 = inputParams_.isMxPerGroup &&
                   (inputParams_.aDtype == ge::DT_FLOAT4_E2M1 || inputParams_.aDtype == ge::DT_FLOAT4_E1M2);
    bool balanceAfterFixp = inputParams_.kSize < static_cast<uint64_t>(BASIC_BLOCK_SIZE_1024);
    bool isInnerAxisAlign = GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype) % MTE2_CACHELINE_SIZE == 0UL;
    if (mBaseTail > 0UL && !inputParams_.transA &&
        (isInnerAxisAlign || (inputParams_.mSize >= LOAD_BALANCE_THRESHOLD && !isMxfp4))) {
        if (!GetOuterMAxisTailCnt(adaptiveWin_.mBaseTailSplitCnt, adaptiveWin_.mTailMain)) {
            return false;
        };
    }
    if (nBaseTail > 0UL && inputParams_.transB && !balanceAfterFixp && !inputParams_.isPerBlock && !isMxfp4 &&
        (isInnerAxisAlign || (inputParams_.nSize >= LOAD_BALANCE_THRESHOLD))) {
        if (!GetOuterNAxisTailCnt(adaptiveWin_.nBaseTailSplitCnt, adaptiveWin_.nTailMain)) {
            return false;
        };
    }
    return true;
}

bool AdaptiveSlidingWindowTiling::IsCalL1TilingDepth4MmadS8S4() const
{
    // supportMmadS8S4平台 优化分支使能判断：
    // 输入为A8W8/A8W4/A4W4、量化模式为PerChannel/PerTensor
    bool A8W8Flag = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT8;
    bool A8W4Flag = inputParams_.aDtype == ge::DT_INT8 && inputParams_.bDtype == ge::DT_INT4;
    bool A4W4Flag = inputParams_.aDtype == ge::DT_INT4 && inputParams_.bDtype == ge::DT_INT4;
    bool quantModeSupport = inputParams_.isPerChannel || inputParams_.isPerTensor;
    return compileInfo_.supportMmadS8S4 && (A8W8Flag || A8W4Flag || A4W4Flag) && quantModeSupport;
}

bool AdaptiveSlidingWindowTiling::CalculateOptimalSplit(uint64_t &baseM, uint64_t &baseN, uint64_t baseMAlignNum,
                                                        uint64_t baseNAlignNum, uint64_t baseKAlignNum)
{
    // 计算最大切分候选
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);

    // 初始化变量
    uint64_t maxUsedCore = MathUtil::CeilDivision(inputParams_.mSize, adaptiveWin_.baseM) *
                           MathUtil::CeilDivision(inputParams_.nSize, adaptiveWin_.baseN);
    uint64_t maxDiff = UINT64_MAX;
    uint64_t iterMSplite = std::min(mMaxtile, aicoreParams_.aicNum);
    uint64_t iterNSplite = std::min(nMaxtile, aicoreParams_.aicNum);
    bool optimalFound = false;
    // 遍历所有可能的切分方案
    for (uint64_t mFactor = 1; mFactor <= iterMSplite; ++mFactor) {
        for (uint64_t nFactor = 1; nFactor <= iterNSplite; ++nFactor) {
            uint64_t tempMBase = mFactor * baseMAlignNum;
            uint64_t mCore = MathUtil::CeilDivision(inputParams_.mSize, tempMBase);
            uint64_t tempNBase = nFactor * baseNAlignNum;
            uint64_t nCore = MathUtil::CeilDivision(inputParams_.nSize, tempNBase);
            uint64_t usedCore = mCore * nCore;
            uint64_t diff = (tempMBase >= tempNBase) ? tempMBase - tempNBase : tempNBase - tempMBase;
            uint64_t kValueMax = GetShapeWithDataType(aicoreParams_.l0aSize / DB_SIZE, inputParams_.aDtype) /
                                 std::max(tempMBase, tempNBase);
            if (usedCore > aicoreParams_.aicNum) {
                continue;
            }
            if ((usedCore > maxUsedCore || (usedCore == maxUsedCore && diff < maxDiff)) && kValueMax >= baseKAlignNum) {
                maxUsedCore = usedCore;
                maxDiff = diff;
                baseM = tempMBase;
                baseN = tempNBase;
                optimalFound = true;
            }
        }
    }
    return optimalFound;
}

void AdaptiveSlidingWindowTiling::AdjustBasicBlock4MmadS8S4(uint64_t oriBlock)
{
    uint64_t baseMAlignNum =
        inputParams_.transA ? GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum = L2_ALIGN_SIZE;
    uint64_t baseKAlignNum = (inputParams_.transA && !inputParams_.transB)
                                 ? GetShapeWithDataType(BASIC_BLOCK_SIZE_32, inputParams_.aDtype)
                                 : GetShapeWithDataType(L2_ALIGN_SIZE, inputParams_.aDtype);
    baseKAlignNum = std::min(baseKAlignNum, inputParams_.kSize);
    uint64_t mMaxtile = MathUtil::CeilDivision(inputParams_.mSize, baseMAlignNum);
    uint64_t nMaxtile = MathUtil::CeilDivision(inputParams_.nSize, baseNAlignNum);
    uint64_t tempBaseM = baseMAlignNum;
    uint64_t tempBaseN = baseNAlignNum;
    bool optimalFound = false;
    if (mMaxtile * nMaxtile >= oriBlock || (mMaxtile == 1UL) || (nMaxtile == 1UL)) {
        if (mMaxtile == 1UL || nMaxtile == 1UL) {
            // 当M或者N方向上最多只能用1core的场景下，另一个方向尽可能用满所有核
            tempBaseM = mMaxtile == 1UL
                            ? baseMAlignNum
                            : std::max(baseMAlignNum,
                                       ops::CeilAlign(MathUtil::CeilDivision(inputParams_.mSize, aicoreParams_.aicNum),
                                                      baseMAlignNum));
            tempBaseN = nMaxtile == 1UL
                            ? baseNAlignNum
                            : std::max(baseNAlignNum,
                                       ops::CeilAlign(MathUtil::CeilDivision(inputParams_.nSize, aicoreParams_.aicNum),
                                                      baseNAlignNum));
            optimalFound = true;
        } else {
            optimalFound = CalculateOptimalSplit(tempBaseM, tempBaseN, baseMAlignNum, baseNAlignNum, baseKAlignNum);
        }
        uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(inputParams_.kSize), baseKAlignNum);
        uint64_t kValueMax =
            GetShapeWithDataType(aicoreParams_.l0aSize / DB_SIZE, inputParams_.aDtype) / std::max(tempBaseM, tempBaseN);
        if (kValueMax >= baseKAlignNum && optimalFound) {
            adaptiveWin_.baseM = tempBaseM;
            adaptiveWin_.baseN = tempBaseN;
            kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
            adaptiveWin_.baseK = std::min(kValueAlign, kValueMax);
            adaptiveWin_.baseK = adaptiveWin_.baseK > BASEK_LIMIT ? adaptiveWin_.baseK / NUM_HALF : adaptiveWin_.baseK;
            uint64_t maxAlignSize = std::max(GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.aDtype),
                                             GetShapeWithDataType(CUBE_REDUCE_BLOCK, inputParams_.bDtype));
            adaptiveWin_.baseK = ops::CeilAlign(adaptiveWin_.baseK, maxAlignSize);
            adaptiveWin_.useTailWinLogic = false;
        }
    }
}

void AdaptiveSlidingWindowTiling::IsBFullLoad()
{
    // 下列情况下，B全载不生效:
    // 非supportMmadS8S4平台
    // 有batch
    // x1/x2矩阵不为INT8
    // y矩阵不为INT8/FLOAT16
    // 量化模式不为per-channel/per-tensor
    // 使能AL1 FULL
    // MC2场景
    if (!compileInfo_.supportMmadS8S4 || inputParams_.batchA != 1 || inputParams_.batchB != 1 ||
        inputParams_.aDtype != ge::DT_INT8 || inputParams_.bDtype != ge::DT_INT8 ||
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16) ||
        !(inputParams_.isPerTensor || inputParams_.isPerChannel) || isAFullLoad_ || isTilingOut_) {
        isBFullLoad_ = false;
        return;
    }
    singleCoreBSizeWithFullLoad_ =
        adaptiveWin_.baseN *
        (inputParams_.transB
             ? ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype), CUBE_REDUCE_BLOCK)
             : GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.bDtype));

    isBFullLoad_ =
        singleCoreBSizeWithFullLoad_ <= aicoreParams_.l1Size / NUM_HALF &&
        adaptiveWin_.mBlockCnt > adaptiveWin_.nBlockCnt &&
        (((adaptiveWin_.nBlockCnt == 1UL || adaptiveWin_.nBlockCnt == 2UL || adaptiveWin_.nBlockCnt == 4UL) &&
          aicoreParams_.aicNum >= adaptiveWin_.nBlockCnt && aicoreParams_.aicNum % adaptiveWin_.nBlockCnt == 0) ||
         adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum);
}

void AdaptiveSlidingWindowTiling::IsABFullLoad()
{
    // 下列情况下，AB全载不生效
    // 有batch
    // perblock pertoken
    // x1矩阵不为INT8
    // y矩阵不为INT8/FLOAT16
    // 前置约束：当x1为INT8, y为INT8/FLOAT16, nopertoken时，scale为uint64/int64, bias为int32，见
    // QuantBatchMatmulV3Checker::CheckScalesDtype()和QuantBatchMatmulV3Checker::CheckBiasDtype()
    if (inputParams_.batchA != 1 || inputParams_.batchB != 1 || isTilingOut_ || inputParams_.isPerBlock ||
        !compileInfo_.supportMmadS8S4 || inputParams_.isPertoken || inputParams_.aDtype != ge::DT_INT8 ||
        !(inputParams_.cDtype == ge::DT_INT8 || inputParams_.cDtype == ge::DT_FLOAT16)) {
        isABFullLoad_ = false;
        return;
    }
    // AB矩阵L1全载时的安全空间，当格式为NZ且k轴为内轴时，L1占用空间最大
    singleCoreASizeWithFullLoad_ =
        adaptiveWin_.baseM *
        (inputParams_.transA
             ? GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.aDtype)
             : ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.aDtype), CUBE_REDUCE_BLOCK));
    singleCoreBSizeWithFullLoad_ =
        adaptiveWin_.baseN *
        (inputParams_.transB
             ? ops::CeilAlign(GetSizeWithDataType(inputParams_.kSize, inputParams_.bDtype), CUBE_REDUCE_BLOCK)
             : GetSizeWithDataType(ops::CeilAlign(inputParams_.kSize, CUBE_BLOCK), inputParams_.bDtype));

    uint64_t biasDtypeSize = ge::GetSizeByDataType(inputParams_.biasDtype);
    uint64_t scaleDtypeSize = ge::GetSizeByDataType(inputParams_.scaleDtype);
    uint64_t singleCoreBiasSize = inputParams_.hasBias ? basicTiling_.baseN * biasDtypeSize : 0;
    uint64_t singleCoreScaleSize = inputParams_.isPerChannel ? basicTiling_.baseN * scaleDtypeSize : 0;
    uint64_t singleCoreABSize =  singleCoreASizeWithFullLoad_ + singleCoreBSizeWithFullLoad_;
    constexpr uint64_t AB_L1_SINGLE_LOAD_THRESHOLD = 64 * 1024UL;
    constexpr uint64_t AB_L1_BOTH_LOAD_THRESHOLD = 272 * 1024UL;

    // 条件是：L1空间大小允许，核数足够让每个block独占一个核, 且MTE2搬运量小于阈值(参照非量化mm)。
    isABFullLoad_ = singleCoreABSize + singleCoreBiasSize + singleCoreScaleSize <= aicoreParams_.l1Size &&
                    adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt <= aicoreParams_.aicNum &&
                    (singleCoreASizeWithFullLoad_ <= AB_L1_SINGLE_LOAD_THRESHOLD ||
                     singleCoreBSizeWithFullLoad_ <= AB_L1_SINGLE_LOAD_THRESHOLD) &&
                    singleCoreABSize <= AB_L1_BOTH_LOAD_THRESHOLD;
}

bool AdaptiveSlidingWindowTiling::CheckL1Size(uint64_t leftL1Size, uint64_t tempStepKa, uint64_t tempStepKb) const
{
    uint64_t singleCoreASize =
        tempStepKa *
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype) *
        DB_SIZE;

    uint64_t singleCoreBSize =
        tempStepKb *
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype) *
        DB_SIZE;

    if (isAFullLoad_) {
        // A矩阵L1全载时的安全空间，当格式为NZ且k轴为内轴时，L1占用空间最大
        singleCoreASize = singleCoreASizeWithFullLoad_;
    } else if (isBFullLoad_) {
        // B矩阵L1全载时的安全空间，当格式为NZ且k轴为内轴时，L1占用空间最大
        singleCoreBSize = singleCoreBSizeWithFullLoad_;
    }
    return leftL1Size >= singleCoreASize + singleCoreBSize;
}

void AdaptiveSlidingWindowTiling::AdjustStepK(uint64_t leftL1Size, uint64_t &tempStepKa, uint64_t &tempStepKb,
                                              bool isStepKa) const
{
    uint64_t oneBaseADataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype) *
        DB_SIZE;
    uint64_t oneBaseBDataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype) *
        DB_SIZE;
    // 当isStepKa true : 固定tempStepKb, 调整tempStepKa; false : 固定tempStepKa, 调整tempStepKb
    if (isStepKa) {
        uint64_t singleCoreBSize = tempStepKb * oneBaseBDataSize;
        if (isBFullLoad_) {
            // B矩阵L1全载时的安全空间，当格式为NZ且k轴为内轴时，L1占用空间最大
            singleCoreBSize = singleCoreBSizeWithFullLoad_;
        }
        if (leftL1Size < singleCoreBSize + oneBaseADataSize) {
            return;
        }
        tempStepKa = (leftL1Size - singleCoreBSize) / oneBaseADataSize;
    } else {
        uint64_t singleCoreASize = tempStepKa * oneBaseADataSize;
        if (isAFullLoad_) {
            // A矩阵L1全载时的安全空间，当格式为NZ且k轴为内轴时，L1占用空间最大
            singleCoreASize = singleCoreASizeWithFullLoad_;
        }
        if (leftL1Size < singleCoreASize + oneBaseBDataSize) {
            return;
        }
        tempStepKb = (leftL1Size - singleCoreASize) / oneBaseBDataSize;
    }
}

// 调整stepKa stepKb, 使其搬运量可以打满带宽
void AdaptiveSlidingWindowTiling::CarryDataSizePass(uint64_t leftL1Size, uint64_t maxStepK)
{
    uint64_t tempStepKa = static_cast<uint64_t>(basicTiling_.stepKa);
    uint64_t tempStepKb = static_cast<uint64_t>(basicTiling_.stepKb);
    // 计算a矩阵打满带宽的最小搬运量,并保证不超过shape k，不超过L1Size
    // 当走AL1全载模板时，无需考虑stepKa
    if (!isAFullLoad_) {
        tempStepKa = ops::CeilDiv(
            MIN_CARRY_DATA_SIZE_32K,
            GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype));
        tempStepKa = std::min(tempStepKa, maxStepK);
        if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb)) {
            AdjustStepK(leftL1Size, tempStepKa, tempStepKb, true);
        }
    }
    // 计算b矩阵打满带宽的最小搬运量,并保证不超过shape k，不超过L1Size
    // 当走BL1全载模板时，无需考虑stepKb
    if (!isBFullLoad_) {
        tempStepKb = ops::CeilDiv(
            MIN_CARRY_DATA_SIZE_32K,
            GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype));
        tempStepKb = std::min(tempStepKb, maxStepK);
        if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb)) {
            AdjustStepK(leftL1Size, tempStepKa, tempStepKb, false);
        }
    }

    // 该pass不可使stepKa或stepKb变小
    if (tempStepKa < static_cast<uint64_t>(basicTiling_.stepKa) ||
        tempStepKb < static_cast<uint64_t>(basicTiling_.stepKb)) {
        return;
    }
    basicTiling_.stepKa = static_cast<uint32_t>(tempStepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(tempStepKb);
    OP_LOGD(inputParams_.opName, "CarryDataSizePass: stepKa stepKb %u %u", basicTiling_.stepKa, basicTiling_.stepKb);
}

// 调整stepKa stepKb, 使其相等或满足倍数关系
void AdaptiveSlidingWindowTiling::BalanceStepKPass(uint64_t leftL1Size)
{
    // 当走AL1/BL1全载模板时，无需考虑stepKa和stepKb相等
    if (isAFullLoad_ || isBFullLoad_) {
        return;
    }
    // stepKa stepKb已相等，或者L1Size允许两者相等
    if (basicTiling_.stepKa == basicTiling_.stepKb) {
        return;
    }
    uint64_t biggerStepK = static_cast<uint64_t>(std::max(basicTiling_.stepKa, basicTiling_.stepKb));
    if (CheckL1Size(leftL1Size, biggerStepK, biggerStepK)) {
        basicTiling_.stepKa = biggerStepK;
        basicTiling_.stepKb = biggerStepK;
        return;
    }
    // 受限于L1Size，无法使stepKa stepKb相等，次优方案是调整两者呈倍数关系
    uint64_t tempStepKa = static_cast<uint64_t>(basicTiling_.stepKa);
    uint64_t tempStepKb = static_cast<uint64_t>(basicTiling_.stepKb);
    if (tempStepKa > tempStepKb && tempStepKa % tempStepKb == 0UL) {
        uint64_t bestStepKb = tempStepKa;
        // 在[tempStepKb，tempStepKa]区间内寻找满足L1Size且满足倍数关系的bestStepKb
        for (; bestStepKb >= tempStepKb; --bestStepKb) {
            if (tempStepKa % bestStepKb == 0UL && CheckL1Size(leftL1Size, tempStepKa, bestStepKb)) {
                break;
            }
        }
        tempStepKb = bestStepKb;
    } else if (tempStepKb > tempStepKa && tempStepKb % tempStepKa == 0UL) {
        uint64_t bestStepKa = tempStepKb;
        // 在[tempStepKa，tempStepKb]区间内寻找满足L1Size且满足倍数关系的bestStepKa
        for (; bestStepKa >= tempStepKa; --bestStepKa) {
            if (tempStepKb % bestStepKa == 0UL && CheckL1Size(leftL1Size, bestStepKa, tempStepKb)) {
                break;
            }
        }
        tempStepKa = bestStepKa;
    } else {
        return;
    }
    basicTiling_.stepKa = static_cast<uint32_t>(tempStepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(tempStepKb);
    OP_LOGD(inputParams_.opName, "BalanceStepKPass: stepKa stepKb %u %u", basicTiling_.stepKa, basicTiling_.stepKb);
}

// L1全载和非全载下的CacheLine对齐优化
void AdaptiveSlidingWindowTiling::L1FullLoadCacheLinePass(uint64_t &tempStepKa, uint64_t &tempStepKb,
                                                          uint64_t aCacheLine, uint64_t bCacheLine)
{
    // 当全载场景, 单核K除以baseK小于8时, stepK过大会导致第一次MTE2搬运量过大，导致性能下降。
    uint32_t maxStepKWithSmallCase = 4U * DB_SIZE;
    if (ops::CeilDiv(basicTiling_.singleCoreK, basicTiling_.baseK) < maxStepKWithSmallCase) {
        return;
    }
    // stepKa或stepKb是否可以乘以某个整数，从而满足CacheLine
    bool isEnableA = CACHE_LINE_512B % aCacheLine == 0;
    bool isEnableB = CACHE_LINE_512B % bCacheLine == 0;
    if (isAFullLoad_) {
        if (inputParams_.transB && isEnableB && (basicTiling_.baseN * bCacheLine <= FULL_LOAD_DATA_SIZE_64K)) {
            tempStepKb *= (CACHE_LINE_512B / bCacheLine);
        }
    } else {
        // 当全载场景, 单核K除以baseK小于8时, stepK过大会导致第一次MTE2搬运量过大，导致性能下降。
        if (ops::CeilDiv(basicTiling_.singleCoreK, basicTiling_.baseK) < MAX_STEPK_With_BL1_FULL) {
            return;
        }
        if (!inputParams_.transA && isEnableA && (basicTiling_.baseM * aCacheLine <= FULL_LOAD_DATA_SIZE_64K)) {
            tempStepKa *= (CACHE_LINE_512B / aCacheLine);
        }
    }
}

void AdaptiveSlidingWindowTiling::NONL1FullLoadCacheLinePass(uint64_t &tempStepKa, uint64_t &tempStepKb,
                                                             uint64_t aCacheLine, uint64_t bCacheLine)
{
    // stepKa或stepKb是否可以乘以某个整数，从而满足CacheLine
    bool isEnableA = CACHE_LINE_512B % aCacheLine == 0;
    bool isEnableB = CACHE_LINE_512B % bCacheLine == 0;
    uint64_t aDataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseM) * basicTiling_.baseK, inputParams_.aDtype) *
        tempStepKa;
    uint64_t bDataSize =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseN) * basicTiling_.baseK, inputParams_.bDtype) *
        tempStepKb;
    uint64_t factor = 1UL;
    if (inputParams_.transA && inputParams_.transB) {
        factor = isEnableB ? CACHE_LINE_512B / bCacheLine : 1UL;
    } else if (!inputParams_.transA && !inputParams_.transB) {
        factor = isEnableA ? CACHE_LINE_512B / aCacheLine : 1UL;
    } else {
        // 使搬运量和内轴都较大的矩阵满足cacheline对齐，不必过于追求a，b矩阵都对齐，会导致搬运量过大，性能负收益。
        if (aDataSize > bDataSize && aCacheLine > bCacheLine && isEnableA) {
            factor = CACHE_LINE_512B / aCacheLine;
        } else if (aDataSize < bDataSize && aCacheLine < bCacheLine && isEnableB) {
            factor = CACHE_LINE_512B / bCacheLine;
        }
    }
    tempStepKa *= factor;
    tempStepKb *= factor;
}

// 调整stepKa stepKb, 使其内轴对齐cacheline
void AdaptiveSlidingWindowTiling::PostCacheLinePass(uint64_t leftL1Size, uint64_t maxStepK)
{
    // transA true, transB false时，cacheLine和stepKa,stepKb无关
    if (inputParams_.transA && !inputParams_.transB) {
        return;
    }
    uint64_t tempStepKa = static_cast<uint64_t>(basicTiling_.stepKa);
    uint64_t tempStepKb = static_cast<uint64_t>(basicTiling_.stepKb);
    uint64_t aCacheLine =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseK), inputParams_.aDtype) * tempStepKa;
    uint64_t bCacheLine =
        GetSizeWithDataType(static_cast<uint64_t>(basicTiling_.baseK), inputParams_.bDtype) * tempStepKb;

    // 多trans组合并考虑全载，调整内轴512B对齐
    if (isAFullLoad_ || isBFullLoad_) {
        L1FullLoadCacheLinePass(tempStepKa, tempStepKb, aCacheLine, bCacheLine);
    } else {
        NONL1FullLoadCacheLinePass(tempStepKa, tempStepKb, aCacheLine, bCacheLine);
    }

    if (!CheckL1Size(leftL1Size, tempStepKa, tempStepKb) || tempStepKa > maxStepK || tempStepKb > maxStepK) {
        return;
    }
    basicTiling_.stepKa = static_cast<uint32_t>(tempStepKa);
    basicTiling_.stepKb = static_cast<uint32_t>(tempStepKb);
    OP_LOGD(inputParams_.opName, "PostCacheLinePass: stepKa stepKb %u %u", basicTiling_.stepKa, basicTiling_.stepKb);
}

void AdaptiveSlidingWindowTiling::CalL1TilingDepth4MmadS8S4(uint64_t leftL1Size)
{
    basicTiling_.stepKa = 1U;
    basicTiling_.stepKb = 1U;
    basicTiling_.depthA1 = 1U;
    basicTiling_.depthB1 = 1U;
    // 当ABL1全载时，stepKa和stepKb无作用，默认1
    if (isABFullLoad_) {
        return;
    }

    // 计算shape k约束下的stepKa或stepKb的最大值
    uint64_t maxStepK = ops::CeilDiv(inputParams_.kSize, static_cast<uint64_t>(basicTiling_.baseK));

    // 计算stepKa和stepKb
    CarryDataSizePass(leftL1Size, maxStepK);
    BalanceStepKPass(leftL1Size);
    PostCacheLinePass(leftL1Size, maxStepK);

    // 计算depthA1和depthB1
    basicTiling_.stepKa = isAFullLoad_ ? 1U : basicTiling_.stepKa;
    basicTiling_.depthA1 = isAFullLoad_ ? basicTiling_.stepKa : basicTiling_.stepKa * DB_SIZE;
    basicTiling_.stepKb = isBFullLoad_ ? 1U : basicTiling_.stepKb;
    basicTiling_.depthB1 = isBFullLoad_ ? basicTiling_.stepKb:  basicTiling_.stepKb * DB_SIZE;
}

void AdaptiveSlidingWindowTiling::CalcTailBasicBlockBfullLoad()
{
    // B全载全载时, n方向不切分
    adaptiveWin_.nTailTile = 1UL;

    uint64_t mTile = 1UL;
    constexpr uint64_t MIN_BASEN_PER_TILE = 16UL;  // 尾窗口切分后M方向不少于16

    if (adaptiveWin_.tailWinBlockCnt != 0UL) {
        while (CalUsedCoreNum((mTile + 1UL), adaptiveWin_.nTailTile) <= aicoreParams_.aicNum &&
               adaptiveWin_.baseM / (mTile + 1UL) >= MIN_BASEN_PER_TILE) {
            mTile += 1UL;
        }
    }
    adaptiveWin_.mTailTile = mTile;
}

}  // namespace optiling
