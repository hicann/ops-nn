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
 * \file dual_level_quant_batch_matmul_adaptive_sliding_window_tiling.cpp
 * \brief
 */

#include "dual_level_quant_batch_matmul_adaptive_sliding_window_tiling.h"
#include "../../op_kernel/dual_level_quant_batch_matmul_tiling_key.h"
#include "../../op_kernel/dual_level_quant_batch_matmul_tiling_data.h"
#include "matmul/common/op_host/math_util.h"
#include "error_util.h"

using namespace platform_ascendc;
using namespace optiling::tool;

namespace {
template <typename T>
T GetShapeWithDataType(T size, ge::DataType dtype)
{
    if (dtype == ge::DT_INT4 || dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2) {
        return size + size;
    } else {
        return size / static_cast<T>(ge::GetSizeByDataType(dtype));
    }
}

template <typename T>
T GetSizeWithDataType(T shape, ge::DataType dtype)
{
    if (dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4) {
        return (shape + 1) >> 1;
    } else {
        return shape * static_cast<T>(ge::GetSizeByDataType(dtype));
    }
}
} // namespace

namespace optiling {
namespace dual_level_quant_batch_matmul {
constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t L2_ALIGN_SIZE = 128;
constexpr uint64_t MICROSCALE_GROUP_SIZE = 32;
constexpr uint64_t MXFP_MULTI_BASE_SIZE = 2;
constexpr uint64_t MXFP_DIVISOR_SIZE = MICROSCALE_GROUP_SIZE * MXFP_MULTI_BASE_SIZE;
constexpr uint64_t L0_SPLIT_NUM = 2;
constexpr uint64_t BASEM_BASEN_RATIO = 2;
constexpr uint32_t DOUBLE_CORE_NUM = 2;

constexpr uint32_t BASIC_BLOCK_SIZE_16 = 16;
constexpr uint32_t BASIC_BLOCK_SIZE_32 = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_1024 = 1024;

constexpr uint32_t CORE_NUM_HALF = 2;
constexpr uint32_t DB_SIZE = 2;

constexpr uint64_t WINDOW_LEN = 4;                // v100 的滑窗大小
constexpr uint64_t MTE2_CACHELINE_SIZE = 256;     // MTE2 的最小 CACHELINE
constexpr uint64_t LOAD_BALANCE_THRESHOLD = 1792; // 内轴非对齐时，执行负载均衡的阈值

constexpr uint64_t BASEK_LIMIT = 4095;

DualLevelQuantBatchMatmulTilingASW::DualLevelQuantBatchMatmulTilingASW(gert::TilingContext* context)
    : DualLevelQuantBatchMatmulBaseTiling(context)
{
    if (!isCompileInfoInit) {
        InitCompileInfo();
    }
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::GetShapeAttrsInfo()
{
    tilingDataSize_ = sizeof(DualLevelQuantBatchMatmulBasicTilingData);
    return DualLevelQuantBatchMatmulBaseTiling::GetShapeAttrsInfo();
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::GetPlatformInfo()
{
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, opName_, "GetPlatformInfo fail");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of adaptive sliding window tiling strategy.");
    OP_TILING_CHECK(
        InstantiateTilingData() == ge::GRAPH_FAILED,
        CUBE_INNER_ERR_REPORT(opName_, "unable to get pointer of tiling data"), return ge::GRAPH_FAILED);

    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(opName_, "DoOpTiling fail");
        return ge::GRAPH_FAILED;
    }
    LoadBalanceDataReset();
    if (!OptimizeEdgeBasicBlock()) {
        OP_LOGE(opName_, "OptimizeEdgeBasicBlock fail");
        return ge::GRAPH_FAILED;
    }
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t DualLevelQuantBatchMatmulTilingASW::GetTilingKey() const
{
    uint64_t socVersionType = DLQBMM_SOC_RESERVERD;
    uint64_t subSocVersionType = DLQBMM_DEFAULT;
    uint64_t templateCustom = DLQBMM_TEMPLATE_CUBEBOUND;
    uint64_t level1QuantType = DLQBMM_QUANT_TYPE_MX;
    uint64_t level0QuantType = DLQBMM_QUANT_TYPE_PER_GROUP;
    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isWeightNz = matmulInfoPtr_->x2Format == ge::FORMAT_FRACTAL_NZ;
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        socVersionType, subSocVersionType, templateCustom, level1QuantType, level0QuantType, transA, transB, hasBias,
        isWeightNz);
    return tilingKey;
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::GetWorkspaceSize()
{
    workspaceSize_ = matmulInfoPtr_->libApiWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize_);
    OP_TILING_CHECK(
        tilingDataSize_ % sizeof(uint64_t) != 0,
        CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize_),
        return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(
        context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
        reinterpret_cast<void*>(tilingData_.get()), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    size_t* workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OPS_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

void DualLevelQuantBatchMatmulTilingASW::LoadBalanceDataReset()
{
    adaptiveWin_.mBaseTailSplitCnt = 1UL;
    adaptiveWin_.nBaseTailSplitCnt = 1UL;
    adaptiveWin_.mTailMain = 0UL;
    adaptiveWin_.nTailMain = 0UL;
}

ge::graphStatus DualLevelQuantBatchMatmulTilingASW::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        try {
            // make_unique不会返回空指针，只会返回异常，无需在后面加空指针校验
            tilingData_ = std::make_unique<DualLevelQuantBatchMatmulBasicTilingData>();
        } catch (std::bad_alloc&) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(
        context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(
            opName_, "tiling data capacity %zu < actual tiling data size %zu",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool DualLevelQuantBatchMatmulTilingASW::AnalyseSlidingWinInfo()
{
    if (!CalcBasicBlock()) {
        OP_LOGE(opName_, "inappropriate basicBlock");
        return false;
    }
    adaptiveWin_.mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    adaptiveWin_.nBlockCnt = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    adaptiveWin_.totalBlockCnt = adaptiveWin_.mBlockCnt * adaptiveWin_.nBlockCnt;
    adaptiveWin_.mTail = matmulInfoPtr_->mSize - (adaptiveWin_.mBlockCnt - 1) * adaptiveWin_.baseM;
    adaptiveWin_.nTail = matmulInfoPtr_->nSize - (adaptiveWin_.nBlockCnt - 1) * adaptiveWin_.baseN;
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, static_cast<uint64_t>(compileInfo_.aicNum));
    adaptiveWin_.tailWinBlockCnt = (adaptiveWin_.totalBlockCnt) % compileInfo_.aicNum;

    if (adaptiveWin_.useTailWinLogic) {
        CalcTailBasicBlock();
    }
    return true;
}

// baseM/baseN/baseK对齐原则
// 原则1：L1上x1、x2都是NZ排布，转置可选          须满足内轴32B对齐，外轴16对齐
// 原则2：L0上x1是Zz排布，x2是Zn排布，非转置      须满足baseM、baseN关于16对齐，baseK关于32B对齐
// 理论上需要同时满足原则1和原则2,但全量化的x1Dtype可取{8bit, 4bit}, x2Dtype可取{8bit, 4bit}
// sizeof(x1Dtype) <= 1, sizeof(x2Dtype) <= 1
// baseM满足原则1时，可能是关于32B对齐或关于16对齐，间接满足原则2
// baseN满足原则1时，可能是关于32B对齐或关于16对齐，间接满足原则2
// baseK满足32B对齐时，同时满足原则1和原则2
bool DualLevelQuantBatchMatmulTilingASW::CalcBasicBlock()
{
    // baseM=256, baseN=256, baseK=512
    adaptiveWin_.baseM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    adaptiveWin_.baseM =
        ops::CeilAlign(adaptiveWin_.baseM, GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->x1Dtype));
    adaptiveWin_.baseN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_256));
    adaptiveWin_.baseN = ops::CeilAlign(adaptiveWin_.baseN, CUBE_BLOCK);
    adaptiveWin_.baseK =
        ops::CeilAlign(std::min(matmulInfoPtr_->kSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_512)), MXFP_DIVISOR_SIZE);

    uint64_t oriBlock = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM) *
                        ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    bool isSmallBlock = oriBlock < compileInfo_.aicNum;
    if (isSmallBlock) {
        AdjustBasicBlock();
    }
    return true;
}

template <typename F>
void AdjustMNCoreRatio(
    uint64_t& xCore, uint64_t& yCore, uint64_t& baseX, uint64_t& baseY, uint64_t baseAlignNum, uint64_t aicNum,
    F&& updateFunc)
{
    while (baseX >= baseY * BASEM_BASEN_RATIO && xCore < aicNum / CORE_NUM_HALF && baseX != baseAlignNum) {
        xCore = xCore * DOUBLE_CORE_NUM;
        yCore = aicNum / xCore;
        updateFunc();
    }
}

void DualLevelQuantBatchMatmulTilingASW::AdjustBasicBlock()
{
    uint64_t baseMAlignNum =
        matmulInfoPtr_->transA ? GetShapeWithDataType(L2_ALIGN_SIZE, matmulInfoPtr_->x1Dtype) : CUBE_BLOCK;
    uint64_t baseNAlignNum =
        matmulInfoPtr_->transB ? CUBE_BLOCK : GetShapeWithDataType(L2_ALIGN_SIZE, matmulInfoPtr_->x1Dtype);
    uint64_t baseKAlignNum = (matmulInfoPtr_->transA && !matmulInfoPtr_->transB) ?
                                 GetShapeWithDataType(BASIC_BLOCK_SIZE_32, matmulInfoPtr_->x1Dtype) :
                                 GetShapeWithDataType(L2_ALIGN_SIZE, matmulInfoPtr_->x1Dtype);
    uint64_t mMaxtile = ops::CeilDiv(matmulInfoPtr_->mSize, baseMAlignNum);
    uint64_t nMaxtile = ops::CeilDiv(matmulInfoPtr_->nSize, baseNAlignNum);
    uint64_t tempBaseM = adaptiveWin_.baseM;
    uint64_t tempBaseN = adaptiveWin_.baseN;
    if (mMaxtile * nMaxtile >= compileInfo_.aicNum || (!matmulInfoPtr_->transA && matmulInfoPtr_->transB)) {
        uint64_t mCore = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
        uint64_t nCore = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
        if (mMaxtile < nMaxtile || (mMaxtile == nMaxtile && baseNAlignNum == CUBE_BLOCK)) {
            tempBaseM = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->mSize, mCore), baseMAlignNum);
            mCore = ops::CeilDiv(matmulInfoPtr_->mSize, tempBaseM);
            nCore = compileInfo_.aicNum / mCore;
            tempBaseN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nCore), baseNAlignNum);
        } else {
            tempBaseN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nCore), baseNAlignNum);
            nCore = ops::CeilDiv(matmulInfoPtr_->nSize, tempBaseN);
            mCore = compileInfo_.aicNum / nCore;
            tempBaseM = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->mSize, mCore), baseMAlignNum);
        }

        auto updateFunc = [&, this]() {
            tempBaseM = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->mSize, mCore), baseMAlignNum);
            tempBaseN = ops::CeilAlign(ops::CeilDiv(matmulInfoPtr_->nSize, nCore), baseNAlignNum);
            mCore = ops::CeilDiv(matmulInfoPtr_->mSize, static_cast<uint64_t>(tempBaseM));
            nCore = ops::CeilDiv(matmulInfoPtr_->nSize, static_cast<uint64_t>(tempBaseN));
        };
        AdjustMNCoreRatio(
            nCore, mCore, tempBaseN, tempBaseM, baseNAlignNum, static_cast<uint64_t>(compileInfo_.aicNum), updateFunc);
        AdjustMNCoreRatio(
            mCore, nCore, tempBaseM, tempBaseN, baseMAlignNum, static_cast<uint64_t>(compileInfo_.aicNum), updateFunc);

        uint64_t kValueAlign = ops::CeilAlign(static_cast<uint64_t>(matmulInfoPtr_->kSize), baseKAlignNum);
        uint64_t kValueMax = GetShapeWithDataType(compileInfo_.l0aSize / DB_SIZE, matmulInfoPtr_->x1Dtype) /
                             std::max(tempBaseM, tempBaseN) / L0_SPLIT_NUM;
        if (kValueMax >= baseKAlignNum) {
            adaptiveWin_.baseM = tempBaseM;
            adaptiveWin_.baseN = tempBaseN;
            kValueMax = ops::FloorAlign(kValueMax, baseKAlignNum);
            adaptiveWin_.baseK = std::min(kValueAlign, kValueMax);
            adaptiveWin_.baseK =
                adaptiveWin_.baseK > BASEK_LIMIT ? adaptiveWin_.baseK / CORE_NUM_HALF : adaptiveWin_.baseK;
            adaptiveWin_.useTailWinLogic = false;
        }
    }
}

void DualLevelQuantBatchMatmulTilingASW::CalcTailBasicBlock()
{
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    auto& preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto& secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;

    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    while (CalUsedCoreNum(preSplit + 1, secSplit) <= compileInfo_.aicNum) {
        preSplit += 1UL;
        if (!IsInvalidWeightNzTailSplit(preSplit, true)) {
            preSplitValid = preSplit;
        }

        if (CalUsedCoreNum(preSplit, secSplit + 1) <= compileInfo_.aicNum) {
            secSplit += 1UL;
            if (!IsInvalidWeightNzTailSplit(secSplit, false)) {
                secSplitValid = secSplit;
            }
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

uint32_t DualLevelQuantBatchMatmulTilingASW::CalUsedCoreNum(uint32_t mTile, uint32_t nTile)
{
    return mTile * nTile * static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt);
}

uint32_t DualLevelQuantBatchMatmulTilingASW::CalBlockDim()
{
    if (adaptiveWin_.totalWinCnt > 1UL || adaptiveWin_.tailWinBlockCnt == 0UL) {
        return compileInfo_.aicNum;
    }

    return static_cast<uint32_t>(adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile);
}

bool DualLevelQuantBatchMatmulTilingASW::IsInvalidWeightNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (matmulInfoPtr_->x2Format != ge::FORMAT_FRACTAL_NZ ||
        (((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
          (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)))) {
        return false;
    }
    uint64_t tailN = adaptiveWin_.baseN / splitCnt;
    return tailN % GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->x2Dtype) != 0;
}

bool DualLevelQuantBatchMatmulTilingASW::OptimizeEdgeBasicBlock()
{
    uint64_t mCore = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t nCore = ops::CeilDiv(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    if (mCore == 1UL || nCore == 1UL) {
        return true;
    }

    uint64_t mBaseTail = static_cast<uint64_t>(matmulInfoPtr_->mSize % adaptiveWin_.baseM);
    uint64_t nBaseTail = static_cast<uint64_t>(matmulInfoPtr_->nSize % adaptiveWin_.baseN);
    bool isMxfp4 = (matmulInfoPtr_->x1Dtype == ge::DT_FLOAT4_E2M1 || matmulInfoPtr_->x1Dtype == ge::DT_FLOAT4_E1M2) &&
                   matmulInfoPtr_->x1Level1ScaleDtype == ge::DT_FLOAT8_E8M0 &&
                   matmulInfoPtr_->level1GroupSize == MICROSCALE_GROUP_SIZE;
    bool balanceAfterFixp = matmulInfoPtr_->kSize < static_cast<uint64_t>(BASIC_BLOCK_SIZE_1024);
    bool isInnerAxisAlign =
        GetSizeWithDataType(matmulInfoPtr_->kSize, matmulInfoPtr_->x1Dtype) % MTE2_CACHELINE_SIZE == 0UL;
    if (mBaseTail > 0UL && !matmulInfoPtr_->transA &&
        (isInnerAxisAlign || (matmulInfoPtr_->mSize >= LOAD_BALANCE_THRESHOLD && !isMxfp4))) {
        if (!GetOuterMAxisTailCnt(adaptiveWin_.mBaseTailSplitCnt, adaptiveWin_.mTailMain)) {
            return false;
        };
    }
    if (nBaseTail > 0UL && matmulInfoPtr_->transB && !balanceAfterFixp &&
        (isInnerAxisAlign || (matmulInfoPtr_->nSize >= LOAD_BALANCE_THRESHOLD))) {
        if (!GetOuterNAxisTailCnt(adaptiveWin_.nBaseTailSplitCnt, adaptiveWin_.nTailMain)) {
            return false;
        };
    }
    return true;
}

bool DualLevelQuantBatchMatmulTilingASW::GetOuterMAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    OP_TILING_CHECK(
        matmulInfoPtr_->mSize == 0UL, CUBE_INNER_ERR_REPORT(opName_, "Input size of the M-axis is zero."),
        return false);
    uint64_t mCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t mTailSize = matmulInfoPtr_->mSize % adaptiveWin_.baseM;
    uint64_t baseTailCntMax = std::min((adaptiveWin_.baseM - mTailSize) / BASIC_BLOCK_SIZE_16, mCnt);
    uint64_t windowSize = std::min(WINDOW_LEN, mCnt);
    uint64_t mainWindowNum = mCnt / windowSize - 1UL;
    uint64_t tailWindowSize = mCnt - mainWindowNum * windowSize;
    uint64_t perfRes = (mainWindowNum + 1UL) * adaptiveWin_.baseM;
    uint64_t mergeWindowNum = 1UL;

    for (uint64_t mergeLen = tailWindowSize - 1UL; mergeLen < baseTailCntMax;
         mergeLen += windowSize, ++mergeWindowNum) {
        uint64_t newTailMain = ops::CeilAlign(
            ops::CeilDiv((mergeLen * adaptiveWin_.baseM + mTailSize), mergeLen + 1UL),
            static_cast<uint64_t>(BASIC_BLOCK_SIZE_16));
        OP_TILING_CHECK(
            mainWindowNum + 1UL < mergeWindowNum,
            CUBE_INNER_ERR_REPORT(
                opName_, "Subtraction underflow: mainWindowNum(%lu) + 1UL - mergeWindowNum(%lu).", mainWindowNum,
                mergeWindowNum),
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

bool DualLevelQuantBatchMatmulTilingASW::GetOuterNAxisTailCnt(uint64_t& baseTailSplitCnt, uint64_t& tailMain)
{
    uint64_t baseN = adaptiveWin_.baseN;
    uint64_t nCnt = ops::CeilDiv(matmulInfoPtr_->nSize, baseN);
    uint64_t mCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t nTail = matmulInfoPtr_->nSize % baseN;
    uint64_t totalWindows = ops::CeilDiv<uint64_t>(nCnt * mCnt, compileInfo_.aicNum);

    OP_TILING_CHECK(
        nCnt == 0UL,
        CUBE_INNER_ERR_REPORT(
            opName_,
            "Subtraction underflow: nCnt(%lu) - 1UL and \
the divisor is zero: WINDOW_LEN %% nCnt.",
            nCnt),
        return false);
    uint64_t mainWindows =
        ops::CeilDiv<uint64_t>((nCnt - 1UL) * mCnt + mCnt % compileInfo_.aicNum, compileInfo_.aicNum);

    OP_TILING_CHECK(
        compileInfo_.aicNum == 0UL, CUBE_INNER_ERR_REPORT(opName_, "The number of enabled Cube cores is 0."),
        return false);
    if (nCnt * mCnt <= compileInfo_.aicNum ||
        (mCnt % compileInfo_.aicNum == 0UL && (nCnt % WINDOW_LEN == 0UL || WINDOW_LEN % nCnt == 0UL))) {
        mainWindows = totalWindows;
    }
    OP_TILING_CHECK(
        totalWindows < mainWindows,
        CUBE_INNER_ERR_REPORT(
            opName_, "Subtraction underflow: totalWindows(%lu) - mainWindows(%lu).", totalWindows, mainWindows),
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

uint64_t DualLevelQuantBatchMatmulTilingASW::CalculateCurrentPerf(
    uint64_t mergeLen, uint64_t nTail, uint64_t mCnt, uint64_t nCnt, uint64_t& newTailMain)
{
    uint64_t totalWindows = ops::CeilDiv<uint64_t>(nCnt * mCnt, compileInfo_.aicNum);
    newTailMain = ops::CeilAlign(
        ops::CeilDiv((mergeLen * adaptiveWin_.baseN + nTail), mergeLen + 1UL),
        static_cast<uint64_t>(BASIC_BLOCK_SIZE_16));
    OP_TILING_CHECK(
        adaptiveWin_.baseN < newTailMain,
        CUBE_INNER_ERR_REPORT(
            opName_, "Subtraction underflow: adaptiveWin_.baseN(%lu) - newTailMain(%lu).", adaptiveWin_.baseN,
            newTailMain),
        return static_cast<uint64_t>(-1));
    uint64_t newTailLast = mergeLen * (adaptiveWin_.baseN - newTailMain) + nTail;
    uint64_t newMainRound = 0UL;
    uint64_t newTailRound = 0UL;

    if (mergeLen < nCnt - 1UL) {
        newMainRound = ops::CeilDiv<uint64_t>(
            (nCnt - 1UL - mergeLen) * mCnt + (mergeLen + 1UL) * mCnt % compileInfo_.aicNum, compileInfo_.aicNum);
    }
    if (mergeLen > 0UL) {
        OP_TILING_CHECK(
            totalWindows < newMainRound,
            CUBE_INNER_ERR_REPORT(
                opName_, "Subtraction underflow: totalWindows(%lu) - newMainRound(%lu).", totalWindows, newMainRound),
            return static_cast<uint64_t>(-1));
        newTailRound = std::min(
            ops::CeilDiv<uint64_t>(mergeLen * mCnt + mCnt % compileInfo_.aicNum, compileInfo_.aicNum),
            totalWindows - newMainRound);
    }

    OP_TILING_CHECK(
        totalWindows < newMainRound + newTailRound,
        CUBE_INNER_ERR_REPORT(
            opName_, "Subtraction underflow: totalWindows(%lu) - newMainRound(%lu) - newTailRound(%lu).", totalWindows,
            newMainRound, newTailRound),
        return static_cast<uint64_t>(-1));
    return newMainRound * adaptiveWin_.baseN + newTailRound * newTailMain +
           (totalWindows - newMainRound - newTailRound) * newTailLast;
}

void DualLevelQuantBatchMatmulTilingASW::SetTilingData()
{
    usedCoreNum = CalBlockDim();

    tilingData_->l1BufferNum = 2;
    tilingData_->hasBias = matmulInfoPtr_->hasBias;
    tilingData_->l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT;
    tilingData_->usedCoreNum = usedCoreNum;
    tilingData_->mSize = matmulInfoPtr_->mSize;
    tilingData_->nSize = matmulInfoPtr_->nSize;
    tilingData_->kSize = matmulInfoPtr_->kSize;
    tilingData_->mL1Size = adaptiveWin_.baseM;
    tilingData_->nL1Size = adaptiveWin_.baseN;
    tilingData_->kL1Size = adaptiveWin_.baseK;
    tilingData_->level0GroupSize = matmulInfoPtr_->level0GroupSize;

    tilingData_->mTailTile = adaptiveWin_.mTailTile;
    tilingData_->nTailTile = adaptiveWin_.nTailTile;
    tilingData_->mBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.mBaseTailSplitCnt);
    tilingData_->nBaseTailSplitCnt = static_cast<uint32_t>(adaptiveWin_.nBaseTailSplitCnt);
    tilingData_->mTailMain = static_cast<uint32_t>(adaptiveWin_.mTailMain);
    tilingData_->nTailMain = static_cast<uint32_t>(adaptiveWin_.nTailMain);

    OP_LOGD(opName_, "coreNum: %u", usedCoreNum);
}
} // namespace dual_level_quant_batch_matmul
} // namespace optiling