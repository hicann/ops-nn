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
 * \file weight_quant_batch_matmul_v2_adaptive_sliding_window_general_tiling.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_adaptive_sliding_window_general_tiling.h"
#include "common/inc/op_host/op_const_def_nn.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "tiling/platform/platform_ascendc.h"
#include "matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_data.h"
#include "../../../op_kernel/arch35/weight_quant_batch_matmul_v2_arch35_tiling_key.h"

using namespace platform_ascendc;

namespace optiling {
namespace weight_quant_batch_matmul_v2 {

constexpr uint64_t CUBE_BLOCK = 16;
constexpr uint64_t L1_ALIGN_SIZE = 32;
constexpr uint64_t CUBE_BOUND_BASE_K = 64UL;
constexpr uint64_t MEGABYTE = 1024 * 1024;
constexpr uint64_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint64_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint64_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint64_t DB_SIZE = 2;
constexpr uint64_t SLIDING_WINOW_SHORTER_EDGE = 4;
constexpr uint64_t SLIDING_WINOW_LONGER_EDGE = 16;
constexpr uint64_t MAX_PERF_STEPKB = 4;
constexpr uint64_t MIN_BASIC_BLOCK_N = 32UL;
constexpr uint64_t BYTE_BITS = 8UL;
constexpr uint64_t L1_SIZE = 524288UL;
constexpr uint64_t ASW_GENERAL_PRIORITY = 13;
constexpr uint64_t L0B_SIZE = 65536UL;

ge::graphStatus WeightQuantBatchMatmulV2TilingASWGeneral::DoOpTiling()
{
    OP_LOGD(opName_, "DoOpTiling of adaptive sliding window tiling strategy.");
    OP_TILING_CHECK(InstantiateTilingData() == ge::GRAPH_FAILED,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "unable to get pointer of tiling data"),
                    return ge::GRAPH_FAILED);

    if (!AnalyseSlidingWinInfo()) {
        OP_LOGE(opName_, "DoOpTiling fail");
        return ge::GRAPH_FAILED;
    }
    CalL1Tiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWGeneral::InstantiateTilingData()
{
    if (tilingData_ == nullptr) {
        try {
            tilingData_ = std::make_unique<wqbmmv2_tiling::WeightQuantBatchMatmulV2ASWTilingData>();
        } catch (std::bad_alloc&) {
            OP_LOGE(opName_, "tiling data memory allocation failed");
            return ge::GRAPH_FAILED;
        }
    }
    OP_TILING_CHECK(tilingData_ == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to instantiate tilingData"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName_, "tiling data capacity %zu < actual tiling data size %zu",
                                                    context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingASWGeneral::AnalyseSlidingWinInfo()
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
    adaptiveWin_.totalWinCnt = ops::CeilDiv(adaptiveWin_.totalBlockCnt, static_cast<uint64_t>(compileInfoPtr_->aicNum));
    if (adaptiveWin_.mBlockCnt <= adaptiveWin_.nBlockCnt) {
        adaptiveWin_.singleWinM = SLIDING_WINOW_SHORTER_EDGE;
        adaptiveWin_.singleWinN = SLIDING_WINOW_LONGER_EDGE;
    } else {
        adaptiveWin_.singleWinM = SLIDING_WINOW_LONGER_EDGE;
        adaptiveWin_.singleWinN = SLIDING_WINOW_SHORTER_EDGE;
    }
    if (adaptiveWin_.mBlockCnt % adaptiveWin_.singleWinM == 0) {
        adaptiveWin_.mainRow = adaptiveWin_.mBlockCnt / adaptiveWin_.singleWinM;
    } else {
        adaptiveWin_.mainRow = adaptiveWin_.mBlockCnt < adaptiveWin_.singleWinM ?
                                   0 :
                                   adaptiveWin_.mBlockCnt / adaptiveWin_.singleWinM - 1;
    }
    adaptiveWin_.tailWinBlockCnt = (adaptiveWin_.totalBlockCnt) % compileInfoPtr_->aicNum;
    if (adaptiveWin_.useTailWinLogic) {
        CalcTailBasicBlock();
    }
    return true;
}

void WeightQuantBatchMatmulV2TilingASWGeneral::CalculateBaseN()
{
    uint64_t mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    uint64_t baseNAlignSize = matmulInfoPtr_->transB ? CUBE_BLOCK :
                                                       GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype);
    uint64_t baseNMax = std::min(
        BASIC_BLOCK_SIZE_256 - baseNAlignSize,
        ops::CeilAlign(L0B_SIZE / (DB_SIZE * ge::GetSizeByDataType(matmulInfoPtr_->aDtype) * BASIC_BLOCK_SIZE_128),
                       CUBE_BLOCK));
    uint64_t edgeLengthM = matmulInfoPtr_->mSize > matmulInfoPtr_->nSize ? SLIDING_WINOW_LONGER_EDGE :
                                                                           SLIDING_WINOW_SHORTER_EDGE;
    if (matmulInfoPtr_->transB) {
        adaptiveWin_.baseN = std::max(
            MIN_BASIC_BLOCK_N,
            ops::CeilDiv(matmulInfoPtr_->nSize,
                         static_cast<uint64_t>(compileInfoPtr_->aicNum / std::min(mBlockCnt, edgeLengthM))));
        adaptiveWin_.baseN = std::min(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
        if (adaptiveWin_.baseN > baseNMax) {
            if (adaptiveWin_.baseN % baseNMax < baseNMax / 2) { // 取模小于baseN的1/2说明最后会有一个不好的尾块
                adaptiveWin_.baseN = adaptiveWin_.baseN / (ops::CeilDiv(adaptiveWin_.baseN, baseNMax));
            } else {
                adaptiveWin_.baseN = baseNMax;
            }
        }
        adaptiveWin_.baseN = ops::CeilAlign(adaptiveWin_.baseN, baseNAlignSize);
    } else {
        adaptiveWin_.baseN = BASIC_BLOCK_SIZE_128; // baseN用于分配内轴空间，不和nSize取最小值
    }
}

bool WeightQuantBatchMatmulV2TilingASWGeneral::CalcBasicBlock()
{
    adaptiveWin_.baseM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    uint64_t mBlockCnt = ops::CeilDiv(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    adaptiveWin_.baseM = ops::CeilDiv(matmulInfoPtr_->mSize, mBlockCnt);
    adaptiveWin_.baseM = !matmulInfoPtr_->transA ?
                             ops::CeilAlign(adaptiveWin_.baseM, CUBE_BLOCK) :
                             ops::CeilAlign(adaptiveWin_.baseM,
                                            GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->aDtype));

    CalculateBaseN();
    if (matmulInfoPtr_->transB) {
        adaptiveWin_.baseK = BASIC_BLOCK_SIZE_128;
    } else {
        adaptiveWin_.baseK = std::min(ops::CeilAlign(matmulInfoPtr_->kSize, CUBE_BLOCK),
                                      static_cast<uint64_t>(BASIC_BLOCK_SIZE_128));
    }
    CalStepKs();
    adaptiveWin_.enablePreAlloc = !ClosePreAllocCond();
    return true;
}

constexpr bool WeightQuantBatchMatmulV2TilingASWGeneral::Is4BitInput(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT4_E2M1 || dtype == ge::DT_FLOAT4_E1M2 || dtype == ge::DT_INT4 ||
           dtype == Ops::NN::DT_HIFLOAT4;
}
uint64_t WeightQuantBatchMatmulV2TilingASWGeneral::GetShapeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    if (Is4BitInput(dtype)) {
        return shapeSize + shapeSize;
    } else {
        return shapeSize / static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

uint64_t WeightQuantBatchMatmulV2TilingASWGeneral::GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype) const
{
    // shapeSize应该是偶数
    if (Is4BitInput(dtype)) {
        OP_TILING_CHECK(
            shapeSize % 2 != 0, // 2: 判断是否是偶数
            CUBE_INNER_ERR_REPORT(
                opName_, "To get size of matrix/array, the number of elements must be even when dtype is FLOAT4/INT4"),
            return 0);
        // 1/2: 这几种数据类型的dsize=1/2
        return shapeSize / 2;
    } else {
        return shapeSize * static_cast<uint64_t>(ge::GetSizeByDataType(dtype));
    }
}

void WeightQuantBatchMatmulV2TilingASWGeneral::CalcTailBasicBlock()
{
    bool isPerGroup = matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP;
    bool isNzWeightNoTrans = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ && !matmulInfoPtr_->transB;
    if (isPerGroup || isNzWeightNoTrans) {
        return;
    }
    if (adaptiveWin_.tailWinBlockCnt == 0UL) {
        return;
    }

    uint64_t mTile = 1UL;
    uint64_t nTile = 1UL;
    uint64_t preSplit = 1UL;
    uint64_t secSplit = 1UL;
    auto& preSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? mTile : nTile;
    auto& secSplitValid = adaptiveWin_.mTail >= adaptiveWin_.nTail ? nTile : mTile;
    while (CalUsedCoreNum(preSplit + 1, secSplit) <= compileInfoPtr_->aicNum) {
        preSplit += 1UL;
        if (!IsInValidWeighNzTailSplit(preSplit, true)) {
            preSplitValid = preSplit;
        }

        if (CalUsedCoreNum(preSplit, secSplit + 1) <= compileInfoPtr_->aicNum) {
            secSplit += 1UL;
            if (!IsInValidWeighNzTailSplit(secSplit, false)) {
                secSplitValid = secSplit;
            }
        }
    }
    adaptiveWin_.mTailTile = mTile;
    adaptiveWin_.nTailTile = nTile;
}

bool WeightQuantBatchMatmulV2TilingASWGeneral::IsInValidWeighNzTailSplit(uint64_t splitCnt, bool isPreSplit) const
{
    if (matmulInfoPtr_->bFormat != ge::FORMAT_FRACTAL_NZ ||
        (((isPreSplit && adaptiveWin_.mTail >= adaptiveWin_.nTail) ||
          (!isPreSplit && adaptiveWin_.mTail < adaptiveWin_.nTail)))) {
        return false;
    }
    if (splitCnt == 0) {
        return false;
    }
    uint64_t tailN = adaptiveWin_.baseN / splitCnt;
    uint64_t baseNAlignSize = matmulInfoPtr_->transB ? CUBE_BLOCK :
                                                       GetShapeWithDataType(L1_ALIGN_SIZE, matmulInfoPtr_->bDtype);
    if (tailN % baseNAlignSize != 0) {
        return true;
    }

    // 如果是perchannel并且含有bias，需要通过fixpipe搬运到ub，重切分之后的内轴需要32B对齐
    if (matmulInfoPtr_->antiQuantType == QuantType::PER_CHANNEL && matmulInfoPtr_->hasBias) {
        if (adaptiveWin_.nTail % splitCnt != 0) {
            return true;
        }
        // 8:搬运到ub的数据类型是float(4B)，则重切分之后的n需要对32 / 4 = 8对齐
        if ((adaptiveWin_.nTail / splitCnt) % 8 != 0) {
            return true;
        }
    }

    return false;
}

uint32_t WeightQuantBatchMatmulV2TilingASWGeneral::CalUsedCoreNum()
{
    if (adaptiveWin_.totalWinCnt > 1 || adaptiveWin_.tailWinBlockCnt == 0) {
        return compileInfoPtr_->aicNum;
    }

    return adaptiveWin_.tailWinBlockCnt * adaptiveWin_.mTailTile * adaptiveWin_.nTailTile;
}

uint32_t WeightQuantBatchMatmulV2TilingASWGeneral::CalUsedCoreNum(uint32_t mTile, uint32_t nTile)
{
    return mTile * nTile * adaptiveWin_.tailWinBlockCnt;
}

void WeightQuantBatchMatmulV2TilingASWGeneral::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(opName_, "coreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;

    basicTiling_.stepM = 1;
    basicTiling_.stepN = 1;
    basicTiling_.singleCoreM = std::min(matmulInfoPtr_->mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(matmulInfoPtr_->nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = matmulInfoPtr_->kSize;

    basicTiling_.iterateOrder = 0;
    basicTiling_.dbL0c = ((basicTiling_.baseM * basicTiling_.baseN * sizeof(float) * DB_SIZE <=
                           compileInfoPtr_->l0cSize)) ?
                             DB_SIZE :
                             1;
}

void WeightQuantBatchMatmulV2TilingASWGeneral::CalStepKs()
{
    basicTiling_.stepKa = 2; // kL1和kUb默认取2 * baseK = 256
    basicTiling_.stepKb = basicTiling_.stepKa;
    // 2: stepK限制为偶数
    for (uint64_t stepK = 2; stepK <= MAX_PERF_STEPKB; stepK += 2) {
        uint64_t weightInSize = adaptiveWin_.baseN * adaptiveWin_.baseK * stepK * GetDtypeBits(matmulInfoPtr_->bDtype) /
                                BYTE_BITS;
        uint64_t weightOutSize = (adaptiveWin_.baseN + 1) * adaptiveWin_.baseK * stepK *
                                 GetDtypeBits(matmulInfoPtr_->aDtype) / BYTE_BITS;
        uint64_t scaleSize = adaptiveWin_.baseN * GetDtypeBits(matmulInfoPtr_->antiQuantScaleDtype) / BYTE_BITS;
        if (matmulInfoPtr_->antiQuantType == QuantType::PER_GROUP) {
            weightOutSize = adaptiveWin_.baseN * adaptiveWin_.baseK * stepK * GetDtypeBits(matmulInfoPtr_->aDtype) /
                            BYTE_BITS;
            scaleSize = scaleSize * ops::CeilDiv<uint64_t>(stepK * adaptiveWin_.baseK, matmulInfoPtr_->groupSize);
        }
        if (matmulInfoPtr_->hasAntiQuantOffset) {
            // 2：offset 大小与scale一致，不再重复计算
            scaleSize = scaleSize * 2;
        }
        uint64_t ubSize = (weightInSize + weightOutSize + scaleSize) * DB_SIZE;
        uint64_t l1Size = (adaptiveWin_.baseM + adaptiveWin_.baseN) * adaptiveWin_.baseK * stepK * DB_SIZE *
                          GetDtypeBits(matmulInfoPtr_->aDtype) / BYTE_BITS;
        if (matmulInfoPtr_->hasBias) {
            l1Size += adaptiveWin_.baseN * DB_SIZE * GetDtypeBits(matmulInfoPtr_->biasDtype) / BYTE_BITS;
        }
        if (ubSize <= compileInfoPtr_->ubSize && l1Size <= compileInfoPtr_->l1Size) {
            basicTiling_.stepKa = stepK;
            basicTiling_.stepKb = stepK;
        }
    }
    basicTiling_.depthA1 = basicTiling_.stepKa * DB_SIZE;
    basicTiling_.depthB1 = basicTiling_.stepKb * DB_SIZE;
}

void WeightQuantBatchMatmulV2TilingASWGeneral::SetTilingData()
{
    tilingData_->mTailTile = adaptiveWin_.mTailTile;
    tilingData_->nTailTile = adaptiveWin_.nTailTile;
    tilingData_->mainTileNum = adaptiveWin_.mainTileNum;
    tilingData_->lastRoundBaseN = adaptiveWin_.lastRoundBaseN;
    tilingData_->lastRoundBaseNTail = adaptiveWin_.lastRoundBaseNTail;
    tilingData_->evadeDuplicateBlock = static_cast<uint16_t>(adaptiveWin_.evadeDuplicateBlock);
    tilingData_->groupSize = matmulInfoPtr_->groupSize;
    tilingData_->enablePreAlloc = adaptiveWin_.enablePreAlloc;
    tilingData_->hasBias = static_cast<int32_t>(matmulInfoPtr_->hasBias);
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWGeneral::DoLibApiTiling()
{
    tilingData_->matmulTiling.M = matmulInfoPtr_->mSize;
    tilingData_->matmulTiling.N = matmulInfoPtr_->nSize;
    tilingData_->matmulTiling.Ka = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.Kb = matmulInfoPtr_->kSize;
    tilingData_->matmulTiling.usedCoreNum = basicTiling_.usedCoreNum;
    tilingData_->matmulTiling.singleCoreM = basicTiling_.singleCoreM;
    tilingData_->matmulTiling.singleCoreN = basicTiling_.singleCoreN;
    tilingData_->matmulTiling.singleCoreK = basicTiling_.singleCoreK;
    tilingData_->matmulTiling.baseM = basicTiling_.baseM;
    tilingData_->matmulTiling.baseN = basicTiling_.baseN;
    tilingData_->matmulTiling.baseK = basicTiling_.baseK;
    tilingData_->matmulTiling.depthA1 = basicTiling_.depthA1;
    tilingData_->matmulTiling.depthB1 = basicTiling_.depthB1;
    tilingData_->matmulTiling.stepM = basicTiling_.stepM;
    tilingData_->matmulTiling.stepN = basicTiling_.stepN;
    tilingData_->matmulTiling.stepKa = basicTiling_.stepKa;
    tilingData_->matmulTiling.stepKb = basicTiling_.stepKb;
    tilingData_->matmulTiling.isBias = static_cast<int32_t>(matmulInfoPtr_->hasBias);
    tilingData_->matmulTiling.iterateOrder = basicTiling_.iterateOrder;
    tilingData_->matmulTiling.dbL0A = 2; // db switch, 1: off, 2: on
    tilingData_->matmulTiling.dbL0B = 2; // db switch, 1: off, 2: on
    tilingData_->matmulTiling.dbL0C = basicTiling_.dbL0c;
    if (matmulInfoPtr_->antiQuantType == QuantType::MX) {
        // _mxTypePara的高8位是拷贝scaleB的倍数，257是 1 << 8 + 1 表示scale就拷贝一倍的k
        tilingData_->matmulTiling.mxTypePara = 257;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWGeneral::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName_, "failed to get workspace size"),
                    return ge::GRAPH_FAILED);
    // 申请的 workspace 大小为 16 MB
    workspaces[0] = 16 * MEGABYTE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus WeightQuantBatchMatmulV2TilingASWGeneral::PostTiling()
{
    OP_LOGD(opName_, "final tiling data size: %zu", tilingDataSize_);
    OP_TILING_CHECK(tilingDataSize_ % sizeof(uint64_t) != 0,
                    CUBE_INNER_ERR_REPORT(opName_, "tiling data size[%zu] is not aligned to 8", tilingDataSize_),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(),
                           static_cast<const void*>(tilingData_.get()), tilingDataSize_);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->SetBlockDim(basicTiling_.usedCoreNum);
    context_->GetRawTilingData()->SetDataSize(tilingDataSize_);
    return ge::GRAPH_SUCCESS;
}

bool WeightQuantBatchMatmulV2TilingASWGeneral::ClosePreAllocCond() const
{
    int64_t baseK = std::min(adaptiveWin_.baseK, matmulInfoPtr_->kSize);
    int64_t actualMSize = std::min(matmulInfoPtr_->mSize, adaptiveWin_.baseM);
    int64_t actualNSize = std::min(matmulInfoPtr_->nSize, adaptiveWin_.baseN);
    int64_t aL1Size = actualMSize * baseK * basicTiling_.stepKa * ge::GetSizeByDataType(matmulInfoPtr_->aDtype);
    int64_t bL1Size = actualNSize * baseK * basicTiling_.stepKa * GetDtypeBits(matmulInfoPtr_->bDtype) / BYTE_BITS;
    // AL1 + BL1载入数据量较大(128 * 1024 B,经验值)时，开启prealloc会导致后2轮的mte2抢占前面的ostd,
    // 导致前面的带宽跟不上cube计算 当K较小时，单轮计算量较小， 如果不开启prealloc,
    // 下一轮MN的计算的MTE2无法提前发出，会导致mte2断流
    bool condition = aL1Size + bL1Size >= 128 * 1024 && matmulInfoPtr_->kSize > 1024;
    // 尾轮切分可能会破坏 aL1Size + bL1Size 较大的条件，当只有一轮计算并且进行了重切分时，数据量会变小，需要开启
    condition = condition && !(adaptiveWin_.totalWinCnt == 1 && adaptiveWin_.mTailTile * adaptiveWin_.nTailTile > 1);
    return condition;
}

uint64_t WeightQuantBatchMatmulV2TilingASWGeneral::GetTilingKey() const
{
    uint64_t socVersionType = WQBMMV2_SOC_ASCEND960DT;
    uint64_t subSocVersionType = WQBMMV2_DEFAULT;
    uint64_t antiquantScenario = WQBMMV2_DEFAULT;
    uint64_t algorithm = WQBMMV2_ALGO_VECTOR_ANTIQUANT;
    uint64_t subAlgorithm = WQBMMV2_SUB_ALGO_ASW;

    uint64_t templateCustom = 0UL;
    uint64_t apiConstexpr = 0UL;

    bool transA = matmulInfoPtr_->transA;
    bool transB = matmulInfoPtr_->transB;
    uint64_t antiquantType = static_cast<uint64_t>(matmulInfoPtr_->antiQuantType);
    uint64_t quantType = WQBMMV2_QUANT_TYPE_NONE;
    bool hasAntiquantOffset = matmulInfoPtr_->hasAntiQuantOffset;
    bool hasBias = matmulInfoPtr_->hasBias;
    bool isBiasFp32 = matmulInfoPtr_->biasDtype == ge::DT_FLOAT;
    bool isWeightNz = matmulInfoPtr_->bFormat == ge::FORMAT_FRACTAL_NZ;
    uint64_t tilingKey = GET_TPL_TILING_KEY(socVersionType, subSocVersionType, antiquantScenario, algorithm,
                                            subAlgorithm, templateCustom, apiConstexpr, transA, transB, antiquantType,
                                            quantType, hasAntiquantOffset, hasBias, isBiasFp32, isWeightNz);
    return tilingKey;
}

REGISTER_TILING_TEMPLATE("WeightQuantBatchMatmulV2", WeightQuantBatchMatmulV2TilingASWGeneral, ASW_GENERAL_PRIORITY);
} // namespace weight_quant_batch_matmul_v2
} // namespace optiling
