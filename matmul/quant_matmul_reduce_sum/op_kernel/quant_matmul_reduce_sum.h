/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_reduce_sum.h
 * \brief
 */
#ifndef ASCENDC_QUANT_MATMUL_REDUCE_SUM_H
#define ASCENDC_QUANT_MATMUL_REDUCE_SUM_H

#include "quant_matmul_reduce_sum_utils.h"
#include "quant_matmul_reduce_sum_tiling_data.h"

namespace QUANT_MATMUL_REDUCE_SUM {

constexpr uint32_t thresholdBlockNum = 8; // 8 is obtained by tests, indicating the threshold of basic block numbers
                                          // in both directions when assigning data blocks to cube cores when using
                                          // diagnal strategy
constexpr uint32_t thresholdDimM = 5;     // 5 is obtained by tests, indicating the threshold for distinguishing
                                          // strategies for large/small shapes

/*@brief store variables for core split configuration
 */
struct QbmmRSComputeInitParams {
    GM_ADDR x1;
    GM_ADDR x2;
    GM_ADDR x1Scale;
    GM_ADDR x2Scale;
    GM_ADDR y;
    GM_ADDR workspace;
    GM_ADDR user1;
    GM_ADDR tiling;
};

struct MNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t x1BaseOffset = 0;
    uint64_t x2BaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
    uint64_t workSpaceOffset = 0;
};

template <typename T>
__aicore__ inline void DataCopyPad2D(
    const AscendC::LocalTensor<T> dst, const AscendC::GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
    uint32_t fullDim0)
{
    DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (fullDim0 - dim0) * sizeof(T);
    params.dstStride =
        Ceil(dim0 * sizeof(T), UB_BLOCK_DOUBLE_UNIT_SIZE) * 2 - Ceil(dim0 * sizeof(T), UB_BLOCK_UNIT_SIZE);

    DataCopyPadExtParams<T> padParams;
    padParams.isPad = true;
    padParams.rightPadding = 0;
    padParams.leftPadding = 0;
    padParams.paddingValue = 0;
    DataCopyPad(dst, src, params, padParams);
}

/** @brief GroupMatmul operator Class
 */
template <typename ComputeType>
class QbmmRSProcess
{
protected:
    using B = typename ComputeType::B;
    ComputeType& computeOp; // inernal computation operator
    const QuantMatmulReduceSumParams* __restrict qbmmParams;
    const TCubeTiling* __restrict mmTilingData;

    uint32_t blockIdx;
    uint32_t coreIdx;
    uint32_t batchNum;
    int32_t preOffset = 0;

public:
    /** @brief constructor */
    __aicore__ inline QbmmRSProcess(ComputeType& computeOp_) : computeOp(computeOp_)
    {}

    __aicore__ inline void Init(
        const QuantMatmulReduceSumParams* __restrict qbmmParamsIn, const TCubeTiling* __restrict mmTilingDataIn);

    __aicore__ inline void Process();

protected:
    __aicore__ inline void SetMNConfig(MNConfig& mnConfig);

    __aicore__ inline void SetMKN(MNConfig& mnConfig);

    __aicore__ inline void UpdateMnConfig(MNConfig& mnConfig);

    __aicore__ inline void MNBlockIdxCompute(
        MNConfig& mnConfig, const uint32_t curBlock, const uint32_t count, const uint32_t thresholdM_dimN);
};

template <typename ComputeType>
__aicore__ inline void QbmmRSProcess<ComputeType>::Init(
    const QuantMatmulReduceSumParams* __restrict qbmmParamsIn, const TCubeTiling* __restrict mmTilingDataIn)
{
    blockIdx = GetBlockIdx();
    coreIdx = blockIdx;
    int64_t coreRation = GetTaskRation();
    if (coreRation > 1) {
        coreIdx /= coreRation;
    }
    qbmmParams = qbmmParamsIn;
    mmTilingData = mmTilingDataIn;
    batchNum = qbmmParams->batchNum;
}

template <typename ComputeType>
__aicore__ inline void QbmmRSProcess<ComputeType>::SetMNConfig(MNConfig& mnConfig)
{
    SetMKN(mnConfig);
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
}

template <typename ComputeType>
__aicore__ inline void QbmmRSProcess<ComputeType>::SetMKN(MNConfig& mnConfig)
{
    mnConfig.m = mmTilingData->M;
    mnConfig.n = mmTilingData->N;
    mnConfig.k = mmTilingData->Ka;
    return;
}

template <typename ComputeType>
__aicore__ inline void QbmmRSProcess<ComputeType>::UpdateMnConfig(MNConfig& mnConfig)
{
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    // only support w in NZ format -- int8_t: k0 = 16， n0 = 32
    mnConfig.x1BaseOffset += mnConfig.m * mnConfig.k;
    mnConfig.x2BaseOffset += AlignUp<16>(mnConfig.k) * AlignUp<32>(mnConfig.n); // 16: nz format last two dim size
    mnConfig.yBaseOffset += mnConfig.m * mnConfig.n;
}

template <typename ComputeType>
__aicore__ inline void QbmmRSProcess<ComputeType>::MNBlockIdxCompute(
    MNConfig& mnConfig, const uint32_t curBlock, const uint32_t count, const uint32_t thresholdM_dimN)
{
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdM_dimN) ?
                                     mnConfig.blockDimM % thresholdBlockNum :
                                     thresholdBlockNum;
        uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN =
            relativeBlock % thresholdM_dimN >= AlignDown(curThresholdM * mnConfig.blockDimN, curThresholdM_thresholdN) ?
                mnConfig.blockDimN % thresholdBlockNum :
                thresholdBlockNum;

        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN)) %
                            curThresholdN +
                        relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
    }
}

template <typename ComputeType>
__aicore__ inline void QbmmRSProcess<ComputeType>::Process()
{
    MNConfig mnConfig;
    AscendC::WaitPreTaskEnd();
    for (uint32_t batchIdx = 0, count = 0; batchIdx < batchNum; ++batchIdx) {
        UpdateMnConfig(mnConfig);
        SetMNConfig(mnConfig);
        if (mnConfig.m <= 0 || mnConfig.k <= 0 || mnConfig.n <= 0) {
            continue;
        }
        mnConfig.blockDimM = Ceil(mnConfig.m, mnConfig.baseM);
        mnConfig.blockDimN = Ceil(mnConfig.n, mnConfig.baseN);

        uint32_t curCount = count + mnConfig.blockDimM * mnConfig.blockDimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + qbmmParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * mnConfig.blockDimN;

        while (curBlock < curCount) {
            MNBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            computeOp.MMCompute(batchIdx, mnConfig, coreIdx);
            computeOp.VectorCompute(batchIdx, mnConfig);
            curBlock += qbmmParams->coreNum;
        }
        if (qbmmParams->isDetermine != 1) {
            count = curCount % qbmmParams->coreNum;
        }
    }
    computeOp.PostCompute();
    AscendC::SetNextTaskStart();
}

/** @brief intenal computation class
 */
template <class mmType, bool sync = false>
class QbmmRSCompute
{
public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using B = typename mmType::BT;
    using CT = typename mmType::CT::T;
    using BiasT = typename mmType::BiasT::T;
    constexpr static bool transposeX1 = mmType::AT::isTrans;
    constexpr static bool transposeX2 = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline QbmmRSCompute(typename mmType::MT& mm_) : mm(mm_)
    {}

    __aicore__ inline void Init(
        const QbmmRSComputeInitParams* __restrict initParams, const QuantMatmulReduceSumParams* __restrict qbmmParams,
        const TCubeTiling* __restrict mmTilingData, TPipe* tPipe);

protected:
    __aicore__ inline AscendC::GlobalTensor<BT> SetGlobalBufferW(uint32_t batchIdx, uint32_t tailN, MNConfig& mnConfig);

    __aicore__ inline uint64_t Setx2Offset(uint32_t tailN, uint32_t k);

protected:
    TPipe* pipe;
    typename mmType::MT& mm; // matmul operator
    bool hasBias = false;
    GM_ADDR x1TensorPtr;
    GM_ADDR x2TensorPtr;
    GM_ADDR yTensorPtr;
    AscendC::GlobalTensor<AT> x1Gm;
    AscendC::GlobalTensor<BT> x2Gm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t ubCalSize;
    uint32_t coreNum;
    uint32_t subBlockIdx;
};

template <typename mmType, bool sync>
__aicore__ inline void QbmmRSCompute<mmType, sync>::Init(
    const QbmmRSComputeInitParams* __restrict initParams, const QuantMatmulReduceSumParams* __restrict qbmmParams,
    const TCubeTiling* __restrict mmTilingData, TPipe* tPipe)
{
    x1TensorPtr = initParams->x1;
    x2TensorPtr = initParams->x2;
    yTensorPtr = initParams->y;
    pipe = tPipe;
    ubCalSize = qbmmParams->ubCalSize;
    coreNum = qbmmParams->coreNum;
    subBlockIdx = GetSubBlockIdx();
}

template <typename mmType, bool sync>
__aicore__ inline uint64_t QbmmRSCompute<mmType, sync>::Setx2Offset(uint32_t tailN, uint32_t k)
{
    uint64_t x2Offset = 0;
    if constexpr (mmType::BT::format == CubeFormat::NZ && transposeX2) {
        x2Offset = tailN * (UB_BLOCK_UNIT_SIZE / sizeof(BT)); // 32: quant is 32, float16 is 16
    } else if constexpr (mmType::BT::format == CubeFormat::NZ) {
        x2Offset = tailN * AlignUp<16>(k); // 16: nz format last two dim size
    } else if constexpr (transposeX2) {
        x2Offset = tailN * k;
    } else {
        x2Offset = tailN;
    }
    return x2Offset;
}

template <typename mmType, bool sync>
__aicore__ inline AscendC::GlobalTensor<typename mmType::BT::T> QbmmRSCompute<mmType, sync>::SetGlobalBufferW(
    uint32_t batchIdx, uint32_t tailN, MNConfig& mnConfig)
{
    uint64_t x2Offset = Setx2Offset(tailN, mnConfig.k) + mnConfig.x2BaseOffset;
    AscendC::GlobalTensor<BT> x2GmLocal;
    x2GmLocal.SetGlobalBuffer((__gm__ BT*)this->x2TensorPtr + x2Offset);
#if !(defined(ASCENDC_OOM) && ASCENDC_OOM == 1)
    if (mnConfig.blockDimM == 1) {
        x2GmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
#endif
    return x2GmLocal;
}

} // namespace QUANT_MATMUL_REDUCE_SUM

#endif // ASCENDC_QUANT_MATMUL_REDUCE_SUM_H
