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
 * \file qbmm_api_utils.h
 * \brief
 */
#ifndef QBMM_API_UTILS_H
#define QBMM_API_UTILS_H

#include "../quant_batch_matmul_v3_base.h"
#include "qbmm_asw_block.h"


namespace QuantBatchMatmulV3 {

using namespace AscendC;
using namespace matmul;

// 这里入参由blockIdx改成mCntIdx，目的是abL1全载时调用更方便
template <typename T, bool trans>
__aicore__ inline void CopyInA1(const QuantBmmAswBlock &block, uint64_t mCntIdx, bool isMultiCore,
                                LocalTensor<T> &al1Local, const GlobalTensor<T> &aGlobal)
{
    auto &matmulTilingData = *block.tilingData_;
    uint64_t shapeM = matmulTilingData.matmulTiling.M;
    uint64_t shapeK = matmulTilingData.matmulTiling.Ka;
    uint64_t singleCoreM = isMultiCore && mCntIdx == block.params_.mCnt - 1UL
                               ? block.params_.mBaseTail
                               : static_cast<uint64_t>(matmulTilingData.matmulTiling.singleCoreM);
    uint64_t nDim = singleCoreM;
    uint64_t dDim = shapeK;
    uint64_t offsetA = mCntIdx * matmulTilingData.matmulTiling.singleCoreM * shapeK;
    // supportMmadS8S4平台的al1全载支持s8s8, s8s4, s4s4; david的al1全载支持s8s8, f8s8, f4f4
    constexpr uint64_t c0Size = DequantBmm::GetC0Size<T>();
    constexpr int64_t int4Factor = c0Size == K0_INT4 ? 2 : 1;
    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;

    if constexpr (trans) {
        nDim = shapeK;
        dDim = singleCoreM;
        offsetA = mCntIdx * matmulTilingData.matmulTiling.singleCoreM;
        nd2nzParams.srcDValue = DequantBmm::CeilDiv(shapeM, int4Factor);
    } else {
        nd2nzParams.srcDValue = DequantBmm::CeilDiv(shapeK, int4Factor);
    }

    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = DequantBmm::CeilDiv(dDim, int4Factor);
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = DequantBmm::Align(nDim, BMM_BLOCK_NUM);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    if constexpr (c0Size == K0_INT4) {
        GlobalTensor<int8_t> aGlobalInt8;
        aGlobalInt8.SetGlobalBuffer(((__gm__ int8_t *)(aGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
        auto al1LocalImpl = al1Local.template ReinterpretCast<int8_t>();
        DataCopy(al1LocalImpl, aGlobalInt8[offsetA >> 1], nd2nzParams);
    } else {
        DataCopy(al1Local, aGlobal[offsetA], nd2nzParams);
    }
}

template <typename T, bool trans>
__aicore__ inline void CopyInNDB1(const QuantBmmAswBlock &block, uint64_t nCntIdx, bool isMultiCore,
                                  LocalTensor<T> &bl1Local, const GlobalTensor<T> &bGlobal)
{
    auto &matmulTilingData = *block.tilingData_;
    uint64_t shapeK = matmulTilingData.matmulTiling.Kb;
    uint64_t shapeN = matmulTilingData.matmulTiling.N;
    uint64_t singleCoreN = isMultiCore && nCntIdx == block.params_.nCnt - 1UL
                               ? block.params_.nBaseTail
                               : static_cast<uint64_t>(matmulTilingData.matmulTiling.singleCoreN);
    uint64_t offsetB = 0;
    // 临时代码，abl1支持s8s8 s8s4, bl1支持s8s8, 无fp4情况
    constexpr int64_t int4Factor = (AscendC::IsSameType<T, AscendC::int4b_t>::value) ? 2 : 1;
    uint64_t nDim = shapeK;
    uint64_t dDim = singleCoreN;

    Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    if constexpr (trans) {
        offsetB = nCntIdx * matmulTilingData.matmulTiling.singleCoreN * shapeK;
        nDim = singleCoreN;
        dDim = shapeK;
        nd2nzParams.srcDValue = DequantBmm::CeilDiv(shapeK, int4Factor);
    } else {
        offsetB = nCntIdx * matmulTilingData.matmulTiling.singleCoreN;
        nd2nzParams.srcDValue = DequantBmm::CeilDiv(shapeN, int4Factor);
    }
    nd2nzParams.nValue = nDim;
    nd2nzParams.dValue = DequantBmm::CeilDiv(dDim, int4Factor);
    nd2nzParams.srcNdMatrixStride = 0;
    nd2nzParams.dstNzC0Stride = DequantBmm::Align(nDim, BMM_BLOCK_NUM);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;
    if constexpr (AscendC::IsSameType<T, AscendC::int4b_t>::value) {
        GlobalTensor<int8_t> bGlobalInt8;
        bGlobalInt8.SetGlobalBuffer(((__gm__ int8_t *)(bGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
        auto bl1LocalImpl = bl1Local.template ReinterpretCast<int8_t>();
        DataCopy(bl1LocalImpl, bGlobalInt8[offsetB >> 1], nd2nzParams);
    } else {
        DataCopy(bl1Local, bGlobal[offsetB], nd2nzParams);
    }
}

template <typename T, bool trans>
__aicore__ inline void CopyInNZB1(const QuantBmmAswBlock &block, uint64_t nCntIdx, bool isMultiCore,
                                  LocalTensor<T> &bl1Local, const GlobalTensor<T> &bGlobal,
                                  uint64_t innerX2AlignedBlock)
{
    auto &matmulTilingData = *block.tilingData_;
    uint64_t shapeK = matmulTilingData.matmulTiling.Kb;
    uint64_t shapeN = matmulTilingData.matmulTiling.N;
    uint64_t baseN = matmulTilingData.matmulTiling.baseN;
    uint64_t singleCoreN = isMultiCore && nCntIdx == block.params_.nCnt - 1UL
                               ? block.params_.nBaseTail
                               : static_cast<uint64_t>(matmulTilingData.matmulTiling.singleCoreN);
    uint64_t offsetB = 0;
    DataCopyParams gm2L1Params{1, 0, 0, 0};
    if constexpr (trans) {
        offsetB = nCntIdx * baseN * innerX2AlignedBlock;
        gm2L1Params.blockCount = static_cast<uint16_t>(DequantBmm::CeilDiv(shapeK, innerX2AlignedBlock));
        gm2L1Params.blockLen = static_cast<uint16_t>(DequantBmm::Align(singleCoreN));
        gm2L1Params.srcStride = static_cast<uint16_t>(DequantBmm::Align(shapeN - singleCoreN));
    } else {
        offsetB = nCntIdx * matmulTilingData.matmulTiling.singleCoreN * DequantBmm::Align(shapeK);
        gm2L1Params.blockCount = 1;
        gm2L1Params.blockLen = DequantBmm::Align(shapeK) *
                                DequantBmm::CeilDiv(singleCoreN, innerX2AlignedBlock);
    }
    DataCopy(bl1Local, bGlobal[offsetB], gm2L1Params);
}

template <typename T>
__aicore__ inline void CopyInB1Bias(const QuantBmmAswBlock &block, uint64_t nCntIdx, bool isMultiCore,
                                    LocalTensor<T> &biasl1Local, const GlobalTensor<T> &biasGlobal,
                                    uint64_t innerBiasAlignedBlock)
{
    auto &matmulTilingData = *block.tilingData_;
    uint64_t singleCoreN = isMultiCore && nCntIdx == block.params_.nCnt - 1UL
                               ? block.params_.nBaseTail
                               : static_cast<uint64_t>(matmulTilingData.matmulTiling.singleCoreN);

    // 32B对齐
    uint64_t alignSize = DequantBmm::Align(singleCoreN, innerBiasAlignedBlock) * sizeof(T);
    DataCopyExtParams dataCopyExtParams{1, static_cast<uint32_t>(alignSize), static_cast<uint32_t>(alignSize),
                                        static_cast<uint32_t>(alignSize), 0};

    uint64_t offsetBias = nCntIdx * matmulTilingData.matmulTiling.singleCoreN;
    DataCopyPad(biasl1Local, biasGlobal[offsetBias], dataCopyExtParams, {false, 0, 0, 0});
}

template <typename T, bool trans>
__aicore__ inline void CopyInScaleA(const QuantBmmAswBlock& block, uint32_t blockId, bool isMultiCore,
                                    LocalTensor<T>& scaleAl1Local, const GlobalTensor<T>& scaleAGlobal)
{
    auto &multiTilingData = *block.tilingData_;
    uint64_t shapeM = multiTilingData.matmulTiling.M;
    uint64_t shapeK =
        DequantBmm::Align(DequantBmm::CeilDiv(multiTilingData.matmulTiling.Ka, MXFP_GROUP_SIZE), MXFP_MULTI_BASE_SIZE);
    uint64_t mCntIdx = blockId % block.params_.mCnt;
    uint64_t singleCoreM = isMultiCore && mCntIdx == block.params_.mCnt - 1UL
                          ? block.params_.mBaseTail
                          : static_cast<uint64_t>(multiTilingData.matmulTiling.singleCoreM);

    uint64_t nDim = singleCoreM;
    uint64_t dDim = shapeK;
    uint64_t offsetScaleA = 0;
    if constexpr (trans) {
        nDim = shapeK;
        dDim = singleCoreM;
        offsetScaleA = mCntIdx * multiTilingData.matmulTiling.singleCoreM;
    } else {
        offsetScaleA = mCntIdx * multiTilingData.matmulTiling.singleCoreM * shapeK;
    }

    Dn2NzParams dn2nzParams;
    dn2nzParams.dnNum = 1;
    dn2nzParams.dValue = nDim;
    dn2nzParams.nValue = DequantBmm::CeilDiv(dDim, 2); // 将两个float8e8m0合成一个half类型，一起搬运
    dn2nzParams.srcDnMatrixStride = 0;
    dn2nzParams.srcDValue = shapeK >> 1;
    dn2nzParams.dstNzC0Stride = DequantBmm::CeilDiv(dDim, 2); // 将两个float8e8m0合成一个half类型，一起搬运
    dn2nzParams.dstNzNStride = 1;
    dn2nzParams.dstNzMatrixStride = 0;

    GlobalTensor<half> aScaleGlobalB16;
    aScaleGlobalB16.SetGlobalBuffer(((__gm__ half *)(scaleAGlobal.GetPhyAddr())), (nDim * dDim) >> 1);
    auto scaleAl1LocalImpl = scaleAl1Local.template ReinterpretCast<half>();
    DataCopy(scaleAl1LocalImpl, aScaleGlobalB16[offsetScaleA >> 1], dn2nzParams);
}

template <typename T>
__aicore__ inline void ProcessWithBatch(QuantBmmAswBlock& block, T& object)
{
    uint64_t batchC3C4 =
        static_cast<uint64_t>(block.tilingData_->params.batchC3) * block.tilingData_->params.batchC4;
    uint64_t batchC2C3C4 = block.tilingData_->params.batchC2 * batchC3C4;
    uint64_t batchB3B4 =
        static_cast<uint64_t>(block.tilingData_->params.batchB3) * block.tilingData_->params.batchB4;
    uint64_t batchB2B3B4 = block.tilingData_->params.batchB2 * batchB3B4;
    uint64_t batchA3A4 =
        static_cast<uint64_t>(block.tilingData_->params.batchA3) * block.tilingData_->params.batchA4;
    uint64_t batchA2A3A4 = block.tilingData_->params.batchA2 * batchA3A4;
    uint32_t multiA1C1 = block.tilingData_->params.batchA1 / block.tilingData_->params.batchC1;
    uint32_t multiA2C2 = block.tilingData_->params.batchA2 / block.tilingData_->params.batchC2;
    uint32_t multiA3C3 = block.tilingData_->params.batchA3 / block.tilingData_->params.batchC3;
    uint32_t multiA4C4 = block.tilingData_->params.batchA4 / block.tilingData_->params.batchC4;
    uint32_t multiB1C1 = block.tilingData_->params.batchB1 / block.tilingData_->params.batchC1;
    uint32_t multiB2C2 = block.tilingData_->params.batchB2 / block.tilingData_->params.batchC2;
    uint32_t multiB3C3 = block.tilingData_->params.batchB3 / block.tilingData_->params.batchC3;
    uint32_t multiB4C4 = block.tilingData_->params.batchB4 / block.tilingData_->params.batchC4;
    uint64_t batchC1Offset = 0;
    uint64_t batchA1Offset = 0;
    uint64_t batchB1Offset = 0;
    for (uint64_t b1Index = 0; b1Index < block.tilingData_->params.batchC1; ++b1Index) {
        uint64_t batchC2Offset = batchC1Offset;
        uint64_t batchA2Offset = batchA1Offset;
        uint64_t batchB2Offset = batchB1Offset;
        for (uint64_t b2Index = 0; b2Index < block.tilingData_->params.batchC2; ++b2Index) {
            uint64_t batchC3Offset = batchC2Offset;
            uint64_t batchA3Offset = batchA2Offset;
            uint64_t batchB3Offset = batchB2Offset;
            for (uint64_t b3Index = 0; b3Index < block.tilingData_->params.batchC3; ++b3Index) {
                block.offset_.batchCOffset = batchC3Offset;
                block.offset_.batchAOffset = batchA3Offset;
                block.offset_.batchBOffset = batchB3Offset;
                for (uint64_t b4Index = 0; b4Index < block.tilingData_->params.batchC4; ++b4Index) {
                    block.ResetAddressOffsets();
                    object.ProcessWithoutBatch();
                    block.offset_.batchCOffset += 1;
                    block.offset_.batchAOffset += multiA4C4;
                    block.offset_.batchBOffset += multiB4C4;
                }
                batchC3Offset += block.tilingData_->params.batchC4;
                batchA3Offset += block.tilingData_->params.batchA4 * static_cast<uint64_t>(multiA3C3);
                batchB3Offset += block.tilingData_->params.batchB4 * static_cast<uint64_t>(multiB3C3);
            }
            batchC2Offset += batchC3C4;
            batchA2Offset += batchA3A4 * multiA2C2;
            batchB2Offset += batchB3B4 * multiB2C2;
        }
        batchC1Offset += batchC2C3C4;
        batchA1Offset += batchA2A3A4 * multiA1C1;
        batchB1Offset += batchB2B3B4 * multiB1C1;
    }
}

}  // namespace QuantBatchMatmulV3

#endif  // QBMM_API_UTILS_H