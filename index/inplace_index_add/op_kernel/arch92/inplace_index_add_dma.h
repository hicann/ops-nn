/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_add_dma.h
 * \brief A6(ascend960) ScatterDMA kernel of inplace_index_add
 */
#ifndef ASCENDC_INPLACE_INDEX_ADD_DMA_H_
#define ASCENDC_INPLACE_INDEX_ADD_DMA_H_

#include "../arch35/inplace_index_add_common.h"

namespace InplaceIndexAdd {
using namespace AscendC;

constexpr int8_t SPLIT_PRE = 0;
constexpr int8_t SPLIT_INDICES = 1;
constexpr int8_t SPLIT_AFTER = 2;

constexpr static AscendC::Reg::CastTrait castTraitB32B64 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
};

template <typename T>
__simd_callee__ inline void LoadOneTensorForDtypeIntVf(__ubuf__ T* input, AscendC::Reg::RegTensor<int16_t>& dst,
                                                       AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        AscendC::Reg::LoadAlign<int8_t, AscendC::Reg::LoadDist::DIST_UNPACK_B8>((AscendC::Reg::RegTensor<int8_t>&)(dst),
                                                                                ((__ubuf__ int8_t*)(input) + (offset)));
    } else if constexpr (IsSameType<T, bool>::value) {
        AscendC::Reg::LoadAlign<bool, AscendC::Reg::LoadDist::DIST_UNPACK_B8>((AscendC::Reg::RegTensor<bool>&)(dst),
                                                                              ((__ubuf__ bool*)(input) + (offset)));
    } else {
        AscendC::Reg::LoadAlign(dst, ((__ubuf__ int16_t*)(input) + (offset)));
    }
}

template <typename T>
__simd_callee__ inline void StoreOneTensorForDtypeIntVf(__ubuf__ T* output, AscendC::Reg::RegTensor<int16_t>& src,
                                                        AscendC::Reg::MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, int8_t>::value) {
        AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK_B16>(
            ((__ubuf__ int8_t*)output + offset), (AscendC::Reg::RegTensor<int8_t>&)src, preg);
    } else if constexpr (IsSameType<T, bool>::value) {
        AscendC::Reg::StoreAlign<bool, AscendC::Reg::StoreDist::DIST_PACK_B16>(
            ((__ubuf__ bool*)output + offset), (AscendC::Reg::RegTensor<bool>&)src, preg);
    } else {
        AscendC::Reg::StoreAlign(((__ubuf__ int16_t*)output + offset), src, preg);
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
class InplaceIndexAddDma {
public:
    __aicore__ inline InplaceIndexAddDma(const InplaceIndexAddDmaTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void ComputeMulWithCast(LocalTensor<VAR_T> updatesLocal, LocalTensor<VAR_T> updateMulLocal,
                                              int16_t alphaValue, int64_t colLen);
    __aicore__ inline void CopyInUpdates(LocalTensor<VAR_T> updatesLocal, int64_t rowLen, int64_t colLen,
                                         uint32_t stride, int64_t updatesOfset);
    template <int8_t SPLIT_MODE>
    __aicore__ inline void ProcessSingleLoop(int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessPre();
    __aicore__ inline void ProcessIndices();
    __aicore__ inline void ProcessAfter();
    __aicore__ inline void Process();
    __aicore__ inline void CopyXToOut(__ubuf__ int8_t* inAddr, __ubuf__ int8_t* outAddr, int64_t dataCount);
    __aicore__ inline void ComputeOffset(LocalTensor<IDX_T> indicesLocal, LocalTensor<int64_t> offsetLocal,
                                         uint32_t indicesCount, int64_t maxIndicesValue, int64_t preVarOffset,
                                         int64_t colOfsetSzie, int64_t afterAxisSize);
    __aicore__ inline void ScatterDMA(GlobalTensor<VAR_T> dstGm, LocalTensor<VAR_T> updatesLocal,
                                      LocalTensor<int64_t> offsetLocal, uint16_t offsetNum, uint32_t burstLen,
                                      uint32_t srcStride);
    static __simd_vf__ inline void ComputeMulWithCastVf(__ubuf__ VAR_T* updatesAddr, __ubuf__ VAR_T* updateMulAddr,
                                                        uint16_t loopSize, uint32_t maskLen, int16_t alphaValue,
                                                        uint32_t vfLen);
    static __simd_vf__ inline void ComputeOffsetvf(__ubuf__ IDX_T* indicesAddr, __ubuf__ int64_t* offsetAddr,
                                                   uint16_t loopCnt, uint32_t vfLen, int64_t maxIndicesValue,
                                                   int64_t preVarOffset, int64_t colOfsetSzie, int64_t afterAxisSize,
                                                   uint32_t size, uint32_t sizeTwo);
    static __simd_vf__ inline void CopyXToOutVf(__ubuf__ int8_t* inAddr, __ubuf__ int8_t* outAddr, uint32_t totalBytes,
                                                uint16_t stride, uint16_t size);

private:
    using mulsSelType = typename MulsSelType<VAR_T>::type;
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> offsetsQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outQue_;

    TPipe& pipe_;
    const InplaceIndexAddDmaTilingData& tilingData_;
    int64_t blockIdx_;
    int64_t blockNum_;
    int64_t normBlockData_{0};
    int64_t usedCoreNum_{0};
    int64_t loopNum_{0};
    int64_t tailLoopLength_{0};
    mulsSelType alphaValue_{0};
    int64_t curPreAxisCount_{0};
};

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::Init(GM_ADDR var, GM_ADDR indices,
                                                                             GM_ADDR updates, GM_ADDR alpha,
                                                                             GM_ADDR workspace)
{
    updates_.SetGlobalBuffer((__gm__ VAR_T*)(updates));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    alpha_.SetGlobalBuffer((__gm__ VAR_T*)(alpha));
    var_.SetGlobalBuffer((__gm__ VAR_T*)(var));

    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * sizeof(IDX_T));
    pipe_.InitBuffer(offsetsQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * sizeof(int64_t));
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER,
                     tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));
    pipe_.InitBuffer(outQue_, DOUBLE_BUFFER, tilingData_.ubIndexFactor * tilingData_.afterAxisFactor * sizeof(VAR_T));

    if (tilingData_.isWithAlpha) {
        alphaValue_ = alpha_(0);
    }
    if (tilingData_.isSplitPreAxis == 1) {
        curPreAxisCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCorePreAxisCount :
                                                                                   tilingData_.tailCorePreAxisCount);
    } else {
        curPreAxisCount_ = tilingData_.preAxis;
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__simd_vf__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::CopyXToOutVf(__ubuf__ int8_t* inAddr,
                                                                                      __ubuf__ int8_t* outAddr,
                                                                                      uint32_t totalBytes,
                                                                                      uint16_t stride, uint16_t size)
{
    AscendC::Reg::RegTensor<int8_t> inputRegTensor;
    uint32_t sreg = totalBytes;
    AscendC::Reg::MaskReg preg;
    for (uint16_t i = 0; i < size; i++) {
        preg = AscendC::Reg::UpdateMask<int8_t>(sreg);
        AscendC::Reg::AddrReg offset = AscendC::Reg::CreateAddrReg<int8_t>(i, stride);
        AscendC::Reg::LoadAlign(inputRegTensor, inAddr, offset);
        AscendC::Reg::StoreAlign(outAddr, inputRegTensor, offset, preg);
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::CopyXToOut(__ubuf__ int8_t* inAddr,
                                                                                   __ubuf__ int8_t* outAddr,
                                                                                   int64_t dataCount)
{
    uint32_t totalBytes = dataCount * sizeof(VAR_T);
    uint16_t stride = platform::GetVRegSize();
    uint16_t size = (totalBytes + stride - 1) / stride;
    CopyXToOutVf(inAddr, outAddr, totalBytes, stride, size);
    return;
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__simd_vf__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ComputeOffsetvf(
    __ubuf__ IDX_T* indicesAddr, __ubuf__ int64_t* offsetAddr, uint16_t loopCnt, uint32_t vfLen,
    int64_t maxIndicesValue, int64_t preVarOffset, int64_t colOfsetSzie, int64_t afterAxisSize, uint32_t size,
    uint32_t sizeTwo)
{
    AscendC::Reg::RegTensor<IDX_T> indicesReg;
    AscendC::Reg::RegTensor<IDX_T> invalidIndicesReg;
    AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> indicesCastReg;
    AscendC::Reg::MaskReg maskReg;
    AscendC::Reg::MaskReg maskRegTwo;
    AscendC::Reg::MaskReg cmpMask;
    AscendC::Reg::Duplicate(invalidIndicesReg, static_cast<IDX_T>(-1));
    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = AscendC::Reg::UpdateMask<IDX_T>(size);
        AscendC::Reg::LoadAlign<IDX_T, AscendC::Reg::LoadDist::DIST_NORM>(indicesReg, indicesAddr + i * vfLen);
        AscendC::Reg::Compares<IDX_T, CMPMODE::GE>(cmpMask, indicesReg, 0, maskReg); // 负越界
        AscendC::Reg::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        AscendC::Reg::Compares<IDX_T, CMPMODE::LT>(cmpMask, indicesReg, maxIndicesValue, maskReg); // 正越界
        AscendC::Reg::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        if constexpr (IsSameType<IDX_T, int32_t>::value) {
            maskRegTwo = AscendC::Reg::UpdateMask<int64_t, AscendC::Reg::RegTraitNumTwo>(sizeTwo);
            AscendC::Reg::Cast<int64_t, int32_t, castTraitB32B64>(indicesCastReg, indicesReg, maskRegTwo);
            AscendC::Reg::Muls(indicesCastReg, indicesCastReg, afterAxisSize, maskRegTwo); // 行偏移
            AscendC::Reg::Adds(indicesCastReg, indicesCastReg, colOfsetSzie + preVarOffset,
                               maskRegTwo); // 列偏移+batch偏移
            AscendC::Reg::StoreAlign(offsetAddr + i * vfLen, indicesCastReg, maskRegTwo);
        } else {
            AscendC::Reg::Muls(indicesReg, indicesReg, afterAxisSize, maskReg);               // 行偏移
            AscendC::Reg::Adds(indicesReg, indicesReg, colOfsetSzie + preVarOffset, maskReg); // 列偏移+batch偏移
            AscendC::Reg::StoreAlign(offsetAddr + i * vfLen, indicesReg, maskReg);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ComputeOffset(
    LocalTensor<IDX_T> indicesLocal, LocalTensor<int64_t> offsetLocal, uint32_t indicesCount, int64_t maxIndicesValue,
    int64_t preVarOffset, int64_t colOfsetSzie, int64_t afterAxisSize)
{
    __ubuf__ IDX_T* indicesAddr = (__ubuf__ IDX_T*)indicesLocal.GetPhyAddr();
    __ubuf__ int64_t* offsetAddr = (__ubuf__ int64_t*)offsetLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(IDX_T);
    uint16_t loopCnt = ops::CeilDiv(indicesCount, vfLen);
    uint32_t size = indicesCount;
    uint32_t sizeTwo = indicesCount;

    ComputeOffsetvf(indicesAddr, offsetAddr, loopCnt, vfLen, maxIndicesValue, preVarOffset, colOfsetSzie, afterAxisSize,
                    size, sizeTwo);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__simd_vf__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ComputeMulWithCastVf(
    __ubuf__ VAR_T* updateMulAddr, __ubuf__ VAR_T* updatesAddr, uint16_t loopSize, uint32_t maskLen, int16_t alphaValue,
    uint32_t vfLen)
{
    AscendC::Reg::RegTensor<int16_t> sumReg;
    AscendC::Reg::RegTensor<int16_t> castReg;
    AscendC::Reg::RegTensor<int16_t> alphaReg;
    AscendC::Reg::MaskReg maskReg;
    for (uint16_t j = 0; j < loopSize; j++) {
        maskReg = AscendC::Reg::UpdateMask<int16_t>(maskLen);
        AscendC::Reg::Duplicate(alphaReg, alphaValue, maskReg);
        auto updatesOffet = j * vfLen;
        LoadOneTensorForDtypeIntVf<VAR_T>(updatesAddr, castReg, maskReg, updatesOffet);
        AscendC::Reg::Mul(sumReg, castReg, alphaReg, maskReg);
        StoreOneTensorForDtypeIntVf<VAR_T>(updateMulAddr, sumReg, maskReg, updatesOffet);
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ComputeMulWithCast(
    LocalTensor<VAR_T> updateMulLocal, LocalTensor<VAR_T> updatesLocal, int16_t alphaValue, int64_t colLen)
{
    __ubuf__ VAR_T* updatesAddr = (__ubuf__ VAR_T*)updatesLocal.GetPhyAddr();
    __ubuf__ VAR_T* updateMulAddr = (__ubuf__ VAR_T*)updateMulLocal.GetPhyAddr();

    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(int16_t);
    int32_t loopSize = ops::CeilDiv(static_cast<uint32_t>(colLen), vfLen);
    int32_t idLocation = 0;
    int64_t colLenAlignSize = ops::CeilAlign(colLen * sizeof(VAR_T), UB_AGLIN_VALUE) / sizeof(VAR_T);
    int64_t colLenAlignIn16 = ops::CeilAlign(colLen * sizeof(int16_t), UB_AGLIN_VALUE) / sizeof(int16_t);

    ComputeMulWithCastVf(updateMulAddr, updatesAddr, static_cast<uint16_t>(loopSize), static_cast<uint32_t>(colLen),
                         alphaValue, vfLen);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::CopyInUpdates(LocalTensor<VAR_T> updatesLocal,
                                                                                      int64_t rowLen, int64_t colLen,
                                                                                      uint32_t stride,
                                                                                      int64_t updatesOfset)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(VAR_T)),
                                    stride, static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    DataCopyPad(updatesLocal, updates_[updatesOfset], copyParams, padParams);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ScatterDMA(
    GlobalTensor<VAR_T> dstGm, LocalTensor<VAR_T> updatesLocal, LocalTensor<int64_t> offsetLocal, uint16_t offsetNum,
    uint32_t burstLen, uint32_t srcStride)
{
    ScatterParams scatterDataCopyParams;
    scatterDataCopyParams.offsetNum = offsetNum;
    scatterDataCopyParams.burstLen = burstLen;
    scatterDataCopyParams.srcStride = srcStride;

    ScatterDataCopy(dstGm, updatesLocal, offsetLocal, scatterDataCopyParams);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
template <int8_t SPLIT_MODE>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ProcessSingleLoop(int64_t rowIdx,
                                                                                          int64_t colIdx,
                                                                                          int64_t rowLen,
                                                                                          int64_t colLen)
{
    uint32_t colLenSize = colLen * sizeof(VAR_T);
    uint32_t colLenAlign = ops::CeilAlign(static_cast<uint64_t>(colLenSize), UB_AGLIN_VALUE);
    int64_t colLenAlignSize = static_cast<int64_t>(colLenAlign) / sizeof(VAR_T);

    int64_t indicesOfset;
    if constexpr (SPLIT_MODE == SPLIT_INDICES) {
        indicesOfset = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.ubIndexFactor;
    } else {
        indicesOfset = rowIdx * tilingData_.ubIndexFactor;
    }
    LocalTensor<IDX_T> indicesLocal = indicesQue_.AllocTensor<IDX_T>();
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_[indicesOfset], rowLen);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_[indicesOfset * tilingData_.indicesStride], rowLen, 1,
                                  tilingData_.indicesStride - 1);
    }
    indicesQue_.EnQue(indicesLocal);

    indicesLocal = indicesQue_.DeQue<IDX_T>();
    int64_t BlockCount;
    if constexpr (SPLIT_MODE == SPLIT_AFTER) {
        BlockCount = GetBlockIdx() * tilingData_.eachCoreAfterAxisCount;
    } else {
        BlockCount = 0;
    }
    int64_t colOfset = tilingData_.afterAxisFactor * colIdx;
    int64_t afterAxisSize = tilingData_.afterAxis * sizeof(VAR_T);
    int64_t colOfsetSzie = (BlockCount + colOfset) * sizeof(VAR_T);

    int64_t startPreAxis;
    if constexpr (SPLIT_MODE == SPLIT_PRE) {
        startPreAxis = GetBlockIdx() * tilingData_.eachCorePreAxisCount;
    } else {
        startPreAxis = 0;
    }
    for (int64_t preAxisIdx = startPreAxis; preAxisIdx < startPreAxis + curPreAxisCount_; preAxisIdx++) {
        int64_t rowOfset = (preAxisIdx * tilingData_.updatesInAxis + indicesOfset) * tilingData_.afterAxis;
        int64_t updatesOfset = rowOfset + BlockCount + colOfset;
        LocalTensor<VAR_T> updatesLocal = updatesQue_.AllocTensor<VAR_T>();
        CopyInUpdates(updatesLocal, rowLen, colLen,
                      static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(VAR_T)), updatesOfset);
        updatesQue_.EnQue(updatesLocal);

        updatesLocal = updatesQue_.DeQue<VAR_T>();
        LocalTensor<VAR_T> outLocal = outQue_.AllocTensor<VAR_T>();
        if (tilingData_.isWithAlpha) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, bool>::value) {
                ComputeMulWithCast(outLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
            } else {
                AscendC::Muls(outLocal, updatesLocal, alphaValue_, rowLen * colLenAlignSize);
            }
        } else {
            auto inAddr = reinterpret_cast<__ubuf__ int8_t*>(updatesLocal.GetPhyAddr());
            auto outAddr = reinterpret_cast<__ubuf__ int8_t*>(outLocal.GetPhyAddr());
            CopyXToOut(inAddr, outAddr, rowLen * colLenAlignSize);
        }
        outQue_.EnQue(outLocal);
        updatesQue_.FreeTensor(updatesLocal);

        LocalTensor<int64_t> offsetsLocal = offsetsQue_.AllocTensor<int64_t>();
        int64_t preVarOffset = preAxisIdx * tilingData_.varInAxis * tilingData_.afterAxis * sizeof(VAR_T);
        ComputeOffset(indicesLocal, offsetsLocal, rowLen, tilingData_.varInAxis, preVarOffset, colOfsetSzie,
                      afterAxisSize);
        offsetsQue_.EnQue(offsetsLocal);

        outLocal = outQue_.DeQue<VAR_T>();
        offsetsLocal = offsetsQue_.DeQue<int64_t>();
        if constexpr (IsSameType<VAR_T, bool>::value) {
            SetAtomicMax<int8_t>();
        } else {
            SetAtomicAdd<VAR_T>();
        }
        ScatterDMA(var_, outLocal, offsetsLocal, rowLen, colLenSize, colLenAlign);
        DisableDmaAtomic();
        outQue_.FreeTensor(outLocal);
        offsetsQue_.FreeTensor(offsetsLocal);
    }
    indicesQue_.FreeTensor(indicesLocal);
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ProcessPre()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;

    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessSingleLoop<SPLIT_PRE>(rowIdx, colIdx, rowDataLen, colDataLen);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ProcessIndices()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t rowLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailCoreIndicesLoop :
                                                                                tilingData_.mainCoreIndicesLoop;
    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailCoreTailIndices :
                                                                                    tilingData_.mainCoreTailIndices;

    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessSingleLoop<SPLIT_INDICES>(rowIdx, colIdx, rowDataLen, colDataLen);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::ProcessAfter()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t rowMainDataLen = tilingData_.ubIndexFactor;
    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowTailDataLen = tilingData_.indiceAxisTailNum;

    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
                                                                                tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum :
                                                                                    tilingData_.updateTailNum;
    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            ProcessSingleLoop<SPLIT_AFTER>(rowIdx, colIdx, rowDataLen, colDataLen);
        }
    }
}

template <typename VAR_T, typename IDX_T, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddDma<VAR_T, IDX_T, IS_CONTIGUOUS>::Process()
{
    if (tilingData_.isSplitPreAxis == 1) {
        ProcessPre();
    } else if (tilingData_.isSplitAfterAxis == 1) {
        ProcessAfter();
    } else if (tilingData_.isSplitIndicesAxis == 1) {
        ProcessIndices();
    }
}
} // namespace InplaceIndexAdd

#endif
