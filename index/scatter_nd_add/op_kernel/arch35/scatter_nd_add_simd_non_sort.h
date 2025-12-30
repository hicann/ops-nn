/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_simd_non_sort.h
 * \brief simd kernel of scatter_nd_update
 */

#ifndef SCATTER_ND_UPDATE_SIMD_NON_SORT_H
#define SCATTER_ND_UPDATE_SIMD_NON_SORT_H

#include "scatter_nd_add_common.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"

namespace ScatterNdAdd {
using namespace AscendC;

template <typename T, typename U>
class ScatterNdAddSimdNoSort : public ScatterNdAddBase<T, U> {
public:
    __aicore__ inline ScatterNdAddSimdNoSort(const ScatterNdAddRegBaseTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyInVar(LocalTensor<T>& dstTensor, int64_t varOffset, int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyOutVar(LocalTensor<T>& dstTensor, int64_t varOffset, int64_t rowLen, int64_t colLen);
    __aicore__ inline void ComputeWithCast(LocalTensor<T> yLocal, LocalTensor<T> updatesLocal, int64_t dataLen);
    __aicore__ inline void ProcessSplitAfter();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<T> yGm_;

    TBuf<QuePosition::VECCALC> varBuf_;
    TBuf<QuePosition::VECCALC> varTypeAscentBuf_;
    TBuf<QuePosition::VECCALC> updatesTypeAscentBuf_;
    TBuf<QuePosition::VECCALC> varDataBuf_;
    TBuf<QuePosition::VECCALC> updatesDataBuf_;

    TPipe& pipe_;
    const ScatterNdAddRegBaseTilingData& tilingData_;

    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};  // bf16 --float

    constexpr static AscendC::MicroAPI::CastTrait castTrait1 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};  // float---bf16 

    int64_t curCoreIndexCount_{0};
    uint64_t strideList[MAX_RANK_COUNT];
    static constexpr int64_t colAlignPerBlock_ = platform::GetUbBlockSize() / sizeof(T);
};

template <typename T, typename U>
__aicore__ inline void ScatterNdAddSimdNoSort<T, U>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    varGm_.SetGlobalBuffer((__gm__ T *)(var));
    this->indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
    this->updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    yGm_.SetGlobalBuffer((__gm__ T *)(y));


    this->indicesFactor_ = tilingData_.indicesFactor;
    this->afterAxis_ = tilingData_.afterAxis;
    this->afterAxisFactor_ = tilingData_.afterAxisFactor;
    this->indexRankSize_ = tilingData_.indexRankSize;
    this->eachCoreAfterAxisCount_ = tilingData_.eachCoreAfterAxisCount;

    
    pipe_.InitBuffer(this->indicesBuf_, ROUND_UP32(tilingData_.indicesFactor * tilingData_.indexRankSize * sizeof(U)));
    pipe_.InitBuffer(this->dataQueue_, 1, tilingData_.indicesFactor * ROUND_UP32( tilingData_.afterAxisFactor * sizeof(T)));
    pipe_.InitBuffer(varBuf_, ROUND_UP32(tilingData_.varInAxis * tilingData_.afterAxisFactor * sizeof(T)));
    // 计算完偏移存储indice
    pipe_.InitBuffer(this->outOfstBuf_, ROUND_UP32(tilingData_.indicesFactor * sizeof(U)));
    pipe_.InitBuffer(this->strideBuf_, MAX_RANK_COUNT * sizeof(U));

    curCoreIndexCount_ = (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount :
                                                                                 tilingData_.tailCoreIndexCount);
}


template <typename T, typename U>
__aicore__ inline void ScatterNdAddSimdNoSort<T, U>::CopyInVar(LocalTensor<T>& dstTensor, int64_t varOffset, int64_t rowLen, int64_t colLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen),
                                        static_cast<uint32_t>(colLen * sizeof(T)),
                                        static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(T)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> updatePadParams = {false, 0, 0, 0};
    DataCopyPad(dstTensor, yGm_[varOffset], copyParams, updatePadParams);
}


template <typename T, typename U>
__aicore__ inline void ScatterNdAddSimdNoSort<T, U>::CopyOutVar(LocalTensor<T>& dstTensor, int64_t varOffset, int64_t rowLen, int64_t colLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen),
                                        static_cast<uint32_t>(colLen * sizeof(T)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>((tilingData_.afterAxis - colLen) * sizeof(T)),
                                        static_cast<uint32_t>(0)};
    DataCopyPad(yGm_[varOffset], dstTensor, copyParams);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddSimdNoSort<T, U>::ComputeWithCast(LocalTensor<T> yLocal, LocalTensor<T> updatesLocal, int64_t dataLen)
{
        __local_mem__ T* yLocalAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
        __local_mem__ T* updatesLocalAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
        
        int64_t vfLen = platform::GetVRegSize() / sizeof(float);
        int64_t loopSize = ops::CeilDiv(static_cast<int64_t>(dataLen), vfLen);
        uint32_t size = static_cast<uint32_t>(dataLen);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> vreg_x1;
            MicroAPI::RegTensor<T> vreg_x2;
            MicroAPI::RegTensor<float> vreg0;
            MicroAPI::RegTensor<float> vreg1;
            MicroAPI::RegTensor<float> sumReg;
            MicroAPI::RegTensor<T> vregy;
            MicroAPI::MaskReg maskReg;

            for (uint16_t i = 0; i < static_cast<uint16_t>(loopSize); i++) {
                maskReg = MicroAPI::UpdateMask<float>(size);
                uint32_t offset = i * vfLen;
                ops::LoadOneTensorForDtypeT<T>(yLocalAddr, vreg0, maskReg, offset);
                ops::LoadOneTensorForDtypeT<T>(updatesLocalAddr, vreg1, maskReg, offset);
                MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                    sumReg, vreg0, vreg1, maskReg);
                ops::StoreOneTensorForDtypeT<T>(yLocalAddr, sumReg, maskReg, offset);
            }
        }
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddSimdNoSort<T, U>::ProcessSplitAfter()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
     
    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize
                                                                              : tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum
                                                                                  : tilingData_.updateTailNum;
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = tilingData_.indiceTailNum;
    int64_t rowLoopNum = tilingData_.indicesLoopSize;

    LocalTensor<T> yLocal = varBuf_.template Get<T>();
    for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
        int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
        int64_t varOffset =  GetBlockIdx() * this->eachCoreAfterAxisCount_ + colIdx * colDataLen;

        int64_t colAlignBlock= ops::Aligned(colDataLen, colAlignPerBlock_);      
        CopyInVar(yLocal, varOffset, tilingData_.varInAxis, colDataLen);

        for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
            int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
            this->ProcessAfterSingleNonSort(rowIdx, colIdx, rowDataLen, colDataLen);
            LocalTensor<T> updatesLocal = this->dataQueue_.template DeQue<T>();
            LocalTensor<U> outOfstLocal = this->outOfstBuf_.template Get<U>();
            
            event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

            if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                //类型提升
                for (int64_t i = 0; i < rowDataLen; i++){
                    int64_t yOffset = outOfstLocal(i) * colAlignBlock;
                    int64_t updatesOffset = i * colAlignBlock;
                    LocalTensor<T> varRow = yLocal[yOffset];
                    LocalTensor<T> updatesRow = updatesLocal[updatesOffset];
                    ComputeWithCast(varRow, updatesRow, colDataLen);
                }
            } else {
                for (int64_t i = 0; i < rowDataLen; i++){
                    int64_t yOffset = outOfstLocal(i) * colAlignBlock;
                    int64_t updatesOffset = i * colAlignBlock;
                    Add(yLocal[yOffset], yLocal[yOffset], updatesLocal[updatesOffset], colDataLen);
                }
            }
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            this->dataQueue_.template FreeTensor(updatesLocal);
        }

        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        CopyOutVar(yLocal, varOffset, tilingData_.varInAxis, colDataLen);

        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddSimdNoSort<T, U>::Process()
{
    LocalTensor<U> strideLocal = this->strideBuf_.template Get<U>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideLocal(i) = tilingData_.strideList[i];
    }
    ProcessSplitAfter();
}
}  // namespace ScatterNdAdd
#endif  // SCATTER_ND_ADD_SIMD_H