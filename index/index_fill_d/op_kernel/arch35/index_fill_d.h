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
 * \file index_fill_d.h
 * \brief
 */

#ifndef CANN_CUSTOM_OPS_INDEX_FILL_D_H
#define CANN_CUSTOM_OPS_INDEX_FILL_D_H
 

using namespace AscendC;
static const int BUFFER_NUM = 2;
template<typename T>
class IndexFillD {
public:
    __aicore__ inline IndexFillD(const IndexFillDTilingData& tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR assist1, GM_ADDR assist2, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);
    __aicore__ inline void Compute(int64_t dataLen);
    __aicore__ inline void Process();

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T> assist1Gm_;
    GlobalTensor<T> assist2Gm_;
    GlobalTensor<T> yGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> assist1Queue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> assist2Queue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TPipe &pipe_;
    const IndexFillDTilingData& tilingData_;
};

template<typename T>
__aicore__ inline void IndexFillD<T>::Init(GM_ADDR x, GM_ADDR assist1, GM_ADDR assist2, GM_ADDR y, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ T *)(x) + GetBlockIdx() * tilingData_.normalCoreData);
    assist1Gm_.SetGlobalBuffer((__gm__ T *)(assist1) + GetBlockIdx() * tilingData_.normalCoreData);
    assist2Gm_.SetGlobalBuffer((__gm__ T *)(assist2) + GetBlockIdx() * tilingData_.normalCoreData);
    yGm_.SetGlobalBuffer((__gm__ T *)(y) + GetBlockIdx() * tilingData_.normalCoreData);
    pipe_.InitBuffer(xQueue_, BUFFER_NUM, tilingData_.ubFactor * sizeof(T));
    pipe_.InitBuffer(assist1Queue_, BUFFER_NUM, tilingData_.ubFactor * sizeof(T));
    pipe_.InitBuffer(assist2Queue_, BUFFER_NUM, tilingData_.ubFactor * sizeof(T));
    pipe_.InitBuffer(yQueue_, BUFFER_NUM, tilingData_.ubFactor * sizeof(T));
}

template<typename T>
__aicore__ inline void IndexFillD<T>::CopyIn(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    DataCopyPadExtParams<T> padParams = { false, 0, 0, false };
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    DataCopyPad(xLocal, xGm_[offset], inParams, padParams);
    xQueue_.EnQue(xLocal);

    LocalTensor<T> assist1Local = assist1Queue_.AllocTensor<T>();
    DataCopyPad(assist1Local, assist1Gm_[offset], inParams, padParams);
    assist1Queue_.EnQue(assist1Local);

    LocalTensor<T> assist2Local = assist2Queue_.AllocTensor<T>();
    DataCopyPad(assist2Local, assist2Gm_[offset], inParams, padParams);
    assist2Queue_.EnQue(assist2Local);
}

template<typename T>
__aicore__ inline void IndexFillD<T>::CopyOut(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    LocalTensor<T> yLocal = yQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, outParams);
    yQueue_.FreeTensor(yLocal);
}

template<typename T, typename DTYPE>
static __aicore__ inline void CompareVf(LocalTensor<T> &xLocal, LocalTensor<T> &assist1Local,
                                LocalTensor<T> &assist2Local, LocalTensor<T> &yLocal,
                                uint32_t count, uint16_t onRepeatSize, uint16_t repeatTimes)
{
    __local_mem__ DTYPE* xPtr = (__local_mem__ DTYPE*)xLocal.GetPhyAddr();
    __local_mem__ DTYPE* assist1Ptr = (__local_mem__ DTYPE*)assist1Local.GetPhyAddr();
    __local_mem__ DTYPE* assist2Ptr = (__local_mem__ DTYPE*)assist2Local.GetPhyAddr();
    __local_mem__ DTYPE* yPtr = (__local_mem__ DTYPE*)yLocal.GetPhyAddr();
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<DTYPE> vSrcRegX;
        MicroAPI::RegTensor<DTYPE> vSrcRegAssist1;
        MicroAPI::RegTensor<DTYPE> vSrcRegAssist2;
        MicroAPI::RegTensor<DTYPE> vDstRegY;
        MicroAPI::MaskReg cmpMaskReg;
        for(uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<DTYPE>(count);
            MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<DTYPE, MicroAPI::MaskPattern::ALL>();
            MicroAPI::DataCopy(vSrcRegX, xPtr + i * onRepeatSize);
            MicroAPI::DataCopy(vSrcRegAssist1, assist1Ptr + i * onRepeatSize);
            MicroAPI::DataCopy(vSrcRegAssist2, assist2Ptr + i * onRepeatSize);
            MicroAPI::CompareScalar<DTYPE, CMPMODE::GT>(cmpMaskReg, vSrcRegAssist1, (DTYPE)0, maskAll);
            MicroAPI::Select(vDstRegY, vSrcRegX, vSrcRegAssist2, cmpMaskReg);
            MicroAPI::DataCopy(yPtr + i * onRepeatSize, vDstRegY, maskReg);
        }
    }
}

template<typename T>
__aicore__ inline void IndexFillD<T>::Compute(int64_t dataLen)
{
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<T> assist1Local = assist1Queue_.DeQue<T>();
    LocalTensor<T> assist2Local = assist2Queue_.DeQue<T>();
    LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();
    constexpr uint16_t onRepeatSize = GetVecLen() / sizeof(T);
    uint16_t repeatNum = CeilDivision(dataLen, onRepeatSize);
    if constexpr (std::is_same<T, bool>::value) {
        CompareVf<T, int8_t>(xLocal, assist1Local, assist2Local, yLocal, (uint32_t)dataLen, onRepeatSize, repeatNum);
    } else {
        CompareVf<T, T>(xLocal, assist1Local, assist2Local, yLocal, (uint32_t)dataLen, onRepeatSize, repeatNum);
    }
    yQueue_.EnQue(yLocal);
    xQueue_.FreeTensor(xLocal);
    assist1Queue_.FreeTensor(assist1Local);
    assist2Queue_.FreeTensor(assist2Local);
}

template<typename T>
__aicore__ inline void IndexFillD<T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    int64_t loopSize = tilingData_.normalCoreLoop;
    int64_t tailUbFactor = tilingData_.tailUbFactor;
    int64_t curCoreHandleData = tilingData_.normalCoreData;
    if (GetBlockIdx() == GetBlockNum() - 1) {
        loopSize = tilingData_.tailCoreLoop;
        curCoreHandleData = tilingData_.tailCoreData;
        tailUbFactor = tilingData_.tailCoreTailUbFactor;
    }
    int64_t offset = 0;
    int64_t dataLen = tilingData_.ubFactor;
    for (int64_t loopIdx = 0; loopIdx < loopSize - 1; loopIdx++) {
        CopyIn(offset, dataLen);
        Compute(dataLen);
        CopyOut(offset, dataLen);
        offset += dataLen;
    }
    dataLen = tailUbFactor;
    CopyIn(offset, dataLen);
    Compute(dataLen);
    CopyOut(offset, dataLen);
}

#endif  // CANN_CUSTOM_OPS_INDEX_FILL_D_H
