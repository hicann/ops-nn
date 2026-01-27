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
 * \file modulate_regbase.h
 * \brief modulate
 */

#ifndef MODULATE_REGBASE_H
#define MODULATE_REGBASE_H

#include "modulate_regbase_common.h"
namespace Modulate {
using namespace AscendC;

// tiling 切 L
template<typename T, bool isScale, bool isShift>
class ModulateL : public ModulateBaseKernel<T, isScale, isShift>
{
public:
    __aicore__ inline ModulateL(TPipe &tpipe, const ModulateRegbaseTilingData &tilingData) :
                                ModulateBaseKernel<T, isScale, isShift>(tpipe, tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y);
    __aicore__ inline void Process();
protected:
    __aicore__ inline void InitParams();
    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y);
};

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateL<T, isScale, isShift>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
{
    InitParams();
    this->InitBuffers();
    SetGmAddr(x, scale, shift, y);
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateL<T, isScale, isShift>::InitParams()
{
    // 初始化基本参数
    this->InitBaseParams();

    // 区分大小核
    if (this->blockIdx_ < this->formerCoreNum_) {
        this->dataNum_ = this->formerDataNum_;
        this->currentL_ = this->blockIdx_ * this->formerDataNum_;
    } else {
        this->dataNum_ = this->tailDataNum_;
        this->currentL_ = this->formerCoreNum_ * this->formerDataNum_ +
                          (this->blockIdx_ - this->formerCoreNum_) * this->tailDataNum_;
    }
    this->batchStartId_ = this->currentL_ / this->inputL_;
    this->batchEndId_ = (this->currentL_ + this->dataNum_ - 1) / this->inputL_;
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateL<T, isScale, isShift>::SetGmAddr(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
{
    uint64_t baseOffset = this->currentL_ * this->inputD_;
    uint64_t scaleShiftOffset = this->batchStartId_ * this->inputD_;
    this->xGm_.SetGlobalBuffer((__gm__ T*)x + baseOffset);
    this->yGm_.SetGlobalBuffer((__gm__ T*)y + baseOffset);
    this->scaleGm_.SetGlobalBuffer((__gm__ T*)scale + scaleShiftOffset);
    this->shiftGm_.SetGlobalBuffer((__gm__ T*)shift + scaleShiftOffset);
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateL<T, isScale, isShift>::Process()
{
    this->maxCopyRows_ = this->dataNum_ < this->maxCopyRows_ ? this->dataNum_ : this->maxCopyRows_;
    uint64_t LId = 0;
    for (uint64_t BId = 0; BId < this->batchEndId_ - this->batchStartId_ + 1; BId++) {
        // 当前batch剩余L
        uint64_t remainL = BId ? this->inputL_ : ((this->batchStartId_ + 1) * this->inputL_ - this->currentL_);
        uint64_t endL = (LId + remainL < this->dataNum_) ? (LId + remainL) : this->dataNum_;
        // 处理D方向
        uint64_t numD = 0;
        for (uint64_t DId = 0; DId < this->inputD_; DId += numD) {
            numD = (DId + this->maxCalcNum_ > this->inputD_) ? (this->inputD_ - DId) : this->maxCalcNum_;
            uint64_t scaleOffset = BId * this->inputD_ + DId;
            this->CopyInScaleShift(numD, scaleOffset);
            // 处理L方向, 结束循环应刚好等于endL
            uint64_t copyRows = 0;
            for (uint64_t LIdPerD = LId; LIdPerD < endL; LIdPerD += copyRows) {
                copyRows = (LIdPerD + this->maxCopyRows_ > endL) ? (endL - LIdPerD) : this->maxCopyRows_;
                uint64_t xOffset = LIdPerD * this->inputD_ + DId;
                this->CopyInX(copyRows, numD, xOffset);
                this->Compute(copyRows, numD);
                this->CopyOutY(copyRows, numD, xOffset);
            }
            this->FreeScaleShift();
        }
        LId = endL;
    }
}

// tiling 切 D
template<typename T, bool isScale, bool isShift>
class ModulateD : public ModulateBaseKernel<T, isScale, isShift>
{
public:
    __aicore__ inline ModulateD(TPipe &tpipe, const ModulateRegbaseTilingData &tilingData) :
                                ModulateBaseKernel<T, isScale, isShift>(tpipe, tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y);
    __aicore__ inline void Process();
protected:
    __aicore__ inline void InitParams();
    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y);
private:
    uint64_t currentD_;
};

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateD<T, isScale, isShift>::Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
{
    InitParams();
    this->InitBuffers();
    SetGmAddr(x, scale, shift, y);
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateD<T, isScale, isShift>::InitParams()
{
    // 初始化基本参数
    this->InitBaseParams();

    // 区分大小核
    if (this->blockIdx_ < this->formerCoreNum_) {
        this->dataNum_ = this->formerDataNum_;
        currentD_ = this->blockIdx_ * this->formerDataNum_;
    } else {
        this->dataNum_ = this->tailDataNum_;
        currentD_ = this->formerCoreNum_ * this->formerDataNum_ +
                          (this->blockIdx_ - this->formerCoreNum_) * this->tailDataNum_;
    }
    this->batchStartId_ = 0;
    this->batchEndId_ = this->inputB_;
    this->maxCalcNum_ = this->dataNum_ < this->maxCalcNum_ ? this->dataNum_ : this->maxCalcNum_;
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateD<T, isScale, isShift>::SetGmAddr(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
{
    this->xGm_.SetGlobalBuffer((__gm__ T*)x + currentD_);
    this->yGm_.SetGlobalBuffer((__gm__ T*)y + currentD_);
    this->scaleGm_.SetGlobalBuffer((__gm__ T*)scale + currentD_);
    this->shiftGm_.SetGlobalBuffer((__gm__ T*)shift + currentD_);
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateD<T, isScale, isShift>::Process()
{
    for (uint64_t BId = 0; BId < this->inputB_; BId++) {
        // 处理D方向
        uint64_t numD = 0;
        for (uint64_t DId = 0; DId < this->dataNum_; DId += numD) {
            numD = (DId + this->maxCalcNum_ > this->dataNum_) ? (this->dataNum_ - DId) : this->maxCalcNum_;
            uint64_t scaleOffset = BId * this->inputD_ + DId;
            this->CopyInScaleShift(numD, scaleOffset);
            // 处理L方向
            for (uint64_t LId = 0; LId < this->inputL_; LId++) {
                uint64_t xOffset = BId * this->inputL_ * this->inputD_ + LId * this->inputD_ + DId;
                this->CopyInX(1, numD, xOffset);
                this->Compute(1, numD);
                this->CopyOutY(1, numD, xOffset);
            }
            this->FreeScaleShift();
        }
    }
}
}
#endif // MODULATE_REGBASE_H