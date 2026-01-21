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
 * \file max_pool3d_grad_common.h
 * \brief
 */
#ifndef MAX_POOL3D_GRAD_COMMON_H
#define MAX_POOL3D_GRAD_COMMON_H

#include "kernel_tiling/kernel_tiling.h"

namespace MaxPool3DGradCommon {
using namespace AscendC;

// 公共常量
constexpr uint64_t TRANS_ADDR_LEN = 16;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BLOCK_NUM_16 = BLOCK_SIZE / sizeof(half);
constexpr uint64_t BLOCK_NUM_32 = BLOCK_SIZE / sizeof(float);
constexpr float ZERO = 0.0f;
constexpr uint64_t UINT8_BITS = 8;
constexpr uint32_t UNIT_BLOCK_LEN = 32;

// 公共模板结构体
template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename, typename>
struct is_same : public false_type {};

template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

// 公共函数
__aicore__ inline uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

// 公共结构体
struct BlockParamsCommon {
    uint64_t ncCntIndex = 0;
    uint64_t doCntIndex = 0;
    uint64_t hoCntIndex = 0;
    uint64_t woCntIndex = 0;
    uint64_t ncShape = 0;
    uint64_t doShape = 0;
    uint64_t hoShape = 0;
    uint64_t woShape = 0;
    uint64_t diShape = 0;
    uint64_t hiShape = 0;
    uint64_t wiShape = 0;
    uint64_t diValid = 0;
    uint64_t hiValid = 0;
    uint64_t wiValid = 0;
    uint64_t dohowoShape = 0;
    uint64_t dihiwiAlign = 0;
    uint64_t dohowoAlign8 = 0;
    uint64_t dohowoAlign16 = 0;
    uint64_t offsetX = 0;
    uint64_t offsetGrad = 0;
    uint64_t offsetArgmax = 0;
    uint64_t offsetY = 0;
    uint64_t baseNcOffset = 0;
    uint64_t ShapeSum = 0;
};

struct TilingParamsCommon {
    uint64_t ncDim;
    uint64_t diDim;
    uint64_t hiDim;
    uint64_t wiDim;
    uint64_t doDim;
    uint64_t hoDim;
    uint64_t woDim;
    uint64_t singleCoreNc;
    uint64_t singleCoreDo;
    uint64_t singleCoreHo;
    uint64_t singleCoreWo;
    uint64_t baseNc;
    uint64_t baseDo;
    uint64_t baseHo;
    uint64_t baseWo;
    uint64_t ncCnt;
    uint64_t ncTail;
    uint64_t doTail;
    uint64_t hoTail;
    uint64_t woTail;
    uint64_t totalCnt;
    uint64_t needInitOutput;
    uint64_t usedCoreNum;
    uint64_t preCoreNum;
    uint64_t round;
    uint64_t realRound;
    uint64_t ncIndex;
    uint64_t ncCntRound;
    uint64_t ncRealRound;
    uint64_t diHiWiLen;
    uint64_t ubSize;
    uint64_t initLen;
    uint64_t initOffset;
    uint64_t doCnt;
    uint64_t hoCnt;
    uint64_t woCnt;
};

// [row, col] -> [col, row]: row:align16, col:align8
// only support float/int32_t
template <typename T>
__aicore__ inline void TransposeBase16M8(
    const LocalTensor<T>& dstUb, const LocalTensor<T>& srcUb, uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = rowNum;
        }

        TransDataTo5HD<T>(dstAddrList, srcAddrList, transDataParams);
    }
}

// [row, col] -> [col, row]: row:align8, col:align16
// only support float/int32_t
template <typename T>
__aicore__ inline void TransposeBase8M16(
    const LocalTensor<T>& dstUb, const LocalTensor<T>& srcUb, uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];

    for (uint64_t r = 0; r < colNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] =
                (uint64_t)(srcUb[r * TRANS_ADDR_LEN + i % BLOCK_NUM_32 * colNum + i / BLOCK_NUM_32 * BLOCK_NUM_32]
                               .GetPhyAddr());
            dstAddrList[i] =
                (uint64_t)(dstUb[r * TRANS_ADDR_LEN * rowNum + (i % 2 * BLOCK_NUM_32 + i / 2) * rowNum].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = rowNum / BLOCK_NUM_32;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = colNum;
            transDataParams.dstRepStride = 1;
        }

        TransDataTo5HD<T>(dstAddrList, srcAddrList, transDataParams);
    }
}

// [row, col] -> [col, row]: row:align16, col:align16
// only support float16/bfloat16
template <typename T>
__aicore__ inline void TransposeBase16M16(
    const LocalTensor<T>& dstUb, const LocalTensor<T>& srcUb, uint64_t rowNum, uint64_t colNum)
{
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];

    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i * rowNum].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_16;
        if (transDataParams.repeatTimes == 1) {
            transDataParams.srcRepStride = 0;
            transDataParams.dstRepStride = 0;
        } else {
            transDataParams.srcRepStride = 1;
            transDataParams.dstRepStride = rowNum;
        }
        // TransData5HD does not support bfloat16, only use half
        TransDataTo5HD<half>(dstAddrList, srcAddrList, transDataParams);
    }
}

template <typename ParamsType, typename BlockType, typename DerivedClass>
__aicore__ inline void ProcessCommon(DerivedClass* self, uint64_t ncIndex)
{
    for (uint64_t i = 0; i < self->params_.ncRealRound; i++) {
        if (ncIndex > self->params_.ncCnt) {
            break;
        }
        self->block_.ncCntIndex = ncIndex;
        self->block_.ncShape =
            self->block_.ncCntIndex >= (self->params_.ncCnt - 1UL) ? 
            self->params_.ncTail : self->params_.baseNc;
            
        for (uint64_t j = 0; j < self->params_.doCnt; j++) {
            self->block_.doCntIndex = j;
            self->block_.doShape = 
                self->block_.doCntIndex >= (self->params_.doCnt - 1) ? 
                self->params_.doTail : self->params_.baseDo;
                
            for (uint64_t k = 0; k < self->params_.hoCnt; k++) {
                self->block_.hoCntIndex = k;
                self->block_.hoShape = 
                    self->block_.hoCntIndex >= (self->params_.hoCnt - 1) ?
                    self->params_.hoTail : self->params_.baseHo;
                    
                for (uint64_t l = 0; l < self->params_.woCnt; l++) {
                    self->block_.woCntIndex = l;
                    self->block_.woShape = 
                        self->block_.woCntIndex >= (self->params_.woCnt - 1) ?
                        self->params_.woTail : self->params_.baseWo;
                        
                    self->block_.offsetGrad =
                        self->block_.ncCntIndex * self->params_.baseNc * self->params_.doDim *
                            self->params_.hoDim * self->params_.woDim +
                        self->block_.doCntIndex * self->params_.baseDo * self->params_.hoDim *
                            self->params_.woDim +
                        self->block_.hoCntIndex * self->params_.baseHo * self->params_.woDim +
                        self->block_.woCntIndex * self->params_.baseWo;
                    self->block_.offsetArgmax = self->block_.offsetGrad;
                    
                    // 调用派生类的CalcBlock函数
                    self->CalcBlock();
                }
            }
        }
        ncIndex += 1;
    }
}

} // namespace MaxPool3DGradCommon

#endif // MAX_POOL3D_GRAD_COMMON_H