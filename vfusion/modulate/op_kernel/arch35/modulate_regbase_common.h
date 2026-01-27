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
 * \file modulate_regbase_common.h
 * \brief modulate
 */

#ifndef MODULATE_REGBASE_COMMON_H
#define MODULATE_REGBASE_COMMON_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"
namespace Modulate {
using namespace AscendC;

static const uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
static const uint32_t NO_DOUBLE_BUFFER = 1;
static const uint32_t DOUBLE_BUFFER = 2;

template<typename T, bool isScale, bool isShift>
class ModulateBaseKernel
{
public:
    __aicore__ inline ModulateBaseKernel(TPipe &tpipe, const ModulateRegbaseTilingData &tilingData) : tpipe_(tpipe), tiling_(tilingData){};
protected:
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void CopyInScaleShift(const uint64_t &dataNum, const uint64_t &scaleOffset);
    __aicore__ inline void CopyInX(const uint64_t &copyRows, const uint64_t &dataNum, const uint64_t &xOffset);
    __aicore__ inline void Compute(const uint64_t &rows, const uint64_t &calCount);
    __aicore__ inline void ComputeScaleShift(
        const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &scaleLocal,
        const LocalTensor<T> &shiftLocal, const uint64_t &rows, const uint64_t &calCount);
    __aicore__ inline void ComputeScale(
        const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &scaleLocal,
        const uint64_t &rows, const uint64_t &calCount);
    __aicore__ inline void ComputeShift(
        const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &shiftLocal,
        const uint64_t &rows, const uint64_t &calCount);
    __aicore__ inline void CopyOutY(const uint64_t &copyRows, const uint64_t &dataNum, const uint64_t &yOffset);
    __aicore__ inline void FreeScaleShift();

protected:
    TPipe& tpipe_;

    // tilingData
    const ModulateRegbaseTilingData &tiling_;

    // basic params
    uint64_t blockIdx_;
    uint64_t inputB_;
    uint64_t inputL_;
    uint64_t inputD_;
    uint64_t formerCoreNum_;
    uint64_t tailCoreNum_;
    uint64_t formerDataNum_;
    uint64_t tailDataNum_;
    uint64_t maxDInUB_;
    uint64_t maxCalcNum_;
    uint64_t maxCopyRows_;

    // block params
    uint64_t dataNum_;
    uint64_t xAlign_;
    uint64_t currentL_;
    uint64_t batchStartId_;
    uint64_t batchEndId_;

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TQue<QuePosition::VECIN, 1> scaleQue_;
    TQue<QuePosition::VECIN, 1> shiftQue_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> scaleGm_;
    GlobalTensor<T> shiftGm_;
};

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::InitBaseParams()
{
    blockIdx_ = GetBlockIdx();
    inputB_ = tiling_.inputB;
    inputL_ = tiling_.inputL;
    inputD_ = tiling_.inputD;
    formerCoreNum_ = tiling_.formerCoreNum;
    tailCoreNum_ = tiling_.tailCoreNum;
    formerDataNum_ = tiling_.formerDataNum;
    tailDataNum_ = tiling_.tailDataNum;
    maxDInUB_ = tiling_.maxDInUB;
    maxCalcNum_ = tiling_.maxCalcNum;
    maxCopyRows_ = tiling_.maxCopyRows;
    xAlign_ = BLOCK_SIZE / sizeof(T);
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::InitBuffers()
{
    uint64_t maxCalcNumAlign = ops::CeilAlign(maxCalcNum_, xAlign_);
    tpipe_.InitBuffer(xQue_, DOUBLE_BUFFER, maxCopyRows_ * maxCalcNumAlign * sizeof(T));
    tpipe_.InitBuffer(yQue_, DOUBLE_BUFFER, maxCopyRows_ * maxCalcNumAlign * sizeof(T));
    if constexpr (isScale) {
        tpipe_.InitBuffer(scaleQue_, NO_DOUBLE_BUFFER, maxCalcNumAlign * sizeof(T));
    }
    if constexpr (isShift) {
        tpipe_.InitBuffer(shiftQue_, NO_DOUBLE_BUFFER, maxCalcNumAlign * sizeof(T));
    }
}

/**
 * @brief 搬入Scale和Shift GM->UB
 * @param dataNum 搬运数据量
 * @param scaleOffset 在scaleGm和shiftGm上的偏移量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::CopyInScaleShift(const uint64_t &dataNum, const uint64_t &scaleOffset)
{
    if constexpr (isScale) {
        LocalTensor<T> scaleLocal = scaleQue_.AllocTensor<T>();
        DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(sizeof(T) * dataNum), 0, 0, 0};
        DataCopyPadExtParams<T> scalePadParams{true, 0, 0, 0};
        DataCopyPad(scaleLocal, scaleGm_[scaleOffset], scaleCopyParams, scalePadParams);
        scaleQue_.EnQue<T>(scaleLocal);
    }
    if constexpr (isShift) {
        LocalTensor<T> shiftLocal = shiftQue_.AllocTensor<T>();
        DataCopyExtParams shiftCopyParams{1, static_cast<uint32_t>(sizeof(T) * dataNum), 0, 0, 0};
        DataCopyPadExtParams<T> shiftPadParams{true, 0, 0, 0};
        DataCopyPad(shiftLocal, shiftGm_[scaleOffset], shiftCopyParams, shiftPadParams);
        shiftQue_.EnQue<T>(shiftLocal);
    }
}

/**
 * @brief 搬入X GM->UB
 * @param copyRows 搬运行数
 * @param dataNum 每行搬运数据量
 * @param xOffset 在xGm上的偏移量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::CopyInX(const uint64_t &copyRows, const uint64_t &dataNum, const uint64_t &xOffset)
{
    LocalTensor<T> xLocal = xQue_.AllocTensor<T>();
    uint8_t xPadNum = (xAlign_ - dataNum % xAlign_) % xAlign_;
    DataCopyExtParams xCopyParams{
        static_cast<uint16_t>(copyRows), static_cast<uint32_t>(sizeof(T) * dataNum),
        static_cast<uint32_t>((inputD_ - dataNum) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> xPadParams{true, 0, xPadNum, 0};
    DataCopyPad(xLocal, xGm_[xOffset], xCopyParams, xPadParams);
    xQue_.EnQue<T>(xLocal);
}

/**
 * @brief 计算主函数，依据模板参数走不同计算分支
 * @param rows 计算行数
 * @param calCount 每行计算数据量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::Compute(const uint64_t &rows, const uint64_t &calCount)
{
    LocalTensor<T> xLocal = xQue_.DeQue<T>();
    LocalTensor<T> yLocal = yQue_.AllocTensor<T>();
    if constexpr (isScale && isShift) {
        LocalTensor<T> scaleLocal = scaleQue_.DeQue<T>();
        LocalTensor<T> shiftLocal = shiftQue_.DeQue<T>();
        ComputeScaleShift(yLocal, xLocal, scaleLocal, shiftLocal, rows, calCount);
        scaleQue_.EnQue<T>(scaleLocal);
        shiftQue_.EnQue<T>(shiftLocal);
    } else if constexpr (isScale) {
        LocalTensor<T> scaleLocal = scaleQue_.DeQue<T>();
        ComputeScale(yLocal, xLocal, scaleLocal, rows, calCount);
        scaleQue_.EnQue<T>(scaleLocal);
    } else if constexpr (isShift) {
        LocalTensor<T> shiftLocal = shiftQue_.DeQue<T>();
        ComputeShift(yLocal, xLocal, shiftLocal, rows, calCount);
        shiftQue_.EnQue<T>(shiftLocal);
    }
    xQue_.FreeTensor<T>(xLocal);
    yQue_.EnQue<T>(yLocal);
}

/**
 * @brief 计算缩放及平移
 * @param yLocal 输出y的LocalTensor
 * @param xLocal 输入x的LocalTensor
 * @param scaleLocal 输入scale的LocalTensor
 * @param shiftLocal 输入shift的LocalTensor
 * @param rows 计算行数
 * @param calCount 每行计算数据量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::ComputeScaleShift(
    const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &scaleLocal,
    const LocalTensor<T> &shiftLocal, const uint64_t &rows, const uint64_t &calCount)
{
    __local_mem__ T* xAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* scaleAddr = (__local_mem__ T*)scaleLocal.GetPhyAddr();
    __local_mem__ T* shiftAddr = (__local_mem__ T*)shiftLocal.GetPhyAddr();
    __local_mem__ T* yAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    using CAST_T = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
    uint32_t dtypeSize = sizeof(CAST_T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (calCount + VL - 1) / VL;
    uint64_t calCountAlign = ops::CeilAlign(calCount, xAlign_);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    uint16_t rowLen = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<CAST_T> xReg;
        MicroAPI::RegTensor<CAST_T> scaleReg;
        MicroAPI::RegTensor<CAST_T> shiftReg;
        MicroAPI::RegTensor<CAST_T> yReg;
        MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < vfLoop; i++) {
            preg = MicroAPI::UpdateMask<CAST_T>(sreg);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ops::LoadTwoTensorForDtypeT<T>(scaleAddr, shiftAddr, scaleReg, shiftReg, preg, preg, i * VL, i * VL);
                MicroAPI::Adds(scaleReg, scaleReg, 1.0f, preg);
                for (uint16_t row = 0; row < rowLen; row++) {
                    uint64_t rowOffset = row * calCountAlign;
                    ops::LoadOneTensorForDtypeT<T>(xAddr, xReg, preg, rowOffset + i * VL);
                    MicroAPI::Mul(xReg, xReg, scaleReg, preg);
                    MicroAPI::Add(yReg, xReg, shiftReg, preg);
                    ops::StoreOneTensorForDtypeT<T>(yAddr, yReg, preg, rowOffset + i * VL);
                }
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(scaleReg, scaleAddr + i * VL);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(shiftReg, shiftAddr + i * VL);
                MicroAPI::Adds(scaleReg, scaleReg, 1.0f, preg);
                for (uint16_t row = 0; row < rowLen; row++) {
                    uint64_t rowOffset = row * calCountAlign;
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(xReg, xAddr + rowOffset + i * VL);
                    MicroAPI::Mul(xReg, xReg, scaleReg, preg);
                    MicroAPI::Add(yReg, xReg, shiftReg, preg);
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(yAddr + rowOffset + i * VL, yReg, preg);
                }
            }
        }
    }
}

/**
 * @brief 仅计算缩放
 * @param yLocal 输出y的LocalTensor
 * @param xLocal 输入x的LocalTensor
 * @param scaleLocal 输入scale的LocalTensor
 * @param rows 计算行数
 * @param calCount 每行计算数据量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::ComputeScale(
    const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &scaleLocal,
    const uint64_t &rows, const uint64_t &calCount)
{
    __local_mem__ T* xAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* scaleAddr = (__local_mem__ T*)scaleLocal.GetPhyAddr();
    __local_mem__ T* yAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    using CAST_T = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
    uint32_t dtypeSize = sizeof(CAST_T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (calCount + VL - 1) / VL;
    uint64_t calCountAlign = ops::CeilAlign(calCount, xAlign_);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    uint16_t rowLen = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<CAST_T> xReg;
        MicroAPI::RegTensor<CAST_T> scaleReg;
        MicroAPI::RegTensor<CAST_T> yReg;
        MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < vfLoop; i++) {
            preg = MicroAPI::UpdateMask<CAST_T>(sreg);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ops::LoadOneTensorForDtypeT<T>(scaleAddr, scaleReg, preg, i * VL);
                MicroAPI::Adds(scaleReg, scaleReg, 1.0f, preg);
                for (uint16_t row = 0; row < rowLen; row++) {
                    uint64_t rowOffset = row * calCountAlign;
                    ops::LoadOneTensorForDtypeT<T>(xAddr, xReg, preg, rowOffset + i * VL);
                    MicroAPI::Mul(yReg, xReg, scaleReg, preg);
                    ops::StoreOneTensorForDtypeT<T>(yAddr, yReg, preg, rowOffset + i * VL);
                }
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(scaleReg, scaleAddr + i * VL);
                MicroAPI::Adds(scaleReg, scaleReg, 1.0f, preg);
                for (uint16_t row = 0; row < static_cast<uint16_t>(rows); row++) {
                    uint64_t rowOffset = row * calCountAlign;
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(xReg, xAddr + rowOffset + i * VL);
                    MicroAPI::Mul(yReg, xReg, scaleReg, preg);
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(yAddr + rowOffset + i * VL, yReg, preg);
                }
            }
        }
    }
}

/**
 * @brief 仅计算平移
 * @param yLocal 输出y的LocalTensor
 * @param xLocal 输入x的LocalTensor
 * @param shiftLocal 输入shift的LocalTensor
 * @param rows 计算行数
 * @param calCount 每行计算数据量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::ComputeShift(
    const LocalTensor<T> &yLocal, const LocalTensor<T> &xLocal, const LocalTensor<T> &shiftLocal,
    const uint64_t &rows, const uint64_t &calCount)
{
    __local_mem__ T* xAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* shiftAddr = (__local_mem__ T*)shiftLocal.GetPhyAddr();
    __local_mem__ T* yAddr = (__local_mem__ T*)yLocal.GetPhyAddr();
    using CAST_T = std::conditional_t<std::is_same_v<T, bfloat16_t>, float, T>;
    uint32_t dtypeSize = sizeof(CAST_T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (calCount + VL - 1) / VL;
    uint64_t calCountAlign = ops::CeilAlign(calCount, xAlign_);
    uint32_t sreg = static_cast<uint32_t>(calCount);
    uint16_t rowLen = static_cast<uint16_t>(rows);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<CAST_T> xReg;
        MicroAPI::RegTensor<CAST_T> shiftReg;
        MicroAPI::RegTensor<CAST_T> yReg;
        MicroAPI::MaskReg preg;
        for (uint16_t i = 0; i < vfLoop; i++) {
            preg = MicroAPI::UpdateMask<CAST_T>(sreg);
            if constexpr (IsSameType<T, bfloat16_t>::value) {
                ops::LoadOneTensorForDtypeT<T>(shiftAddr, shiftReg, preg, i * VL);
                for (uint16_t row = 0; row < rowLen; row++) {
                    uint64_t rowOffset = row * calCountAlign;
                    ops::LoadOneTensorForDtypeT<T>(xAddr, xReg, preg, rowOffset + i * VL);
                    MicroAPI::Add(yReg, xReg, shiftReg, preg);
                    ops::StoreOneTensorForDtypeT<T>(yAddr, yReg, preg, rowOffset + i * VL);
                }
            } else {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(shiftReg, shiftAddr + i * VL);
                for (uint16_t row = 0; row < static_cast<uint16_t>(rows); row++) {
                    uint64_t rowOffset = row * calCountAlign;
                    MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(xReg, xAddr + rowOffset + i * VL);
                    MicroAPI::Add(yReg, xReg, shiftReg, preg);
                    MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM>(yAddr + rowOffset + i * VL, yReg, preg);
                }
            }
        }
    }
}

/**
 * @brief 搬出y UB->GM
 * @param copyRows 搬运行数
 * @param dataNum 每行搬运数据量
 * @param yOffset 在yGm上的偏移量
 */
template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::CopyOutY(const uint64_t &copyRows, const uint64_t &dataNum, const uint64_t &yOffset)
{
    LocalTensor<T> yLocal = yQue_.DeQue<T>();
    DataCopyExtParams yCopyParams{
        static_cast<uint16_t>(copyRows), static_cast<uint32_t>(sizeof(T) * dataNum), 0, 0, 0};
    DataCopyPad(yGm_[yOffset], yLocal, yCopyParams);
    yQue_.FreeTensor<T>(yLocal);
}

template<typename T, bool isScale, bool isShift>
__aicore__ inline void ModulateBaseKernel<T, isScale, isShift>::FreeScaleShift()
{
    if constexpr (isScale) {
        LocalTensor<T> scaleLocal = scaleQue_.DeQue<T>();
        scaleQue_.FreeTensor<T>(scaleLocal);
    }
    if constexpr (isShift) {
        LocalTensor<T> shiftLocal = shiftQue_.DeQue<T>();
        shiftQue_.FreeTensor<T>(shiftLocal);
    }
}
}
#endif // MODULATE_REGBASE_COMMON_H