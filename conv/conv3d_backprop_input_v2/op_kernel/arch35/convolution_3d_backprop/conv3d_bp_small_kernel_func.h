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
 * \file conv3d_bp_small_kernel_func.h
 * \brief Small-kernel member function implementations for Conv3dDxSmallKernel.
 */

#ifndef CONV3D_BP_SMALL_KERNEL_FUNC_ADVANCE_H
#define CONV3D_BP_SMALL_KERNEL_FUNC_ADVANCE_H

#include "../../../inc/macro.h"

template <typename srcType>
__aicore__ inline void InitL1ZeroValue(const LocalTensor<srcType>& tensor, bool useOffsetX = false)
{
    uint32_t len = tensor.GetSize() * sizeof(srcType);
    uint16_t padValue = 0;
    if constexpr (std::is_same<srcType, int8_t>::value) {
        if (useOffsetX) {
            uint8_t offsetX = static_cast<uint8_t>(tiling_->offsetX);
            padValue = (static_cast<uint16_t>(offsetX)) << 8 | (static_cast<uint16_t>(offsetX));
        }
    }
    if constexpr (std::is_same<srcType, hifloat8_t>::value || std::is_same<srcType, fp8_e4m3fn_t>::value ||
                  std::is_same<srcType, int8_t>::value) {
        Fill(tensor.template ReinterpretCast<uint16_t>(), {1, static_cast<uint16_t>(len / ONE_BLK_SIZE), 0, padValue});
    } else {
        AscendC::InitConstValueParams<srcType> initParams;
        initParams.repeatTimes = 1;
        initParams.blockNum = len / ONE_BLK_SIZE;
        initParams.dstGap = 0;
        initParams.initValue = static_cast<srcType>(0);
        Fill(tensor, initParams);
    }
    PipeBarrier<PIPE_MTE2>();
}

__aicore__ inline void InitL1ZeroDedy(const LocalTensor<dedyType>& tensor) { InitL1ZeroValue(tensor, true); }

__aicore__ inline void InitL1ZeroFilter(const LocalTensor<filterType>& tensor) { InitL1ZeroValue(tensor); }

__aicore__ inline void CalcLocalA1Params(uint32_t mStart, uint32_t curM, uint32_t& localHoStart, uint32_t& localHoSize,
                                         uint32_t& localPadUp, uint32_t& localMStart) const
{
    uint32_t hiStart = mStart / tiling_->wi;
    localMStart = mStart - hiStart * tiling_->wi;
    uint32_t hiCount = DivCeil(localMStart + curM, tiling_->wi);
    int64_t receptiveStart = static_cast<int64_t>(hiStart) - tiling_->backpropPadUp;
    int64_t receptiveEnd = receptiveStart + hiCount + static_cast<int64_t>(tiling_->hk - 1) * tiling_->dilationH;
    int64_t hoExpand = static_cast<int64_t>(hoExpand_);
    int64_t validStart = receptiveStart < 0 ? 0 : receptiveStart;
    validStart = validStart > hoExpand ? hoExpand : validStart;
    int64_t validEnd = receptiveEnd < 0 ? 0 : receptiveEnd;
    validEnd = validEnd > hoExpand ? hoExpand : validEnd;

    bool hasValidDedy = validEnd > validStart;
    localHoStart = hasValidDedy ? static_cast<uint32_t>(validStart) : static_cast<uint32_t>(hoExpand);
    localHoSize = hasValidDedy ? static_cast<uint32_t>(validEnd - validStart) : 1;
    localPadUp = receptiveStart < 0 ? static_cast<uint32_t>(-receptiveStart) : 0;
}

__aicore__ inline void LoadDedyL1(const LocalTensor<dedyType>& a1, uint32_t batchIdx, uint32_t localHoStart,
                                  uint32_t localHoSize)
{
    InitL1ZeroDedy(a1);
    uint32_t srcHoStart = DivCeil(localHoStart, tiling_->strideH);
    uint32_t srcHoEnd = DivCeil(localHoStart + localHoSize, tiling_->strideH);
    srcHoStart = srcHoStart > tiling_->ho ? tiling_->ho : srcHoStart;
    srcHoEnd = srcHoEnd > tiling_->ho ? tiling_->ho : srcHoEnd;
    if (srcHoStart >= srcHoEnd) {
        return;
    }

    Dn2NzParams params;
    params.dnNum = srcHoEnd - srcHoStart;
    params.nValue = tiling_->wo;
    params.dValue = tiling_->cout;
    params.srcDnMatrixStride = tiling_->wo;
    params.srcDValue = static_cast<uint32_t>(doHoWo_);
    params.dstNzC0Stride = localHoSize * static_cast<uint32_t>(woExpand_);
    params.dstNzNStride = tiling_->strideW;
    params.dstNzMatrixStride = static_cast<uint32_t>(tiling_->strideH * woExpand_) << tiling_->c0BitsA;
    uint64_t srcOffset = static_cast<uint64_t>(batchIdx) * tiling_->cout * doHoWo_ +
                         static_cast<uint64_t>(srcHoStart) * tiling_->wo;
    uint32_t dstHoStart = srcHoStart * tiling_->strideH - localHoStart;
    uint64_t dstOffset = (static_cast<uint64_t>(dstHoStart) * woExpand_) << tiling_->c0BitsA;
    DataCopy(a1[dstOffset], dedyGm_[srcOffset], params);
}

__aicore__ inline void LoadWeightL1(const LocalTensor<filterType>& b1, uint32_t nStart, uint32_t curN,
                                    uint32_t curNAlign)
{
    InitL1ZeroFilter(b1);
    uint64_t srcOffset = static_cast<uint64_t>(nStart);
    if constexpr (filterFormat == FORMAT_FRACTAL_Z) {
        DataCopyPadExtParams<filterType> padParams;
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = static_cast<uint64_t>(Convolution3DBackpropFunc::AlignUp16(tiling_->cin)) *
                                  AlignUp(tiling_->cout, tiling_->c0) * hkWk_ * sizeof(filterType);
        dataCopyParams.srcStride = 0;
        DataCopyPad<filterType>(b1, filterGm_[srcOffset], dataCopyParams, padParams);
    } else {
        Dn2NzParams params;
        params.dnNum = static_cast<uint32_t>(hkWk_);
        params.dValue = tiling_->cout;
        params.nValue = curN;
        params.srcDnMatrixStride = tiling_->cin;
        params.srcDValue = tiling_->cin * static_cast<uint32_t>(hkWk_);
        params.dstNzMatrixStride = curNAlign << tiling_->c0BitsB;
        params.dstNzC0Stride = static_cast<uint32_t>(hkWk_) * curNAlign;
        params.dstNzNStride = 1;
        DataCopy(b1, filterGm_[srcOffset], params);
    }
}

template <typename channelWiseType>
__aicore__ inline void LoadChannelWiseL1(const LocalTensor<channelWiseType>& dst,
                                         const GlobalTensor<channelWiseType>& src, uint32_t loadNum)
{
    uint16_t blockBytes = loadNum * sizeof(channelWiseType);
    if (blockBytes == 0 || ((blockBytes - 1) / ONE_BLK_SIZE) % 2 == 0) {
        InitL1ZeroValue(dst);
    }
    DataCopyParams dataCopyParams(1, blockBytes, 0, 0);
    uint8_t rightPadding = static_cast<uint8_t>(
        DivCeil(blockBytes, ONE_BLK_SIZE) * ONE_BLK_SIZE / sizeof(channelWiseType) - loadNum);
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    DataCopyPad<channelWiseType>(dst, src, dataCopyParams, padParams);
}

__aicore__ inline void LoadBiasScaleL1(uint32_t nStart, uint32_t curN)
{
    if (hasBias_) {
        LocalTensor<biasType> biasL1(TPosition::A1, GetBiasL1OffBytes(), GetBiasL1ElemCount());
        LoadChannelWiseL1<biasType>(biasL1, biasGm_[nStart], curN);
    }
    if constexpr (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            LocalTensor<scale0Type> scaleL1(TPosition::A1, GetScale0L1OffBytes(), tiling_->singleCoreCin);
            LoadChannelWiseL1<scale0Type>(scaleL1, scale0Gm_[nStart], curN);
        }
    }
    if constexpr (GetScaleFormat(scale1Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (hasSecondOutput_ &&
            tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            LocalTensor<scale1Type> scale1L1(TPosition::A1, GetScale1L1OffBytes(), tiling_->singleCoreCin);
            LoadChannelWiseL1<scale1Type>(scale1L1, scale1Gm_[nStart], curN);
        }
    }
}

__aicore__ inline void LoadBiasToBT(uint32_t curN)
{
    LocalTensor<L0cT> biasBT(TPosition::C2, 0, Convolution3DBackpropFunc::AlignUp16(curN));
    LocalTensor<biasType> biasL1(TPosition::A1, GetBiasL1OffBytes(), GetBiasL1ElemCount());
    uint32_t blockCnt = DivCeil(curN * sizeof(biasType), SMALL_KERNEL_BIAS_L1_ALIGN_BYTES) << 1;
    DataCopyParams dataCopyParams(1, static_cast<uint16_t>(blockCnt), 0, 0);
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    if constexpr (std::is_same<dedyType, half>::value && std::is_same<filterType, half>::value) {
        dataCopyParams.fixShiftVal = SHIFT_VALUE_LEN - static_cast<uint8_t>(tiling_->fixedShiftVal);
    }
#endif
    DataCopy(biasBT, biasL1, dataCopyParams);
}

__aicore__ inline void LoadAL0(LocalTensor<dedyType>& a0, const LocalTensor<dedyType>& a1, uint32_t localHoSize,
                               uint32_t localPadUp, uint32_t localMStart, uint32_t curMAlign, uint32_t kOff,
                               uint32_t curK)
{
    LoadData3DParamsV2<dedyType> params;
    params.filterW = tiling_->wk;
    params.filterH = tiling_->hk;
    params.filterSizeW = (tiling_->wk >> FILTER_SIZE_BIT_SHIFT) & FILTER_SIZE_BIT_MASK;
    params.filterSizeH = (tiling_->hk >> FILTER_SIZE_BIT_SHIFT) & FILTER_SIZE_BIT_MASK;
    params.dilationFilterW = tiling_->dilationW;
    params.dilationFilterH = tiling_->dilationH;
    params.channelSize = coutAlign_;
    params.l1H = localHoSize;
    params.l1W = static_cast<uint32_t>(woExpand_);
    params.padList[PAD_LIST_IDX_LEFT] = static_cast<uint8_t>(tiling_->backpropPadLeft);
    params.padList[PAD_LIST_IDX_RIGHT] = static_cast<uint8_t>(tiling_->backpropPadRight);
    params.padList[PAD_LIST_IDX_UP] = static_cast<uint8_t>(localPadUp);
    params.padList[PAD_LIST_IDX_DOWN] = LOAD3D_PAD_DOWN_VALUE;
    params.strideW = 1;
    params.strideH = 1;
    params.mStartPt = localMStart;
    params.mExtension = curMAlign;
    params.kStartPt = kOff;
    params.kExtension = curK;
    params.enTranspose = 0;
    params.fMatrixCtrl = 0;
    // 非对称量化计算场景需要在load3d时传入offsetX
    if constexpr (std::is_same<L0cT, int32_t>::value) {
        params.padValue = tiling_->offsetX;
    }
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeatParam = {0, 1, 0, static_cast<uint16_t>(DivCeil(curMAlign, 16))};
    SetLoadDataRepeatWithStride(repeatParam);
    LoadDataWithStride(a0, a1, params);
#else
    LoadDataRepeatParam repeatParam = {0, 1, 0, static_cast<uint16_t>(DivCeil(curMAlign, 16))};
    SetLoadDataRepeat(repeatParam);
    LoadData(a0, a1, params);
#endif
}

__aicore__ inline void LoadBL0(LocalTensor<filterType>& b0, const LocalTensor<filterType>& b1, uint32_t kOff,
                               uint32_t curK, uint32_t curNAlign)
{
    uint32_t blockBaseN = DivCeil(curNAlign, 16);
    uint32_t blockSize = tiling_->c0 << 4;
    LoadData2DParamsV2 params;
    params.ifTranspose = 0;
    params.mStep = blockBaseN;
    params.dstStride = blockBaseN;
    if constexpr (filterFormat == FORMAT_FRACTAL_Z) {
        params.srcStride = static_cast<int32_t>(blockBaseN);
    } else {
        params.srcStride = -static_cast<int32_t>(blockBaseN);
    }
    uint32_t hkWk = static_cast<uint32_t>(hkWk_);
    uint32_t kStartPos = kOff >> tiling_->c0BitsB;
    uint32_t kEndPos = kStartPos + DivCeil(curK, tiling_->c0);
    uint32_t hwKStartIdx = kStartPos / hkWk;
    uint32_t hwKEndIdx = DivCeil(kEndPos, hkWk);
    uint32_t dstB2Offset = 0;
    for (uint32_t i = 0; i < hwKEndIdx - hwKStartIdx; ++i) {
        uint32_t curHWkStart = (hwKStartIdx + i) * hkWk;
        uint32_t curHWkEnd = curHWkStart + hkWk;
        uint32_t kStepStart = curHWkStart < kStartPos ? kStartPos : curHWkStart;
        uint32_t kStepEnd = curHWkEnd > kEndPos ? kEndPos : curHWkEnd;
        params.kStep = kStepEnd - kStepStart;
        params.kStartPosition = curHWkStart;
        if constexpr (filterFormat == FORMAT_FRACTAL_Z) {
            params.mStartPosition = (kStepStart - curHWkStart) * blockBaseN;
        } else {
            params.mStartPosition = (curHWkEnd - 1 - kStepStart) * blockBaseN;
        }
        LoadData(b0[dstB2Offset], b1, params);
        dstB2Offset += (kStepEnd - kStepStart) * blockBaseN * blockSize;
    }
}

template <bool output1>
__aicore__ inline auto GetFixpipeDeqScalar()
{
    if constexpr (output1) {
        return scale1Gm_.GetValue(0);
    } else {
        return scale0Gm_.GetValue(0);
    }
}

template <bool output1, QuantMode_t vectorMode, QuantMode_t scalarMode>
__aicore__ inline void SetFixpipeScaleQuant(FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR>& params)
{
    constexpr auto scaleFormat = output1 ? GetScaleFormat(scale1Format) : GetScaleFormat(scale0Format);
    params.quantPre = scalarMode;
    params.deqScalar = CONV3D_DX_SMALL_DQ_SCALAR_QF_ONE;
    if constexpr (scaleFormat != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        const uint8_t quantMode = output1 ? tiling_->quantMode1 : tiling_->quantMode0;
        if (quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            params.quantPre = vectorMode;
        } else if (quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::SCALAR_QUANT)) {
            params.deqScalar = GetFixpipeDeqScalar<output1>();
        }
    }
}

template <typename outputType, bool output1 = false>
__aicore__ inline void SetFixpipeQuant(FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR>& params)
{
    if constexpr (std::is_same<outputType, bfloat16_t>::value) {
        params.quantPre = QuantMode_t::F322BF16;
    } else if constexpr (std::is_same<outputType, half>::value && std::is_same<L0cT, int32_t>::value) {
        SetFixpipeScaleQuant<output1, QuantMode_t::VDEQF16, QuantMode_t::DEQF16>(params);
    } else if constexpr (std::is_same<outputType, half>::value) {
        params.quantPre = QuantMode_t::F322F16;
    } else if constexpr (std::is_same<outputType, int8_t>::value) {
        SetFixpipeScaleQuant<output1, QuantMode_t::VREQ8, QuantMode_t::REQ8>(params);
    } else {
        params.quantPre = QuantMode_t::NoQuant;
    }
}

__aicore__ inline void CopyOut(const LocalTensor<L0cT>& c0, uint32_t batchIdx, uint32_t mStart, uint32_t nStart,
                               uint32_t curM, uint32_t curMAlign, uint32_t curN)
{
    FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> params;
    params.params.dnNum = 1;
    params.params.srcNzMatrixStride = 0;
    params.params.dstDnMatrixStride = 0;
    params.params.srcNzC0Stride = 1;
    params.mSize = curM;
    params.nSize = curN;
    params.srcStride = curMAlign;
    params.dstStride = diHiWi_;
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    params.preReluMode = static_cast<ReluMode>(tiling_->enRelu0);
    if constexpr (std::is_same<dedyType, half>::value && std::is_same<filterType, half>::value) {
        params.fixShiftVal = SHIFT_VALUE_LEN - static_cast<uint8_t>(tiling_->fixedShiftVal);
    }
#endif
    SetFixpipeQuant<yType>(params);
    uint64_t batchOffset = static_cast<uint64_t>(batchIdx) * tiling_->cin * diHiWi_;
    uint64_t dstOffset = batchOffset + static_cast<uint64_t>(nStart) * diHiWi_ + mStart;
    if (GetScaleFormat(scale0Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        tiling_->quantMode0 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        LocalTensor<scale0Type> scaleL1(TPosition::A1, GetScale0L1OffBytes(), tiling_->singleCoreCin);
        Fixpipe<yType, L0cT, CFG_COLUMN_MAJOR>(yGm_[dstOffset], c0, scaleL1, params);
    } else {
        Fixpipe<yType, L0cT, CFG_COLUMN_MAJOR>(yGm_[dstOffset], c0, params);
    }
    if (hasSecondOutput_) {
        FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> params1 = params;
        SetFixpipeQuant<y1Type, true>(params1);
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
        // 第二路输出（y1）独立 relu：覆盖为 enRelu1（params 继承的是 y0 的 enRelu）。
        params1.preReluMode = static_cast<ReluMode>(tiling_->enRelu1);
#endif
        if (GetScaleFormat(scale1Format) != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
            tiling_->quantMode1 == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            LocalTensor<scale1Type> scale1L1(TPosition::A1, GetScale1L1OffBytes(), tiling_->singleCoreCin);
            Fixpipe<y1Type, L0cT, CFG_COLUMN_MAJOR>(y1Gm_[dstOffset], c0, scale1L1, params1);
        } else {
            Fixpipe<y1Type, L0cT, CFG_COLUMN_MAJOR>(y1Gm_[dstOffset], c0, params1);
        }
    }
}

#endif // CONV3D_BP_SMALL_KERNEL_FUNC_ADVANCE_H
