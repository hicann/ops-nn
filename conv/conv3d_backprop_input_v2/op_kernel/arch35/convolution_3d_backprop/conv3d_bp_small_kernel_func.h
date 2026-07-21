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
        InitConstValue(tensor.template ReinterpretCast<uint16_t>(),
                       {1, static_cast<uint16_t>(len / ONE_BLK_SIZE), 0, padValue});
    } else {
        AscendC::InitConstValueParams<srcType> initParams;
        initParams.repeatTimes = 1;
        initParams.blockNum = len / ONE_BLK_SIZE;
        initParams.dstGap = 0;
        initParams.initValue = static_cast<srcType>(0);
        InitConstValue(tensor, initParams);
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
    Dn2NzParams params;
    params.dnNum = static_cast<uint32_t>(hkWk_);
    params.dValue = tiling_->cout;
    params.nValue = curN;
    params.srcDnMatrixStride = tiling_->cin;
    params.srcDValue = tiling_->cin * static_cast<uint32_t>(hkWk_);
    params.dstNzMatrixStride = curNAlign << tiling_->c0BitsB;
    params.dstNzC0Stride = static_cast<uint32_t>(hkWk_) * curNAlign;
    params.dstNzNStride = 1;
    uint64_t srcOffset = static_cast<uint64_t>(nStart);
    DataCopy(b1, filterGm_[srcOffset], params);
}

template <typename channelWiseType>
__aicore__ inline void LoadChannelWiseL1(const LocalTensor<channelWiseType>& dst,
                                         const GlobalTensor<channelWiseType>& src, uint32_t loadNum)
{
    InitL1ZeroValue(dst);
    DataCopyParams dataCopyParams(1, loadNum * sizeof(channelWiseType), 0, 0);
    uint8_t rightPadding = static_cast<uint8_t>(
        ((loadNum * sizeof(channelWiseType) + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE) /
            sizeof(channelWiseType) -
        loadNum);
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    DataCopyPad<channelWiseType>(dst, src, dataCopyParams, padParams);
}

__aicore__ inline void LoadBiasScaleL1(uint32_t nStart, uint32_t curN)
{
    if (hasBias_) {
        LocalTensor<biasType> biasL1(TPosition::A1, GetBiasL1OffBytes(), tiling_->singleCoreCin);
        LoadChannelWiseL1<biasType>(biasL1, biasGm_[nStart], curN);
    }
    if constexpr (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
        if (tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
            LocalTensor<scaleType> scaleL1(TPosition::A1, GetScaleL1OffBytes(), tiling_->singleCoreCin);
            LoadChannelWiseL1<scaleType>(scaleL1, scaleGm_[nStart], curN);
        }
    }
}

__aicore__ inline void LoadBiasToBT(uint32_t curN)
{
    LocalTensor<L0cT> biasBT(TPosition::C2, 0, Convolution3DBackpropFunc::AlignUp16(curN));
    LocalTensor<biasType> biasL1(TPosition::A1, GetBiasL1OffBytes(), tiling_->singleCoreCin);
    uint32_t blockCnt = DivCeil(curN * sizeof(biasType), 64) << 1;
    DataCopyParams dataCopyParams(1, static_cast<uint16_t>(blockCnt), 0, 0);
#if (__NPU_ARCH__ == 5102)
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
    params.filterSizeW = (tiling_->wk >> 8) & 255;
    params.filterSizeH = (tiling_->hk >> 8) & 255;
    params.dilationFilterW = tiling_->dilationW;
    params.dilationFilterH = tiling_->dilationH;
    params.channelSize = coutAlign_;
    params.l1H = localHoSize;
    params.l1W = static_cast<uint32_t>(woExpand_);
    params.padList[0] = static_cast<uint8_t>(tiling_->backpropPadLeft);
    params.padList[1] = static_cast<uint8_t>(tiling_->backpropPadRight);
    params.padList[2] = static_cast<uint8_t>(localPadUp);
    params.padList[3] = 255;
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
    params.srcStride = -static_cast<int32_t>(blockBaseN);
    params.mStep = blockBaseN;
    params.dstStride = blockBaseN;
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
        params.mStartPosition = (curHWkEnd - 1 - kStepStart) * blockBaseN;
        LoadData(b0[dstB2Offset], b1, params);
        dstB2Offset += (kStepEnd - kStepStart) * blockBaseN * blockSize;
    }
}

__aicore__ inline void SetFixpipeQuant(FixpipeParamsC310<CO2Layout::COLUMN_MAJOR>& params)
{
    if constexpr (std::is_same<yType, bfloat16_t>::value) {
        params.quantPre = QuantMode_t::F322BF16;
    } else if constexpr (std::is_same<yType, half>::value && std::is_same<L0cT, int32_t>::value) {
        if constexpr (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                params.quantPre = QuantMode_t::VDEQF16;
            } else {
                params.quantPre = QuantMode_t::DEQF16;
                params.deqScalar = scaleGm_.GetValue(0);
            }
        } else {
            params.quantPre = QuantMode_t::DEQF16;
            params.deqScalar = CONV3D_DX_SMALL_DQ_SCALAR_QF_ONE;
        }
    } else if constexpr (std::is_same<yType, half>::value) {
        params.quantPre = QuantMode_t::F322F16;
    } else if constexpr (std::is_same<yType, int8_t>::value) {
        if constexpr (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT) {
            if (tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
                params.quantPre = QuantMode_t::VREQ8;
            } else {
                params.quantPre = QuantMode_t::REQ8;
                params.deqScalar = scaleGm_.GetValue(0);
            }
        } else {
            params.quantPre = QuantMode_t::REQ8;
            params.deqScalar = CONV3D_DX_SMALL_DQ_SCALAR_QF_ONE;
        }
    } else {
        params.quantPre = QuantMode_t::NoQuant;
    }
}

__aicore__ inline void CopyOut(const LocalTensor<L0cT>& c0, uint32_t batchIdx, uint32_t mStart, uint32_t nStart,
                               uint32_t curM, uint32_t curMAlign, uint32_t curN)
{
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> params;
    params.params.dnNum = 1;
    params.params.srcNzMatrixStride = 0;
    params.params.dstDnMatrixStride = 0;
    params.params.srcNzC0Stride = 1;
    params.mSize = curM;
    params.nSize = curN;
    params.srcStride = curMAlign;
    params.dstStride = diHiWi_;
#if (__NPU_ARCH__ == 5102)
    params.preReluMode = static_cast<ReluMode>(tiling_->enRelu);
    if constexpr (std::is_same<dedyType, half>::value && std::is_same<filterType, half>::value) {
        params.fixShiftVal = SHIFT_VALUE_LEN - static_cast<uint8_t>(tiling_->fixedShiftVal);
    }
#endif
    SetFixpipeQuant(params);
    uint64_t batchOffset = static_cast<uint64_t>(batchIdx) * tiling_->cin * diHiWi_;
    uint64_t dstOffset = batchOffset + static_cast<uint64_t>(nStart) * diHiWi_ + mStart;
    if (GetScaleFormat(scaleFormat) != Convolution3DBackprop::CubeFormat::UNSUPPORT &&
        tiling_->quantMode == static_cast<uint8_t>(Convolution3DBackprop::QuantMode::VECTOR_QUANT)) {
        LocalTensor<scaleType> scaleL1(TPosition::A1, GetScaleL1OffBytes(), tiling_->singleCoreCin);
        Fixpipe<yType, L0cT, CFG_COLUMN_MAJOR>(yGm_[dstOffset], c0, scaleL1, params);
    } else {
        Fixpipe<yType, L0cT, CFG_COLUMN_MAJOR>(yGm_[dstOffset], c0, params);
    }
}

#endif // CONV3D_BP_SMALL_KERNEL_FUNC_ADVANCE_H
