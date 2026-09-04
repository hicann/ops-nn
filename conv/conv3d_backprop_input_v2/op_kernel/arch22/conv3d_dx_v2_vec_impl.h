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
 * \file conv3d_dx_v2_vec_impl.h
 * \brief Conv3DBackpropInput vector 兜底 kernel：逐行累加，超 L1 场景下替代 cube 路径。
 *        行计算（ComputeRow*）定义见 conv3d_dx_v2_vec_impl_row.h。
 */
#ifndef CONV3D_DX_V2_VEC_IMPL_H
#define CONV3D_DX_V2_VEC_IMPL_H

#include "kernel_operator.h"
#include "conv3d_backprop_input_v2_tiling_data.h"
#include <type_traits>

using namespace AscendC;

template <typename T>
__aicore__ inline float VecToFloat(T value)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T VecFromFloat(float value)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        return AscendC::Cast(value);
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
class KernelConv3dBackpropInputVecImpl {
public:
    __aicore__ inline KernelConv3dBackpropInputVecImpl() {}

    __aicore__ inline void Init(GM_ADDR gradOutput, GM_ADDR weight, GM_ADDR gradInput, GM_ADDR workspace,
                                const Conv3DBackpropInputV2TilingData* tilingData)
    {
        const auto& dx = tilingData->conv3DDxTiling;
        batch = dx.batch;
        gradOutC = dx.cout;
        gradOutD = dx.dout;
        gradOutH = dx.ho;
        gradOutW = dx.wo;
        gradInC = dx.cin;
        gradInD = dx.di;
        gradInH = dx.hi;
        gradInW = dx.wi;
        kernelD = dx.dk;
        kernelH = dx.hk;
        kernelW = dx.wk;
        // groups>1 时 weight 为分组布局，需按组定位
        groups = dx.group == 0 ? 1 : dx.group;
        cinPerGroup = gradInC / groups;
        coutPerGroup = gradOutC / groups;

        strideD = dx.strideD;
        strideH = dx.strideH;
        strideW = dx.strideW;
        dilationD = dx.dilationD;
        dilationH = dx.dilationH;
        dilationW = dx.dilationW;
        padFront = dx.padFront;
        padHDx = dx.padHDx;
        padUDx = dx.padUDx;
        padLDx = dx.padLDx;
        dilatedHk = dx.dilatedHk;
        dilatedWk = dx.dilatedWk;
        alignedDilatedW = dx.alignedDilatedW;
        dataPerBlock = dx.dataPerBlock == 0 ? static_cast<uint32_t>(32 / sizeof(T)) : dx.dataPerBlock;
        alignedWi = ((gradInW + dataPerBlock - 1) / dataPerBlock) * dataPerBlock;
        // strideW>1 时相位分解的类缓冲长度 ≈ gradInW/strideW
        const uint32_t sW = strideW > 1 ? static_cast<uint32_t>(strideW) : 1;
        classAlignedWi = (((gradInW + sW - 1) / sW + dataPerBlock - 1) / dataPerBlock) * dataPerBlock;
        vecScalarOnly = dx.vecScalarOnly;
        useScalarAcc = dx.useScalarAcc;

        blockDim = static_cast<uint32_t>(tilingData->params.coreNum);

        uint64_t totalGradOut = static_cast<uint64_t>(batch) * gradOutC * gradOutD * gradOutH * gradOutW;
        uint64_t totalWeight = static_cast<uint64_t>(gradOutC) * cinPerGroup * kernelD * kernelH * kernelW;
        uint64_t totalGradIn = static_cast<uint64_t>(batch) * gradInC * gradInD * gradInH * gradInW;

        gradOutputGm.SetGlobalBuffer((__gm__ T*)gradOutput, totalGradOut);
        weightGm.SetGlobalBuffer((__gm__ T*)weight, totalWeight);
        gradInputGm.SetGlobalBuffer((__gm__ T*)gradInput, totalGradIn);

        pipe.InitBuffer(tmpBufWeightDilated, dilatedHk * alignedDilatedW * sizeof(T));
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= blockDim) {
            return;
        }
        // 行分区：各行独立归约 (co,dk)，无竞争、无需 SyncAll
        uint64_t totalRows = static_cast<uint64_t>(batch) * gradInC * gradInD * gradInH;
        uint64_t start, end;
        Partition(blockIdx, blockDim, totalRows, start, end);
        for (uint64_t r = start; r < end; r++) {
            ProcessRow(r);
        }
    }

private:
    __aicore__ inline void Partition(uint32_t idx, uint32_t cores, uint64_t total, uint64_t& start, uint64_t& end) const
    {
        if (cores == 0) {
            cores = 1;
        }
        uint64_t per = (total + cores - 1) / cores;
        start = static_cast<uint64_t>(idx) * per;
        end = (start + per < total) ? (start + per) : total;
        if (start > total)
            start = total;
    }

    // 仅 BF16：strideW=1 时用 DataCopyPad 零填充构造 W 维错位
    __aicore__ inline bool CanUseBf16VecRow() const
    {
        if (strideW != 1) {
            return false;
        }
        if (padLDx < 0 || padLDx > 255) {
            return false;
        }
        if (dilatedWk > 255 || gradInW == 0 || gradOutW == 0 || dataPerBlock == 0) {
            return false;
        }
        // DataCopyPad 左右填充按字节计不能超过 32B（bf16/fp16 元素上限不够）
        if (static_cast<int64_t>(padLDx) * sizeof(T) > 32) {
            return false;
        }
        const int64_t rightPadMax = static_cast<int64_t>(gradInW) - static_cast<int64_t>(gradOutW) +
                                    (static_cast<int64_t>(kernelW) - 1) * dilationW - padLDx;
        if (rightPadMax > 0 && rightPadMax * sizeof(T) > 32) {
            return false;
        }
        return alignedWi <= 0xFFFFU;
    }

    // 仅 BF16：strideW>1 按相位分解向量化，每相位 tap 仍是一条连续 DataCopyPad
    __aicore__ inline bool CanUseBf16VecStridedRow() const
    {
        if (strideW <= 1) {
            return false;
        }
        if (padLDx < 0 || padLDx > 255) {
            return false;
        }
        if (dilatedWk > 255 || gradInW == 0 || gradOutW == 0 || dataPerBlock == 0) {
            return false;
        }
        if (classAlignedWi == 0 || classAlignedWi > 0xFFFFU) {
            return false;
        }
        const int32_t sW = strideW;
        const int32_t gradOutWI = static_cast<int32_t>(gradOutW);
        const int32_t gradInWI = static_cast<int32_t>(gradInW);
        for (int32_t p = 0; p < sW; p++) {
            const int32_t kMax = (gradInWI > p) ? (gradInWI - p + sW - 1) / sW : 0;
            if (kMax == 0) {
                continue;
            }
            const uint32_t kAligned = ((static_cast<uint32_t>(kMax) + dataPerBlock - 1) / dataPerBlock) * dataPerBlock;
            const int32_t dw0 = ((padLDx - p) % sW + sW) % sW;
            const int32_t c0 = (dw0 + p - padLDx) / sW;
            for (int32_t dw = dw0; dw < static_cast<int32_t>(dilatedWk); dw += sW) {
                const int32_t delta = c0 + (dw - dw0) / sW;
                const int32_t first = delta < 0 ? -delta : 0;
                int32_t endExclusive = gradOutWI - delta;
                if (endExclusive > kMax) {
                    endExclusive = kMax;
                }
                if (endExclusive <= first) {
                    continue;
                }
                if (first > 255 || static_cast<int32_t>(kAligned) - endExclusive > 255) {
                    return false;
                }
                // DataCopyPad 左填充按字节计不能超 32B，否则降级标量路径
                if (first * static_cast<int32_t>(sizeof(T)) > 32) {
                    return false;
                }
            }
        }
        return true;
    }

    __aicore__ inline void AccumulateBf16Tap(uint64_t goRowBase, uint32_t dw, float wv, LocalTensor<float>& rowAcc,
                                             LocalTensor<float>& prod, LocalTensor<T>& gradRow)
    {
        const int32_t delta = static_cast<int32_t>(dw) - padLDx;
        const int32_t first = delta < 0 ? -delta : 0;
        const int64_t endExclusive64 = static_cast<int64_t>(gradOutW) - delta;
        if (endExclusive64 <= first) {
            return;
        }
        uint32_t endExclusive = static_cast<uint32_t>(endExclusive64);
        if (endExclusive > gradInW) {
            endExclusive = gradInW;
        }
        const uint32_t leftPad = static_cast<uint32_t>(first);
        if (leftPad >= endExclusive) {
            return;
        }
        const uint32_t validLen = endExclusive - leftPad;
        if (validLen == 0) {
            return;
        }
        const uint32_t rightPad = gradInW - endExclusive;
        const uint32_t srcStart = delta > 0 ? static_cast<uint32_t>(delta) : 0;
        if (srcStart >= gradOutW) {
            return;
        }

        Duplicate(gradRow, static_cast<T>(0), alignedWi);
        SetFlag<HardEvent::V_MTE2>(evtVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(evtVToMte2_);
        DataCopyExtParams tapParams(1, validLen * sizeof(T), 0, 0, 0);
        DataCopyPadExtParams<T> tapPad(true, static_cast<uint8_t>(leftPad), static_cast<uint8_t>(rightPad),
                                       static_cast<T>(0));
        DataCopyPad<T>(gradRow, gradOutputGm[goRowBase + srcStart], tapParams, tapPad);
        SetFlag<HardEvent::MTE2_V>(evtMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToV_);

        if constexpr (std::is_same<T, float>::value) {
            // 同型 Cast 不被 AscendC 支持，fp32 用 Adds+0 等价拷贝
            Adds(prod, gradRow, 0.0f, alignedWi);
        } else {
            Cast(prod, gradRow, RoundMode::CAST_NONE, alignedWi);
        }
        SetFlag<HardEvent::V_S>(evtVToS_); // V->S 屏障：Cast 完成后 Muls 才能读 prod
        WaitFlag<HardEvent::V_S>(evtVToS_);
        Muls(prod, prod, wv, alignedWi);
        SetFlag<HardEvent::V_S>(evtVToS_); // V->S 屏障：Muls 完成后 Add 才能读 prod
        WaitFlag<HardEvent::V_S>(evtVToS_);
        Add(rowAcc, rowAcc, prod, alignedWi);
    }

    // FP32 下 Cast 同型不被支持，用 Adds+0 等价拷贝
    __aicore__ inline void CastRowAccToOutH(const LocalTensor<float>& rowAcc, const LocalTensor<T>& outH, uint32_t len)
    {
        if constexpr (std::is_same<T, float>::value) {
            Adds(outH, rowAcc, 0.0f, len);
        } else {
            Cast(outH, rowAcc, RoundMode::CAST_RINT, len);
        }
    }

    // 相位分解 tap 累加：与 AccumulateBf16Tap 同构，作用于紧凑类缓冲
    __aicore__ inline void AccumulateBf16TapClass(uint64_t goRowBase, int32_t delta, float wv,
                                                  LocalTensor<float>& rowAcc, LocalTensor<float>& prod,
                                                  LocalTensor<T>& gradRow, uint32_t outLen, uint32_t alignedLen)
    {
        const int32_t first = delta < 0 ? -delta : 0;
        const int64_t endExclusive64 = static_cast<int64_t>(gradOutW) - delta;
        if (endExclusive64 <= first) {
            return;
        }
        uint32_t endExclusive = static_cast<uint32_t>(endExclusive64);
        if (endExclusive > outLen) {
            endExclusive = outLen;
        }
        const uint32_t leftPad = static_cast<uint32_t>(first);
        if (leftPad >= endExclusive) {
            return;
        }
        const uint32_t validLen = endExclusive - leftPad;
        if (validLen == 0) {
            return;
        }
        const uint32_t rightPad = alignedLen - endExclusive;
        // DataCopyPad 每侧填充按字节计不能超 32B：超限 clamp 到 32B（置 0 会污染有效位置）
        const uint32_t maxPad = 32U / static_cast<uint32_t>(sizeof(T));
        const uint32_t copyRightPad = (rightPad > maxPad) ? maxPad : rightPad;
        const uint32_t srcStart = delta > 0 ? static_cast<uint32_t>(delta) : 0;
        if (srcStart >= gradOutW) {
            return;
        }

        Duplicate(gradRow, static_cast<T>(0), alignedLen);
        SetFlag<HardEvent::V_MTE2>(evtVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(evtVToMte2_);
        DataCopyExtParams tapParams(1, validLen * sizeof(T), 0, 0, 0);
        DataCopyPadExtParams<T> tapPad(true, static_cast<uint8_t>(leftPad), static_cast<uint8_t>(copyRightPad),
                                       static_cast<T>(0));
        DataCopyPad<T>(gradRow, gradOutputGm[goRowBase + srcStart], tapParams, tapPad);
        SetFlag<HardEvent::MTE2_V>(evtMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(evtMte2ToV_);

        Cast(prod, gradRow, RoundMode::CAST_NONE, alignedLen);
        SetFlag<HardEvent::V_S>(evtVToS_);
        WaitFlag<HardEvent::V_S>(evtVToS_);
        Muls(prod, prod, wv, alignedLen);
        SetFlag<HardEvent::V_S>(evtVToS_);
        WaitFlag<HardEvent::V_S>(evtVToS_);
        Add(rowAcc, rowAcc, prod, alignedLen);
    }

    // 行计算相关：声明在类内，定义见 conv3d_dx_v2_vec_impl_row.h
    __aicore__ inline void InitBuffers();
    __aicore__ inline void ProcessRow(uint64_t r);
    __aicore__ inline void DecomposeRow(uint64_t r, uint32_t& n, uint32_t& ci, uint32_t& diRow, uint32_t& hi) const;
    __aicore__ inline uint64_t GetRowBase(uint32_t n, uint32_t ci, uint32_t diRow, uint32_t hi) const;
    __aicore__ inline void GetGroupRange(uint32_t ci, uint32_t& ciLocal, uint32_t& coStart, uint32_t& coEnd) const;
    __aicore__ inline void BuildWeightDilated(uint32_t co, uint32_t ciLocal, uint32_t dk,
                                              LocalTensor<T>& weightDilated);
    __aicore__ inline uint64_t GetGoPlaneBase(uint32_t n, uint32_t co, uint32_t doIdx) const;
    __aicore__ inline bool GetHoIndex(uint32_t dh, uint32_t hi, uint32_t& ho) const;
    __aicore__ inline void AccumulateBf16VecRowTaps(uint64_t goPlaneBase, uint32_t hi, LocalTensor<float>& rowAcc,
                                                    LocalTensor<float>& prod, LocalTensor<T>& gradRow,
                                                    LocalTensor<T>& weightDilated);
    __aicore__ inline void AccumulateBf16VecStridedTaps(uint32_t ciLocal, uint32_t coStart, uint32_t coEnd, uint32_t n,
                                                        uint32_t diRow, uint32_t hi, uint32_t kMax, uint32_t kAligned,
                                                        int32_t dw0, int32_t c0, uint32_t sW,
                                                        LocalTensor<float>& rowAccP, LocalTensor<float>& prodP,
                                                        LocalTensor<T>& gradRowP);
    __aicore__ inline void AccumulateBf16VecStridedTapRow(uint64_t goPlaneBase, uint32_t hi, uint32_t kMax,
                                                          uint32_t kAligned, int32_t dw0, int32_t c0, uint32_t sW,
                                                          LocalTensor<float>& rowAccP, LocalTensor<float>& prodP,
                                                          LocalTensor<T>& gradRowP, LocalTensor<T>& weightDilated);
    template <bool UseUbAcc>
    __aicore__ inline void AccumulateScalarRowTaps(uint64_t goPlaneBase, uint32_t hi, uint64_t rowBase,
                                                   LocalTensor<float>& rowAcc, LocalTensor<T>& weightDilated);
    __aicore__ inline void ComputeRowBf16Vec(uint64_t r);
    __aicore__ inline void ComputeRowBf16VecStrided(uint64_t r);
    __aicore__ inline void ComputeRowScalarAcc(uint64_t r);
    __aicore__ inline void ComputeRow(uint64_t r);

    uint32_t batch, gradOutC, gradOutD, gradOutH, gradOutW;
    uint32_t gradInC, gradInD, gradInH, gradInW;
    uint32_t groups, cinPerGroup, coutPerGroup;
    uint32_t kernelD, kernelH, kernelW;
    int32_t strideD, strideH, strideW;
    uint32_t dilationD, dilationH, dilationW;
    int32_t padFront, padHDx, padUDx, padLDx;
    uint32_t dilatedHk, dilatedWk, alignedDilatedW;
    uint32_t dataPerBlock, alignedWi, classAlignedWi;
    uint32_t vecScalarOnly = 0;
    uint32_t useScalarAcc = 0;
    uint32_t blockDim;

    TPipe pipe;
    TBuf<QuePosition::VECCALC> tmpBufWeightDilated;
    TBuf<QuePosition::VECCALC> tmpBufRowAcc;
    TBuf<QuePosition::VECCALC> tmpBufProd;
    TBuf<QuePosition::VECCALC> tmpBufGradRow;
    TBuf<QuePosition::VECCALC> tmpBufOutH;
    TBuf<QuePosition::VECCALC> tmpBufRowAccP;
    TBuf<QuePosition::VECCALC> tmpBufProdP;
    TBuf<QuePosition::VECCALC> tmpBufGradRowP;
    TBuf<QuePosition::VECCALC> tmpBufOutHP;

    event_t evtVToMte2_;
    event_t evtMte2ToV_;
    event_t evtVToMte3_;
    event_t evtMte3ToV_;
    event_t evtVToS_;
    event_t evtSToV_;

    GlobalTensor<T> gradOutputGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> gradInputGm;
};

#include "conv3d_dx_v2_vec_impl_row.h"

#endif // CONV3D_DX_V2_VEC_IMPL_H
