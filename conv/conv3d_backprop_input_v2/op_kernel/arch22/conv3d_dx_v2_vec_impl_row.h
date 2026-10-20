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
 * \file conv3d_dx_v2_vec_impl_row.h
 * \brief Conv3DBackpropInput vector 兜底 kernel 的行计算实现（ComputeRow* 与共享辅助）。
 *        由 conv3d_dx_v2_vec_impl.h 在类定义末尾 include。
 */
#ifndef CONV3D_DX_V2_VEC_IMPL_ROW_H
#define CONV3D_DX_V2_VEC_IMPL_ROW_H

// 缓冲初始化：按 dtype 与 UB 预算分派，与 tiling 侧 vecScalarOnly/useScalarAcc 口径一致
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::InitBuffers()
{
    // BF16/FP16 共用：W 维向量化快路径 + FP32 累加兜底；strideW>1 相位分解仅 BF16 使用
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        if (vecScalarOnly) {
            // UB 预算降级：放得下 rowAcc+outH 时走 ComputeRowScalarAcc，否则纯标量 ComputeRow
            if (useScalarAcc) {
                pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
                pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
                pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
            }
        } else if (CanUseBf16VecStridedRow()) {
            // strideW>1 相位分解向量化（仅 BF16）：主缓冲约 gradInW/strideW 宽
            pipe.InitBuffer(tmpBufRowAccP, classAlignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufProdP, classAlignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufGradRowP, classAlignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufOutHP, classAlignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufRowTmp, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufRowTmp2, alignedWi * sizeof(float));
            if (alignedWi < FP32_KAHAN_MIN_ELEMS) {
                // 短行在 ProcessRow 中先于 strided 判据路由到 ComputeRowScalarAcc，
                // 该路径使用全行 rowAcc/outH，必须在此补齐，否则会访问未 InitBuffer 的 TBuf
                pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
                pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
            }
        } else {
            // strideW==1 快路径或标量兜底：全行缓冲
            pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufProd, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufGradRow, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufRowTmp, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufRowTmp2, alignedWi * sizeof(float));
        }
    } else if constexpr (std::is_same<T, half>::value) {
        // FP16 与 BF16 共用：快路径需全行缓冲，标量兜底只需 rowAcc/outH
        if (vecScalarOnly) {
            // UB 预算降级：放得下 rowAcc+outH 时走 ComputeRowScalarAcc，否则纯标量 ComputeRow
            if (useScalarAcc) {
                pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
                pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
                pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
            }
        } else if (CanUseBf16VecRow()) {
            pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufProd, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufGradRow, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufRowTmp, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufRowTmp2, alignedWi * sizeof(float));
        } else {
            pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
        }
    } else if constexpr (std::is_same<T, float>::value) {
        // FP32：strideW==1 走 W 维向量化快路径，否则降级 ComputeRowScalarAcc
        if (vecScalarOnly) {
            // 极端大 wi 时 tiling 已降级：仅保留 weightDilated，走 ComputeRow 标量路径
        } else if (CanUseFp32VecRow()) {
            pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufProd, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufGradRow, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
        } else {
            pipe.InitBuffer(tmpBufRowAcc, alignedWi * sizeof(float));
            pipe.InitBuffer(tmpBufOutH, alignedWi * sizeof(T));
            pipe.InitBuffer(tmpBufRowComp, alignedWi * sizeof(float));
        }
    }
    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value || std::is_same<T, float>::value) {
        // 事件 ID 一次性获取、循环内复用（热循环内反复 Fetch 是未定义行为）
        evtVToMte2_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        evtMte2ToV_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        evtVToMte3_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        evtMte3ToV_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
        evtVToS_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_S));
        evtSToV_ = static_cast<event_t>(pipe.FetchEventID(HardEvent::S_V));
    }
}

// 单行计算分派：与 aclnn IsConv3DVecFallbackCase 能力圈一致
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::ProcessRow(uint64_t r)
{
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        if (vecScalarOnly) {
            if (useScalarAcc) {
                ComputeRowScalarAcc(r);
            } else {
                ComputeRow(r);
            }
        } else if (alignedWi < FP32_KAHAN_MIN_ELEMS) {
            ComputeRowScalarAcc(r);
        } else if (CanUseBf16VecRow()) {
            ComputeRowBf16VecPlain(r);
        } else if (CanUseBf16VecStridedRow()) {
            ComputeRowBf16VecStrided(r);
        } else {
            ComputeRowScalarAcc(r);
        }
    } else if constexpr (std::is_same<T, half>::value) {
        // FP16/BF16：FP32 累加 + 行末单次舍入，消除逐点舍入漂移满足 L2
        if (vecScalarOnly) {
            if (useScalarAcc) {
                ComputeRowScalarAcc(r);
            } else {
                ComputeRow(r);
            }
        } else if (alignedWi < FP32_KAHAN_MIN_ELEMS) {
            ComputeRowScalarAcc(r);
        } else if (CanUseBf16VecRow()) {
            ComputeRowBf16VecPlain(r);
        } else {
            ComputeRowScalarAcc(r);
        }
    } else if constexpr (std::is_same<T, float>::value) {
        // FP32：累加顺序与 ComputeRow 一致，向量化无精度变化，避免大 shape 超时
        if (vecScalarOnly) {
            ComputeRow(r);
        } else if (CanUseFp32VecRow()) {
            if (UseDhBlockedAcc()) {
                ComputeRowBf16Vec(r);
            } else {
                ComputeRowBf16VecPlain(r);
            }
        } else {
            ComputeRowScalarAcc(r);
        }
    } else {
        ComputeRow(r);
    }
}

// 行坐标分解：r -> (n, ci, diRow, hi)，与 Process 的行分区一一对应
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::DecomposeRow(uint64_t r, uint32_t& n, uint32_t& ci,
                                                                         uint32_t& diRow, uint32_t& hi) const
{
    uint64_t rest = r;
    hi = rest % gradInH;
    rest /= gradInH;
    diRow = rest % gradInD;
    rest /= gradInD;
    ci = rest % gradInC;
    rest /= gradInC;
    n = static_cast<uint32_t>(rest);
}

template <typename T>
__aicore__ inline uint64_t KernelConv3dBackpropInputVecImpl<T>::GetRowBase(uint32_t n, uint32_t ci, uint32_t diRow,
                                                                           uint32_t hi) const
{
    return (static_cast<uint64_t>(n) * gradInC + ci) * gradInD * gradInH * gradInW +
           static_cast<uint64_t>(diRow) * gradInH * gradInW + static_cast<uint64_t>(hi) * gradInW;
}

// groups>1：输入行 ci 只归约同组输出通道
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::GetGroupRange(uint32_t ci, uint32_t& ciLocal,
                                                                          uint32_t& coStart, uint32_t& coEnd) const
{
    const uint32_t group = ci / cinPerGroup;
    ciLocal = ci - group * cinPerGroup;
    coStart = group * coutPerGroup;
    coEnd = coStart + coutPerGroup;
}

// 构造膨胀后的 weight 行（dilatedHk x alignedDilatedW），供 W 维 tap 累加复用
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::BuildWeightDilated(uint32_t co, uint32_t ciLocal,
                                                                               uint32_t dk,
                                                                               LocalTensor<T>& weightDilated)
{
    for (uint32_t i = 0; i < dilatedHk * alignedDilatedW; i++) {
        weightDilated.SetValue(i, static_cast<T>(0));
    }
    const uint64_t wBase = (static_cast<uint64_t>(co) * cinPerGroup + ciLocal) * kernelD * kernelH * kernelW +
                           static_cast<uint64_t>(dk) * kernelH * kernelW;
    for (uint32_t kh = 0; kh < kernelH; kh++) {
        for (uint32_t kw = 0; kw < kernelW; kw++) {
            const uint32_t dh = (kernelH - 1 - kh) * dilationH;
            const uint32_t dw = (kernelW - 1 - kw) * dilationW;
            weightDilated.SetValue(dh * alignedDilatedW + dw, weightGm.GetValue(wBase + kh * kernelW + kw));
        }
    }
}

template <typename T>
__aicore__ inline uint64_t KernelConv3dBackpropInputVecImpl<T>::GetGoPlaneBase(uint32_t n, uint32_t co,
                                                                               uint32_t doIdx) const
{
    return (static_cast<uint64_t>(n) * gradOutC + co) * gradOutD * gradOutH * gradOutW +
           static_cast<uint64_t>(doIdx) * gradOutH * gradOutW;
}

// dh 行号校验：返回 false 表示该 dh 无有效输出行（对齐 cube 的 ho 边界判定）
template <typename T>
__aicore__ inline bool KernelConv3dBackpropInputVecImpl<T>::GetHoIndex(uint32_t dh, uint32_t hi, uint32_t& ho) const
{
    const int32_t hoRaw = static_cast<int32_t>(dh + hi) - padUDx;
    if (hoRaw < 0 || hoRaw % strideH != 0) {
        return false;
    }
    ho = static_cast<uint32_t>(hoRaw / strideH);
    return ho < static_cast<uint32_t>(gradOutH);
}

// BF16/FP16/FP32 快路径：dh/dw tap 累加（每 tap 一次向量行累加）
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::AccumulateBf16VecRowTaps(uint64_t goPlaneBase, uint32_t hi,
                                                                                     LocalTensor<float>& rowAcc,
                                                                                     LocalTensor<float>& prod,
                                                                                     LocalTensor<T>& gradRow,
                                                                                     LocalTensor<T>& weightDilated)
{
    for (uint32_t dh = 0; dh < dilatedHk; dh++) {
        uint32_t ho;
        if (!GetHoIndex(dh, hi, ho)) {
            continue;
        }
        const uint64_t goRowBase = goPlaneBase + static_cast<uint64_t>(ho) * gradOutW;
        for (uint32_t dw = 0; dw < dilatedWk; dw++) {
            const float wv = VecToFloat(weightDilated.GetValue(dh * alignedDilatedW + dw));
            if (wv == 0.0f) {
                continue;
            }
            AccumulateBf16Tap(goRowBase, dw, wv, rowAcc, prod, gradRow);
        }
    }
}

// BF16 strided-W 相位 tap：单相位内 dh/dw 累加（dw 按相位间隔 strideW）
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::AccumulateBf16VecStridedTapRow(
    uint64_t goPlaneBase, uint32_t hi, uint32_t kMax, uint32_t kAligned, int32_t dw0, int32_t c0, uint32_t sW,
    LocalTensor<float>& rowAccP, LocalTensor<float>& prodP, LocalTensor<T>& gradRowP, LocalTensor<T>& weightDilated)
{
    for (uint32_t dh = 0; dh < dilatedHk; dh++) {
        uint32_t ho;
        if (!GetHoIndex(dh, hi, ho)) {
            continue;
        }
        const uint64_t goRowBase = goPlaneBase + static_cast<uint64_t>(ho) * gradOutW;
        for (int32_t dw = dw0; dw < static_cast<int32_t>(dilatedWk); dw += static_cast<int32_t>(sW)) {
            const float wv = VecToFloat(weightDilated.GetValue(dh * alignedDilatedW + static_cast<uint32_t>(dw)));
            if (wv == 0.0f) {
                continue;
            }
            const int32_t delta = c0 + (dw - dw0) / static_cast<int32_t>(sW);
            AccumulateBf16TapClass(goRowBase, delta, wv, rowAccP, prodP, gradRowP, kMax, kAligned);
        }
    }
}

// BF16 strided-W：co/doIdx/dk 归约循环（每 (co,dk) 重建膨胀 weight 后做相位 tap 累加）
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::AccumulateBf16VecStridedTaps(
    uint32_t ciLocal, uint32_t coStart, uint32_t coEnd, uint32_t n, uint32_t diRow, uint32_t hi, uint32_t kMax,
    uint32_t kAligned, int32_t dw0, int32_t c0, uint32_t sW, LocalTensor<float>& rowAccP, LocalTensor<float>& prodP,
    LocalTensor<T>& gradRowP)
{
    LocalTensor<T> weightDilated = tmpBufWeightDilated.Get<T>();
    for (uint32_t co = coStart; co < coEnd; co++) {
        for (uint32_t doIdx = 0; doIdx < gradOutD; doIdx++) {
            for (uint32_t dk = 0; dk < kernelD; dk++) {
                const int32_t di = static_cast<int32_t>(doIdx) * strideD + static_cast<int32_t>(dk) * dilationD -
                                   padFront;
                if (di != static_cast<int32_t>(diRow)) {
                    continue;
                }
                BuildWeightDilated(co, ciLocal, dk, weightDilated);
                AccumulateBf16VecStridedTapRow(GetGoPlaneBase(n, co, doIdx), hi, kMax, kAligned, dw0, c0, sW, rowAccP,
                                               prodP, gradRowP, weightDilated);
            }
        }
    }
}

// 标量 tap 累加：useUbAcc=true 写 UB rowAcc（ScalarAcc），false 直接 GM 读改写（纯标量 ComputeRow）
template <typename T>
template <bool UseUbAcc>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::AccumulateScalarRowTaps(uint64_t goPlaneBase, uint32_t hi,
                                                                                    uint64_t rowBase,
                                                                                    LocalTensor<float>& rowAcc,
                                                                                    LocalTensor<T>& weightDilated)
{
    for (uint32_t dh = 0; dh < dilatedHk; dh++) {
        uint32_t ho;
        if (!GetHoIndex(dh, hi, ho)) {
            continue;
        }
        const uint64_t goRowBase = goPlaneBase + static_cast<uint64_t>(ho) * gradOutW;
        for (uint32_t dw = 0; dw < dilatedWk; dw++) {
            const float wv = VecToFloat(weightDilated.GetValue(dh * alignedDilatedW + dw));
            if (wv == 0.0f) {
                continue;
            }
            for (uint32_t wi = 0; wi < gradInW; wi++) {
                const int32_t woRaw = static_cast<int32_t>(dw + wi) - padLDx;
                if (woRaw < 0 || woRaw % strideW != 0) {
                    continue;
                }
                const int32_t wo = woRaw / strideW;
                if (wo >= static_cast<int32_t>(gradOutW)) {
                    continue;
                }
                const float gVal = VecToFloat(gradOutputGm.GetValue(goRowBase + static_cast<uint32_t>(wo)));
                if constexpr (UseUbAcc) {
                    const float prod = gVal * wv;
                    const float acc = rowAcc.GetValue(wi);
                    const float t = acc + prod;
                    const float e = (acc - t) + prod;
                    scalarComp.SetValue(wi, scalarComp.GetValue(wi) + e);
                    rowAcc.SetValue(wi, t);
                } else {
                    const float cur = VecToFloat(gradInputGm.GetValue(rowBase + wi));
                    gradInputGm.SetValue(rowBase + wi, VecFromFloat<T>(cur + gVal * wv));
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::ComputeRowBf16VecPlain(uint64_t r)
{
    uint32_t n, ci, diRow, hi;
    DecomposeRow(r, n, ci, diRow, hi);
    const uint64_t rowBase = GetRowBase(n, ci, diRow, hi);

    LocalTensor<float> rowAcc = tmpBufRowAcc.Get<float>();
    LocalTensor<float> prod = tmpBufProd.Get<float>();
    LocalTensor<T> gradRow = tmpBufGradRow.Get<T>();
    LocalTensor<T> outH = tmpBufOutH.Get<T>();
    LocalTensor<T> weightDilated = tmpBufWeightDilated.Get<T>();

    Duplicate(rowAcc, 0.0f, alignedWi);
    Duplicate(tmpBufRowComp.Get<float>(), 0.0f, alignedWi);

    uint32_t ciLocal, coStart, coEnd;
    GetGroupRange(ci, ciLocal, coStart, coEnd);
    for (uint32_t co = coStart; co < coEnd; co++) {
        for (uint32_t doIdx = 0; doIdx < gradOutD; doIdx++) {
            for (uint32_t dk = 0; dk < kernelD; dk++) {
                const int32_t di = static_cast<int32_t>(doIdx) * strideD + static_cast<int32_t>(dk) * dilationD -
                                   padFront;
                if (di != static_cast<int32_t>(diRow)) {
                    continue;
                }
                BuildWeightDilated(co, ciLocal, dk, weightDilated);
                AccumulateBf16VecRowTaps(GetGoPlaneBase(n, co, doIdx), hi, rowAcc, prod, gradRow, weightDilated);
            }
        }
    }

    Add(rowAcc, rowAcc, tmpBufRowComp.Get<float>(), alignedWi);
    CastAndWriteRow(rowBase, rowAcc, outH);
}

template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::ComputeRowBf16Vec(uint64_t r)
{
    if constexpr (std::is_same<T, float>::value) {
        uint32_t n, ci, diRow, hi;
        DecomposeRow(r, n, ci, diRow, hi);
        const uint64_t rowBase = GetRowBase(n, ci, diRow, hi);

        LocalTensor<float> rowAcc = tmpBufRowAcc.Get<float>();
        LocalTensor<float> prod = tmpBufProd.Get<float>();
        LocalTensor<T> gradRow = tmpBufGradRow.Get<T>();
        LocalTensor<T> outH = tmpBufOutH.Get<T>();
        LocalTensor<T> weightDilated = tmpBufWeightDilated.Get<T>();

        Duplicate(rowAcc, 0.0f, alignedWi);
        dhBlocked = UseDhBlockedAcc() ? 1U : 0U;
        if (dhBlocked != 0) {
            Duplicate(outH, static_cast<T>(0), alignedWi);
        } else {
            Duplicate(tmpBufRowComp.Get<float>(), 0.0f, alignedWi);
        }

        uint32_t ciLocal, coStart, coEnd;
        GetGroupRange(ci, ciLocal, coStart, coEnd);
        for (uint32_t co = coStart; co < coEnd; co++) {
            for (uint32_t doIdx = 0; doIdx < gradOutD; doIdx++) {
                for (uint32_t dk = 0; dk < kernelD; dk++) {
                    const int32_t di = static_cast<int32_t>(doIdx) * strideD + static_cast<int32_t>(dk) * dilationD -
                                       padFront;
                    if (di != static_cast<int32_t>(diRow)) {
                        continue;
                    }
                    BuildWeightDilated(co, ciLocal, dk, weightDilated);
                    if (dhBlocked != 0) {
                        AccumulateBf16VecRowTapsBlocked(GetGoPlaneBase(n, co, doIdx), hi, rowAcc, prod, gradRow,
                                                        weightDilated);
                    } else {
                        AccumulateBf16VecRowTaps(GetGoPlaneBase(n, co, doIdx), hi, rowAcc, prod, gradRow,
                                                 weightDilated);
                    }
                }
            }
        }

        if (dhBlocked != 0) {
            Add(rowAcc, rowAcc, outH, alignedWi);
        } else {
            Add(rowAcc, rowAcc, tmpBufRowComp.Get<float>(), alignedWi);
        }
        CastAndWriteRow(rowBase, rowAcc, outH);
    }
}

template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::AccumulateBf16VecRowTapsBlocked(
    uint64_t goPlaneBase, uint32_t hi, LocalTensor<float>& rowAcc, LocalTensor<float>& prod, LocalTensor<T>& gradRow,
    LocalTensor<T>& weightDilated)
{
    for (uint32_t dh = 0; dh < dilatedHk; dh++) {
        uint32_t ho;
        if (!GetHoIndex(dh, hi, ho)) {
            continue;
        }
        const uint64_t goRowBase = goPlaneBase + static_cast<uint64_t>(ho) * gradOutW;
        for (uint32_t dw = 0; dw < dilatedWk; dw++) {
            const float wv = VecToFloat(weightDilated.GetValue(dh * alignedDilatedW + dw));
            if (wv == 0.0f) {
                continue;
            }
            AccumulateBf16TapBlocked(goRowBase, dw, wv, rowAcc, prod, gradRow);
        }
        LocalTensor<float> bucket = tmpBufOutH.Get<float>();
        Add(bucket, bucket, rowAcc, alignedWi);
        Duplicate(rowAcc, 0.0f, alignedWi);
    }
}

// BF16 strided-W：按相位分解向量化，rowAcc 紧凑类缓冲全程 fp32，行末单次 CAST_RINT
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::ComputeRowBf16VecStrided(uint64_t r)
{
    uint32_t n, ci, diRow, hi;
    DecomposeRow(r, n, ci, diRow, hi);
    const uint64_t rowBase = GetRowBase(n, ci, diRow, hi);

    LocalTensor<float> rowAccP = tmpBufRowAccP.Get<float>();
    LocalTensor<float> prodP = tmpBufProdP.Get<float>();
    LocalTensor<T> gradRowP = tmpBufGradRowP.Get<T>();
    LocalTensor<T> outHP = tmpBufOutHP.Get<T>();

    const uint32_t sW = static_cast<uint32_t>(strideW);
    uint32_t ciLocal, coStart, coEnd;
    GetGroupRange(ci, ciLocal, coStart, coEnd);

    for (uint32_t p = 0; p < sW; p++) {
        const uint32_t kMax = (gradInW > p) ? (gradInW - p + sW - 1) / sW : 0;
        if (kMax == 0) {
            continue;
        }
        const uint32_t kAligned = ((kMax + dataPerBlock - 1) / dataPerBlock) * dataPerBlock;
        const int32_t dw0 = ((padLDx - static_cast<int32_t>(p)) % static_cast<int32_t>(sW) + static_cast<int32_t>(sW)) %
                            static_cast<int32_t>(sW);
        const int32_t c0 = (dw0 + static_cast<int32_t>(p) - padLDx) / static_cast<int32_t>(sW);

        Duplicate(rowAccP, 0.0f, kAligned);
        Duplicate(tmpBufRowComp.Get<float>(), 0.0f, kAligned);
        scalarTapParity = 0U;
        AccumulateBf16VecStridedTaps(ciLocal, coStart, coEnd, n, diRow, hi, kMax, kAligned, dw0, c0, sW, rowAccP, prodP,
                                     gradRowP);
        Add(rowAccP, rowAccP, tmpBufRowComp.Get<float>(), kAligned);

        // 累加完成 → 单次 CAST_RINT（与快路径/兜底同一舍入口径）
        SetFlag<HardEvent::V_S>(evtVToS_);
        WaitFlag<HardEvent::V_S>(evtVToS_);
        Cast(outHP, rowAccP, RoundMode::CAST_RINT, kAligned);
        SetFlag<HardEvent::V_S>(evtVToS_);
        WaitFlag<HardEvent::V_S>(evtVToS_);

        // 按类回写：类内位置行内间隔 strideW，标量写回
        const uint64_t rowBaseP = rowBase + p;
        for (uint32_t k = 0; k < kMax; k++) {
            gradInputGm.SetValue(rowBaseP + static_cast<uint64_t>(k) * sW, outHP.GetValue(k));
        }
        // 标量回写完成后才能进入下一相位（防 Cast 覆写 outHP）
        SetFlag<HardEvent::S_V>(evtSToV_);
        WaitFlag<HardEvent::S_V>(evtSToV_);
    }
}

// 纯标量兜底：逐点 GM 读改写，UB 仅 weightDilated（极端大 wi / UB 预算耗尽场景）
template <typename T>
__aicore__ inline void KernelConv3dBackpropInputVecImpl<T>::ComputeRow(uint64_t r)
{
    uint32_t n, ci, diRow, hi;
    DecomposeRow(r, n, ci, diRow, hi);
    const uint64_t rowBase = GetRowBase(n, ci, diRow, hi);
    for (uint32_t wi = 0; wi < gradInW; wi++) {
        gradInputGm.SetValue(rowBase + wi, T(0));
    }

    LocalTensor<T> weightDilated = tmpBufWeightDilated.Get<T>();
    // 纯标量路径不分配 rowAcc：借用 weightDilated 缓冲地址，useUbAcc=false 分支不会访问
    LocalTensor<float> rowAccDummy = tmpBufWeightDilated.Get<float>();
    uint32_t ciLocal, coStart, coEnd;
    GetGroupRange(ci, ciLocal, coStart, coEnd);
    for (uint32_t co = coStart; co < coEnd; co++) {
        for (uint32_t doIdx = 0; doIdx < gradOutD; doIdx++) {
            for (uint32_t dk = 0; dk < kernelD; dk++) {
                const int32_t di = static_cast<int32_t>(doIdx) * strideD + static_cast<int32_t>(dk) * dilationD -
                                   padFront;
                if (di != static_cast<int32_t>(diRow)) {
                    continue;
                }
                BuildWeightDilated(co, ciLocal, dk, weightDilated);
                AccumulateScalarRowTaps<false>(GetGoPlaneBase(n, co, doIdx), hi, rowBase, rowAccDummy, weightDilated);
            }
        }
    }
}

#endif // CONV3D_DX_V2_VEC_IMPL_ROW_H
