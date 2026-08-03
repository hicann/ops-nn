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
 * \file apply_adam_w_quant_base.h
 * \brief
 */
#ifndef _APPLY_ADAM_W_QUANT_BASE_H_
#define _APPLY_ADAM_W_QUANT_BASE_H_

#include "kernel_operator.h"
#include "apply_adam_w_quant_tiling_data.h"

namespace ApplyAdamWQuantNS {
using namespace AscendC;

constexpr int32_t Q_MAP_SIZE = 256;
constexpr uint32_t CALC_BUF_NUM = 6;
constexpr uint32_t PER_UINT8_8BITS = 8;
constexpr uint32_t REPEAT_NUM = 64;
constexpr uint32_t REPEAT_7_TIMES = 7;
constexpr uint32_t REPEAT_NUM_128 = 128;
constexpr uint32_t STRIDE_8 = 8;
constexpr uint32_t PER_4NUM_ONEMAX = 4;
constexpr uint32_t BROADCAST_DIM2 = 2;
constexpr uint32_t BROADCAST_AXIS1 = 1;

template <typename T>
__aicore__ inline void DataCopyIn(const AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& src,
                                  uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void DataCopyOut(const AscendC::GlobalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
                                   uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};

    AscendC::DataCopyPad(dst, src, copyParams);
}

template <typename T, typename T1>
__aicore__ inline void CastF16ToFp32(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T1>& src,
                                     uint32_t count)
{
    AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
}

template <typename T, typename T1>
__aicore__ inline void CastFp32ToF16(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T1>& src,
                                     uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
    } else { // bf16
        Cast(dst, src, AscendC::RoundMode::CAST_RINT, count);
    }
}

template <AscendC::HardEvent hardEvent>
__aicore__ inline void PipeSync()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
    AscendC::SetFlag<hardEvent>(eventID);
    AscendC::WaitFlag<hardEvent>(eventID);
}

__aicore__ inline float PowS(const AscendC::LocalTensor<float>& dst, float srcScalar,
                             const AscendC::LocalTensor<float>& src)
{
    AscendC::Power<float>(dst, srcScalar, src);
    PipeSync<AscendC::HardEvent::V_S>();
    float ret = dst.GetValue(0);
    AscendC::PipeBarrier<PIPE_ALL>();
    return ret;
}

// RegBase VF：AdamW 状态/参数逐元素更新（fp32 寄存器计算 body）。
// fp32 与 fp16 路径共用——两者进入本函数时 m/v/grad/var 均为 fp32 UB tile。
// 由 UB 层 UpdateStateAndParam(Fp32) 经 asc_vf_call 调用，替代原 tile-API Muls/Mul/Add/Sqrt/Adds/Div 链。
//   grad'  = grad * gnormScale
//   m_out  = m * beta1 + grad' * (1 - beta1)
//   v_out  = v * beta2 + grad'^2 * (1 - beta2)
//   var_out= (var + (m_out * stepSize) / (sqrt(v_out) + eps*correction2)) * wdFactor
//   wdFactor = (weight_decay>0) ? (1 - lr*weight_decay) : 1.0f（乘 1.0 为 no-op，等价于原分支跳过）
template <typename T>
__simd_vf__ inline void AdamWQuantUpdateVF(__ubuf__ T* dqStateMAddr, __ubuf__ T* dqStateVAddr, __ubuf__ T* varAddr,
                                           __ubuf__ T* gradAddr, float gnormScale, float beta1, float oneMinusBeta1,
                                           float beta2, float oneMinusBeta2, float epsCorr, float stepSize,
                                           float wdFactor, uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    using namespace AscendC::Reg;
    RegTensor<T> regM, regV, regGrad, regVar, regTmp;
    MaskReg mask;
    AddrReg aReg;
    uint32_t remain = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = CreateAddrReg<T>(i, oneRepeatSize);
        mask = UpdateMask<T>(remain);

        LoadAlign(regM, dqStateMAddr, aReg);
        LoadAlign(regV, dqStateVAddr, aReg);
        LoadAlign(regVar, varAddr, aReg);
        LoadAlign(regGrad, gradAddr, aReg);

        // grad = grad * gnormScale
        Muls(regGrad, regGrad, gnormScale, mask);
        // m = m*beta1 + grad*(1-beta1)
        Muls(regM, regM, beta1, mask);
        Muls(regTmp, regGrad, oneMinusBeta1, mask);
        Add(regM, regM, regTmp, mask);
        // v = v*beta2 + grad^2*(1-beta2)
        Muls(regV, regV, beta2, mask);
        Mul(regGrad, regGrad, regGrad, mask); // grad^2
        Muls(regTmp, regGrad, oneMinusBeta2, mask);
        Add(regV, regV, regTmp, mask);
        // denom = sqrt(v) + eps*correction2
        Sqrt(regTmp, regV, mask);
        Adds(regTmp, regTmp, epsCorr, mask);
        // var = var + (m*stepSize)/denom（regGrad 复用为 m*stepSize，对齐原 tmpVar1 别名 grad 缓存）
        Muls(regGrad, regM, stepSize, mask);
        Div(regGrad, regGrad, regTmp, mask);
        Add(regVar, regVar, regGrad, mask);
        // var = var * wdFactor（无 weight_decay 时 wdFactor=1）
        Muls(regVar, regVar, wdFactor, mask);

        StoreAlign(dqStateMAddr, regM, aReg, mask);
        StoreAlign(dqStateVAddr, regV, aReg, mask);
        StoreAlign(varAddr, regVar, aReg, mask);
    }
}

} // namespace ApplyAdamWQuantNS

#endif // _APPLY_ADAM_W_QUANT_BASE_H_
