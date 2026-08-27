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
 * \file dequant_swiglu_quant_dynamic_not_full_vf.h
 * \brief 动态非全载模板VF_CALL的实现
 */

#ifndef DEQUANT_SWIGLU_QUANT_DYNAMIC_NOT_FULL_VF_H
#define DEQUANT_SWIGLU_QUANT_DYNAMIC_NOT_FULL_VF_H

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

template <typename XType, typename BiasType, bool IsXInt, bool IsXFloat16, bool IsXBfloat16, bool HasBias,
          bool IsBiasInt, bool HasActiScale, bool IsBiasFloat, bool IsBiasFloat16, bool IsBiasBfloat16>
__aicore__ inline void DequantAndSwiGluV2(__ubuf__ XType* xPtr, __ubuf__ float* wScalePtr, __ubuf__ float* aScalePtr,
                                          __ubuf__ BiasType* biasPtr, float clampLit, float gluAlpha, float gluBias,
                                          __ubuf__ float* xTempPtr, uint32_t tileData)
{
    // 输入x寄存器
    AscendC::Reg::RegTensor<XType> vregX1, vregX2;
    // 输入weight寄存器
    AscendC::Reg::RegTensor<float> vregW1, vregW2;
    // 输入bias为int寄存器
    AscendC::Reg::RegTensor<int32_t> vregBiasInt1, vregBiasInt2;
    // 临时寄存器
    AscendC::Reg::RegTensor<float> vreg0, vreg1;
    // 结果(dst)寄存器
    AscendC::Reg::RegTensor<float> vregD1, vregD2;
    // 输入activation寄存器
    AscendC::Reg::RegTensor<float> vregA;
    // 输入bias为float32寄存器
    AscendC::Reg::RegTensor<float> vregBiasFloat1, vregBiasFloat2;
    // 输入bias为float16寄存器
    AscendC::Reg::RegTensor<half> vregBiasHalf1, vregBiasHalf2;
    // 输入bias为bfloat16寄存器
    AscendC::Reg::RegTensor<bfloat16_t> vregBiasBF1, vregBiasBF2;
    // SwiGlu临时寄存器
    AscendC::Reg::RegTensor<float> vregS0, vregS1, vregS2, vregS3, vregS4, vregS5, vregS6;
    AscendC::Reg::MaskReg mask, tailMask;
    const float scalarOne = 1.0;

    // int32： 256B / 4B = 64次 ，bf16 or float16：256B / 2B = 128次
    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint32_t width = sizePerRepeat;
    uint16_t repeatTimes = CeilDivision(tileData, sizePerRepeat);
    mask = AscendC::Reg::UpdateMask<uint32_t>(width);

    for (uint16_t j = 0; j < repeatTimes - 1; j++) {
        // x的数据类型变换之后，对齐点变化了，应该用xTypeUb参数
        auto x1Addr = xPtr + 2 * j * sizePerRepeat;
        auto x2Addr = x1Addr + sizePerRepeat;
        auto dstAddr = xTempPtr + j * sizePerRepeat;
        // vreg0 -> x1, vreg10 -> x2
        // bf16和float16场景下需要修改，新增搬运模板参数DIST_UNPACK_B16
        if constexpr (IsXFloat16) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1, x1Addr);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX2, x2Addr);
        }
        if constexpr (IsXBfloat16) {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1, x1Addr);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX2, x2Addr);
        }
        if constexpr (IsXInt) {
            AscendC::Reg::LoadAlign(vregX1, x1Addr);
            AscendC::Reg::LoadAlign(vregX2, x2Addr);
        }
        // 如果bias有值，且x=int32，bias=int32，则先将x+bias
        if constexpr (HasBias) {
            // 使用bias的地址指针时，做判空处理
            if constexpr (IsXInt && IsBiasInt) {
                auto bias1Addr = biasPtr + 2 * j * sizePerRepeat;
                auto bias2Addr = bias1Addr + sizePerRepeat;
                AscendC::Reg::LoadAlign(vregBiasInt1, bias1Addr);
                AscendC::Reg::LoadAlign(vregBiasInt2, bias2Addr);
                // x + bias
                AscendC::Reg::Add(vregX1, vregX1, vregBiasInt1, mask);
                AscendC::Reg::Add(vregX2, vregX2, vregBiasInt2, mask);
            }
        }
        // x:int32->float32
        if constexpr (IsXInt) {
            // 计算，当x=int32时，weight_scale才参与计算
            __ubuf__ float* wScale1Addr = wScalePtr + 2 * j * sizePerRepeat;
            __ubuf__ float* wScale2Addr = wScale1Addr + sizePerRepeat;
            AscendC::Reg::LoadAlign(vregW1, wScale1Addr);
            AscendC::Reg::LoadAlign(vregW2, wScale2Addr);
            AscendC::Reg::Cast<float, XType, CAST_INT32_TO_FP32>(vreg0, vregX1, mask);
            AscendC::Reg::Cast<float, XType, CAST_INT32_TO_FP32>(vreg1, vregX2, mask);
            // x * weight
            AscendC::Reg::Mul(vregD1, vreg0, vregW1, mask);
            AscendC::Reg::Mul(vregD2, vreg1, vregW2, mask);
        }
        // x:float16, bfloat16 -> float32, 不进行x * weight
        if constexpr (IsXBfloat16 || IsXFloat16) {
            AscendC::Reg::Cast<float, XType, CAST_BF16_FP16_TO_FP32>(vregD1, vregX1, mask);
            AscendC::Reg::Cast<float, XType, CAST_BF16_FP16_TO_FP32>(vregD2, vregX2, mask);
        }
        // x * activation_scale
        if constexpr (HasActiScale) {
            auto aScaleAddr = aScalePtr;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregA, aScaleAddr);
            AscendC::Reg::Mul(vregD1, vregD1, vregA, mask);
            AscendC::Reg::Mul(vregD2, vregD2, vregA, mask);
        }
        // 如果bias有值，且x!=int32 && bias!=int32，则先将dequant后的结果加上bias
        if constexpr (IsBiasFloat || IsBiasFloat16 || IsBiasBfloat16) {
            // 确认bias的计算地址
            auto bias1Addr = biasPtr + 2 * j * sizePerRepeat;
            auto bias2Addr = bias1Addr + sizePerRepeat;
            if constexpr (IsBiasFloat) {
                AscendC::Reg::LoadAlign(vregBiasFloat1, bias1Addr);
                AscendC::Reg::LoadAlign(vregBiasFloat2, bias2Addr);
            }
            if constexpr (IsBiasFloat16) {
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasHalf1, bias1Addr);
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasHalf2, bias2Addr);
                AscendC::Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vregBiasFloat1, vregBiasHalf1, mask);
                AscendC::Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vregBiasFloat2, vregBiasHalf2, mask);
            }
            if constexpr (IsBiasBfloat16) {
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasBF1, bias1Addr);
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasBF2, bias2Addr);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vregBiasFloat1, vregBiasBF1, mask);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vregBiasFloat2, vregBiasBF2, mask);
            }
            // 将dequant后的结果加上bias
            AscendC::Reg::Add(vregD1, vregD1, vregBiasFloat1, mask);
            AscendC::Reg::Add(vregD2, vregD2, vregBiasFloat2, mask);
        }

        // Swish
        AscendC::Reg::DeInterleave(vregS0, vregS1, vregD1, vregD2);
        AscendC::Reg::Mins(vregS0, vregS0, clampLit, mask);
        AscendC::Reg::Muls(vregS2, vregS0, -gluAlpha, mask);
        AscendC::Reg::Exp(vregS3, vregS2, mask);
        AscendC::Reg::Adds(vregS4, vregS3, scalarOne, mask);
        AscendC::Reg::Div<float, &DIV_MODE>(vregS5, vregS0, vregS4, mask);
        // glu
        AscendC::Reg::Mins(vregS1, vregS1, clampLit, mask);
        AscendC::Reg::Maxs(vregS1, vregS1, -clampLit, mask);
        AscendC::Reg::Adds(vregS1, vregS1, gluBias, mask);
        AscendC::Reg::Mul(vregS6, vregS5, vregS1, mask);
        // store: reg->ub
        AscendC::Reg::StoreAlign(dstAddr, vregS6, mask);
    }

    // 求尾块大小
    uint32_t tailCount = 2 * (tileData - (repeatTimes - 1) * sizePerRepeat);
    // 判断尾块需要处理次数奇偶
    uint32_t ifEven = 0;

    // 尾块需要的VF次数
    uint16_t tailRepeatTimes = CeilDivision(tailCount, sizePerRepeat);
    // 如果需要2次VF
    if ((tailRepeatTimes & 1) == 0) {
        ifEven = 1;
    }

    uint32_t tailWidth = tailCount / 2;
    tailMask = AscendC::Reg::UpdateMask<uint32_t>(tailWidth);

    uint16_t j = repeatTimes - 1;
    // x的数据类型变换之后，对齐点变化了，应该用xTypeUb参数
    auto x1Addr = xPtr + 2 * j * sizePerRepeat;
    auto dstAddr = xTempPtr + j * sizePerRepeat;
    __ubuf__ float* wScale1Addr;
    __ubuf__ BiasType* bias1Addr;
    __ubuf__ float* aScaleAddr;
    // vreg0 -> x1, vreg10 -> x2
    // bf16和float16场景下需要修改，新增搬运模板参数DIST_UNPACK_B16
    if constexpr (IsXFloat16) {
        AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1, x1Addr);
    }
    if constexpr (IsXBfloat16) {
        AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX1, x1Addr);
    }
    if constexpr (IsXInt) {
        AscendC::Reg::LoadAlign(vregX1, x1Addr);
    }
    // 如果bias有值，且x=int32，bias=int32，则先将x+bias
    if constexpr (HasBias) {
        // 使用bias的地址指针时，做判空处理
        if constexpr (IsXInt && IsBiasInt) {
            bias1Addr = biasPtr + 2 * j * sizePerRepeat;
            AscendC::Reg::LoadAlign(vregBiasInt1, bias1Addr);
            // x + bias
            AscendC::Reg::Add(vregX1, vregX1, vregBiasInt1, mask);
        }
    }
    // x:int32->float32
    if constexpr (IsXInt) {
        // 计算，当x=int32时，weight_scale才参与计算
        wScale1Addr = wScalePtr + 2 * j * sizePerRepeat;
        AscendC::Reg::LoadAlign(vregW1, wScale1Addr);
        AscendC::Reg::Cast<float, XType, CAST_INT32_TO_FP32>(vreg0, vregX1, mask);
        // x * weight
        AscendC::Reg::Mul(vregD1, vreg0, vregW1, mask);
    }
    // x:float16, bfloat16 -> float32, 不进行x * weight
    if constexpr (IsXBfloat16 || IsXFloat16) {
        AscendC::Reg::Cast<float, XType, CAST_BF16_FP16_TO_FP32>(vregD1, vregX1, mask);
    }
    // x * activation_scale
    if constexpr (HasActiScale) {
        aScaleAddr = aScalePtr;
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregA, aScaleAddr);
        AscendC::Reg::Mul(vregD1, vregD1, vregA, mask);
    }
    // 如果bias有值，且x!=int32 && bias!=int32，则先将dequant后的结果加上bias
    if constexpr (IsBiasFloat || IsBiasFloat16 || IsBiasBfloat16) {
        // 确认bias的计算地址
        bias1Addr = biasPtr + 2 * j * sizePerRepeat;
        if constexpr (IsBiasFloat) {
            AscendC::Reg::LoadAlign(vregBiasFloat1, bias1Addr);
        }
        if constexpr (IsBiasFloat16) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasHalf1, bias1Addr);
            AscendC::Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vregBiasFloat1, vregBiasHalf1, mask);
        }
        if constexpr (IsBiasBfloat16) {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasBF1, bias1Addr);
            AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vregBiasFloat1, vregBiasBF1, mask);
        }
        // 将dequant后的结果加上bias
        AscendC::Reg::Add(vregD1, vregD1, vregBiasFloat1, mask);
    }
    if (ifEven) {
        auto x2Addr = x1Addr + sizePerRepeat;
        // vreg0 -> x1, vreg10 -> x2
        // bf16和float16场景下需要修改，新增搬运模板参数DIST_UNPACK_B16
        if constexpr (IsXFloat16) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX2, x2Addr);
        }
        if constexpr (IsXBfloat16) {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregX2, x2Addr);
        }
        if constexpr (IsXInt) {
            AscendC::Reg::LoadAlign(vregX2, x2Addr);
        }
        // 如果bias有值，且x=int32，bias=int32，则先将x+bias
        if constexpr (HasBias) {
            // 使用bias的地址指针时，做判空处理
            if constexpr (IsXInt && IsBiasInt) {
                auto bias2Addr = bias1Addr + sizePerRepeat;
                AscendC::Reg::LoadAlign(vregBiasInt2, bias2Addr);
                // x + bias
                AscendC::Reg::Add(vregX2, vregX2, vregBiasInt2, mask);
            }
        }
        // x:int32->float32
        if constexpr (IsXInt) {
            // 计算，当x=int32时，weight_scale才参与计算
            __ubuf__ float* wScale2Addr = wScale1Addr + sizePerRepeat;
            AscendC::Reg::LoadAlign(vregW2, wScale2Addr);
            AscendC::Reg::Cast<float, XType, CAST_INT32_TO_FP32>(vreg1, vregX2, mask);
            // x * weight
            AscendC::Reg::Mul(vregD2, vreg1, vregW2, mask);
        }
        // x:float16, bfloat16 -> float32, 不进行x * weight
        if constexpr (IsXBfloat16 || IsXFloat16) {
            AscendC::Reg::Cast<float, XType, CAST_BF16_FP16_TO_FP32>(vregD2, vregX2, mask);
        }
        // x * activation_scale
        if constexpr (HasActiScale) {
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vregA, aScaleAddr);
            AscendC::Reg::Mul(vregD2, vregD2, vregA, mask);
        }
        // 如果bias有值，且x!=int32 && bias!=int32，则先将dequant后的结果加上bias
        if constexpr (IsBiasFloat || IsBiasFloat16 || IsBiasBfloat16) {
            // 确认bias的计算地址
            auto bias2Addr = bias1Addr + sizePerRepeat;
            if constexpr (IsBiasFloat) {
                AscendC::Reg::LoadAlign(vregBiasFloat2, bias2Addr);
            }
            if constexpr (IsBiasFloat16) {
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasHalf2, bias2Addr);
                AscendC::Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vregBiasFloat2, vregBiasHalf2, mask);
            }
            if constexpr (IsBiasBfloat16) {
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vregBiasBF2, bias2Addr);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vregBiasFloat2, vregBiasBF2, mask);
            }
            // 将dequant后的结果加上bias
            AscendC::Reg::Add(vregD2, vregD2, vregBiasFloat2, mask);
        }
        AscendC::Reg::DeInterleave(vregS0, vregS1, vregD1, vregD2);
    } else {
        AscendC::Reg::DeInterleave(vregS0, vregS1, vregD1, vregD1);
    }

    // Swish
    AscendC::Reg::Mins(vregS0, vregS0, clampLit, mask);
    AscendC::Reg::Muls(vregS2, vregS0, -gluAlpha, mask);
    AscendC::Reg::Exp(vregS3, vregS2, mask);
    AscendC::Reg::Adds(vregS4, vregS3, scalarOne, mask);
    AscendC::Reg::Div<float, &DIV_MODE>(vregS5, vregS0, vregS4, mask);
    // glu
    AscendC::Reg::Mins(vregS1, vregS1, clampLit, mask);
    AscendC::Reg::Maxs(vregS1, vregS1, -clampLit, mask);
    AscendC::Reg::Adds(vregS1, vregS1, gluBias, mask);
    AscendC::Reg::Mul(vregS6, vregS5, vregS1, mask);
    // store: reg->ub
    AscendC::Reg::StoreAlign(dstAddr, vregS6, mask);
}

template <typename XType, typename BiasType, bool IsXInt, bool IsXFloat16, bool IsXBfloat16, bool HasBias,
          bool IsBiasInt, bool HasActiScale, bool IsBiasFloat, bool IsBiasFloat16, bool IsBiasBfloat16>
__aicore__ inline void DequantAndSwiGlu(__ubuf__ XType* x1Ptr, __ubuf__ XType* x2Ptr, __ubuf__ float* wScale1Ptr,
                                        __ubuf__ float* wScale2Ptr, __ubuf__ float* aScalePtr,
                                        __ubuf__ BiasType* bias1Ptr, __ubuf__ BiasType* bias2Ptr,
                                        __ubuf__ float* xTempPtr, uint32_t tileData)
{
    AscendC::Reg::RegTensor<XType> vreg0, vreg10;
    AscendC::Reg::RegTensor<float> vreg1, vreg2, vreg3, vreg4, vreg5, vreg6;
    AscendC::Reg::RegTensor<float> vreg7, vreg8, vreg9, vreg11, vreg12;
    AscendC::Reg::RegTensor<float> vreg13, vreg14, vreg15;
    AscendC::Reg::RegTensor<int32_t> vreg16, vreg17, verg18, vreg19;
    AscendC::Reg::RegTensor<float> vreg20, vreg21;
    AscendC::Reg::RegTensor<half> vreg24, vreg25;
    AscendC::Reg::RegTensor<bfloat16_t> vreg26, vreg27;
    AscendC::Reg::MaskReg mask;

    // int32： 256B / 4B = 64次 ，bf16 or float16：256B / 2B = 128次
    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint32_t width = tileData;
    uint16_t repeatTimes = CeilDivision(tileData, sizePerRepeat);
    const float scalarOne = 1.0;
    for (uint16_t j = 0; j < repeatTimes; j++) {
        mask = AscendC::Reg::UpdateMask<uint32_t>(width);
        // 计算，当x=int32时，weight_scale才参与计算
        __ubuf__ float* wScale1Addr;
        __ubuf__ float* wScale2Addr;
        if constexpr (IsXInt) {
            wScale1Addr = wScale1Ptr + j * sizePerRepeat;
            wScale2Addr = wScale2Ptr + j * sizePerRepeat;
            AscendC::Reg::LoadAlign(vreg2, wScale1Addr);
            AscendC::Reg::LoadAlign(vreg12, wScale2Addr);
        }
        // x的数据类型变换之后，对齐点变化了，应该用xTypeUb参数
        auto x1Addr = x1Ptr + j * sizePerRepeat;
        auto x2Addr = x2Ptr + j * sizePerRepeat;
        auto dstAddr = xTempPtr + j * sizePerRepeat;
        // vreg0 -> x1, vreg10 -> x2
        // bf16和float16场景下需要修改，新增搬运模板参数DIST_UNPACK_B16
        if constexpr (IsXFloat16) {
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg0, x1Addr);
            AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg10, x2Addr);
        }
        if constexpr (IsXBfloat16) {
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg0, x1Addr);
            AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg10, x2Addr);
        }
        if constexpr (IsXInt) {
            AscendC::Reg::LoadAlign(vreg0, x1Addr);
            AscendC::Reg::LoadAlign(vreg10, x2Addr);
        }
        // 如果bias有值，且x=int32，bias=int32，则先将x+bias
        if constexpr (HasBias) {
            // 使用bias的地址指针时，做判空处理
            if constexpr (IsXInt && IsBiasInt) {
                auto bias1Addr = bias1Ptr + j * sizePerRepeat;
                auto bias2Addr = bias2Ptr + j * sizePerRepeat;
                AscendC::Reg::LoadAlign(vreg16, bias1Addr);
                AscendC::Reg::LoadAlign(vreg17, bias2Addr);
                // x + bias
                AscendC::Reg::Add(vreg0, vreg0, vreg16, mask);
                AscendC::Reg::Add(vreg10, vreg10, vreg17, mask);
            }
        }
        // x:int32->float32
        if constexpr (IsXInt) {
            AscendC::Reg::Cast<float, XType, CAST_INT32_TO_FP32>(vreg1, vreg0, mask);
            AscendC::Reg::Cast<float, XType, CAST_INT32_TO_FP32>(vreg11, vreg10, mask);
            // x * weight
            AscendC::Reg::Mul(vreg3, vreg1, vreg2, mask);
            AscendC::Reg::Mul(vreg13, vreg11, vreg12, mask);
        }
        // x:float16, bfloat16 -> float32, 不进行x * weight
        if constexpr (IsXBfloat16 || IsXFloat16) {
            AscendC::Reg::Cast<float, XType, CAST_BF16_FP16_TO_FP32>(vreg3, vreg0, mask);
            AscendC::Reg::Cast<float, XType, CAST_BF16_FP16_TO_FP32>(vreg13, vreg10, mask);
        }
        // x * activation_scale
        if constexpr (HasActiScale) {
            auto aScaleAddr = aScalePtr;
            AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg4, aScaleAddr);
            AscendC::Reg::Mul(vreg3, vreg3, vreg4, mask);
            AscendC::Reg::Mul(vreg13, vreg13, vreg4, mask);
        }
        // 如果bias有值，且x!=int32 && bias!=int32，则先将dequant后的结果加上bias
        if constexpr (IsBiasFloat || IsBiasFloat16 || IsBiasBfloat16) {
            // 确认bias的计算地址
            auto bias1Addr = bias1Ptr + j * sizePerRepeat;
            auto bias2Addr = bias2Ptr + j * sizePerRepeat;
            if constexpr (IsBiasFloat) {
                AscendC::Reg::LoadAlign(vreg20, bias1Addr);
                AscendC::Reg::LoadAlign(vreg21, bias2Addr);
            }
            if constexpr (IsBiasFloat16) {
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg24, bias1Addr);
                AscendC::Reg::LoadAlign<half, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg25, bias2Addr);
                AscendC::Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vreg20, vreg24, mask);
                AscendC::Reg::Cast<float, half, CAST_BF16_FP16_TO_FP32>(vreg21, vreg25, mask);
            }
            if constexpr (IsBiasBfloat16) {
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg26, bias1Addr);
                AscendC::Reg::LoadAlign<bfloat16_t, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg27, bias2Addr);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vreg20, vreg26, mask);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_BF16_FP16_TO_FP32>(vreg21, vreg27, mask);
            }
            // 将dequant后的结果加上bias
            AscendC::Reg::Add(vreg3, vreg3, vreg20, mask);
            AscendC::Reg::Add(vreg13, vreg13, vreg21, mask);
        }
        // Swish
        AscendC::Reg::Muls(vreg6, vreg3, -(scalarOne), mask);
        AscendC::Reg::Exp(vreg7, vreg6, mask);
        AscendC::Reg::Adds(vreg8, vreg7, scalarOne, mask);
        AscendC::Reg::Div<float, &DIV_MODE>(vreg9, vreg3, vreg8, mask);
        // glu
        AscendC::Reg::Mul(vreg15, vreg9, vreg13, mask);
        // store: reg->ub
        AscendC::Reg::StoreAlign(dstAddr, vreg15, mask);
    }
}

template <bool HasQuantScale>
__aicore__ inline void CalculateReduceMax(__ubuf__ float* xTempPtr, __ubuf__ float* qScalePtr, __ubuf__ float* scalePtr,
                                          uint32_t tileData, float scalarMaxValue)
{
    AscendC::Reg::RegTensor<float> vreg0, vreg1, vreg2, vreg3, vreg4, vreg5, vreg6, vreg7, vreg8, vregDiv;
    AscendC::Reg::RegTensor<float> vregTmpX, vregTempScale;
    AscendC::Reg::RegTensor<float> vreg16;
    AscendC::Reg::RegTensor<int16_t> vreg17;
    AscendC::Reg::RegTensor<half> vreg18;
    AscendC::Reg::RegTensor<int8_t> vreg19;
    AscendC::Reg::MaskReg mask, maskTail, maskOne;

    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(tileData, sizePerRepeat);

    uint32_t block = static_cast<uint32_t>(sizePerRepeat);
    uint32_t tailBlock = tileData - sizePerRepeat * (repeatTimes - 1);
    uint32_t numOne = 1;
    mask = AscendC::Reg::UpdateMask<uint32_t>(block);
    maskTail = AscendC::Reg::UpdateMask<uint32_t>(tailBlock);
    maskOne = AscendC::Reg::UpdateMask<uint32_t>(numOne); // 用于读写1个元素

    AscendC::Reg::Duplicate(vregDiv, scalarMaxValue);

    auto scaleAddr = scalePtr;
    float scalarZero = 0;
    AscendC::Reg::Duplicate(vregTmpX, scalarZero);

    // 先处理尾块
    uint16_t j = repeatTimes - 1;
    auto tmpXAddr = xTempPtr + j * sizePerRepeat;
    AscendC::Reg::LoadAlign(vreg0, tmpXAddr);

    // x * quant_scale
    if constexpr (HasQuantScale) {
        auto qScaleAddr = qScalePtr + j * sizePerRepeat;
        AscendC::Reg::LoadAlign(vreg1, qScaleAddr);
        AscendC::Reg::Mul(vreg0, vreg0, vreg1, maskTail);
        AscendC::Reg::StoreAlign(tmpXAddr, vreg0, maskTail);
    }

    // 循环求行内最大值
    AscendC::Reg::Abs(vreg3, vreg0, maskTail);
    AscendC::Reg::Max(vregTmpX, vregTmpX, vreg3, maskTail);

    // 整块处理
    for (uint16_t j = 0; j < static_cast<uint16_t>(repeatTimes - 1); j++) {
        auto tmpXAddr = xTempPtr + j * sizePerRepeat;
        AscendC::Reg::LoadAlign(vreg0, tmpXAddr);

        // x * quant_scale
        if constexpr (HasQuantScale) {
            auto qScaleAddr = qScalePtr + j * sizePerRepeat;
            AscendC::Reg::LoadAlign(vreg1, qScaleAddr);
            AscendC::Reg::Mul(vreg0, vreg0, vreg1, mask);
            AscendC::Reg::StoreAlign(tmpXAddr, vreg0, mask);
        }

        // 循环求行内最大值
        AscendC::Reg::Abs(vreg3, vreg0, mask);
        AscendC::Reg::Max(vregTmpX, vregTmpX, vreg3, mask);
    }

    // vreg[0]为最大值，vreg[1]为最大值对应索引
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(vreg4, vregTmpX, mask);
    AscendC::Reg::Div(vreg5, vreg4, vregDiv, mask);
    // 和Scale已有的最大值进行对比
    AscendC::Reg::LoadAlign(vregTempScale, scaleAddr);
    AscendC::Reg::Max(vreg5, vregTempScale, vreg5, mask);
    // 拷贝第一个数到UB
    AscendC::Reg::StoreAlign(scaleAddr, vreg5, maskOne);
}

template <typename YType, bool IsFloat8E4M3, bool IsFloat8E5M2, bool IsFloat4E2M1, bool IsFloat4E1M2, bool IsHiFloat8>
__aicore__ inline void QuantAndCast(__ubuf__ float* xTempPtr, __ubuf__ YType* yPtr, __ubuf__ uint8_t* yFp4Ptr,
                                    __ubuf__ float* scalePtr, int64_t roundMode, uint32_t tileData)
{
    AscendC::Reg::RegTensor<float> vreg6, vreg7, vreg8;
    AscendC::Reg::RegTensor<int16_t> vreg9;
    AscendC::Reg::RegTensor<half> vreg10;
    AscendC::Reg::RegTensor<int8_t> vreg11;
    AscendC::Reg::RegTensor<fp8_e4m3fn_t> vreg12;
    AscendC::Reg::RegTensor<fp8_e5m2_t> vreg13;
    AscendC::Reg::RegTensor<bfloat16_t> vreg14, vreg15;
    AscendC::Reg::RegTensor<fp4x2_e2m1_t> vreg16;
    AscendC::Reg::RegTensor<fp4x2_e1m2_t> vreg17;
    AscendC::Reg::RegTensor<hifloat8_t> vreg18;
    AscendC::Reg::RegTensor<uint16_t> yRegTensor;
    AscendC::Reg::RegTensor<uint8_t> out;
    AscendC::Reg::MaskReg maskFull8;
    AscendC::Reg::MaskReg mask;
    uint32_t fp4Width = 32;
    constexpr uint16_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(tileData, sizePerRepeat);
    maskFull8 = AscendC::Reg::UpdateMask<uint32_t>(fp4Width);
    auto scaleAddr = scalePtr;
    AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg6, scaleAddr);
    uint32_t width = static_cast<uint32_t>(tileData);
    for (uint16_t j = 0; j < repeatTimes; j++) {
        mask = AscendC::Reg::UpdateMask<uint32_t>(width);
        auto tmpXAddr = xTempPtr + j * sizePerRepeat;
        auto yAddr = yPtr + j * sizePerRepeat;
        auto yFp4Addr = yFp4Ptr + (j * sizePerRepeat / 2);
        AscendC::Reg::LoadAlign(vreg7, tmpXAddr);
        AscendC::Reg::Div(vreg8, vreg7, vreg6, mask);
        // 根据输出类型进行不同的cast操作
        if constexpr (IsFloat8E4M3) {
            // float32 -> float8_e4m3
            AscendC::Reg::Cast<fp8_e4m3fn_t, float, CAST_FP32_TO_FP8>(vreg12, vreg8, mask);
            AscendC::Reg::StoreAlign<fp8_e4m3fn_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(yAddr, vreg12, mask);
        } else if constexpr (IsFloat8E5M2) {
            // float32 -> float8_e5m2
            AscendC::Reg::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FP8>(vreg13, vreg8, mask);
            AscendC::Reg::StoreAlign<fp8_e5m2_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(yAddr, vreg13, mask);
        } else if constexpr (IsFloat4E2M1) {
            // float32 -> bfloat16 -> float4_e2m1
            AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(vreg14, vreg8, mask);
            AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint16_t>&)vreg14, (AscendC::Reg::RegTensor<uint32_t>&)vreg14);
            if (roundMode == 1) {
                AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_ROUND>(vreg16, vreg14, mask);
            } else if (roundMode == 2) {
                AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_FLOOR>(vreg16, vreg14, mask);
            } else if (roundMode == 3) {
                AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_CEIL>(vreg16, vreg14, mask);
            } else if (roundMode == 4) {
                AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_TRUNC>(vreg16, vreg14, mask);
            } else {
                AscendC::Reg::Cast<fp4x2_e2m1_t, bfloat16_t, CAST_BF16_TO_FP4_RINT>(vreg16, vreg14, mask);
            }
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                yFp4Addr, (AscendC::Reg::RegTensor<uint8_t>&)vreg16, maskFull8);
        } else if constexpr (IsFloat4E1M2) {
            AscendC::Reg::Cast<bfloat16_t, float, CAST_FP32_TO_BF16>(vreg15, vreg8, mask);
            AscendC::Reg::Pack((AscendC::Reg::RegTensor<uint16_t>&)vreg15, (AscendC::Reg::RegTensor<uint32_t>&)vreg15);
            if (roundMode == 1) {
                AscendC::Reg::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_ROUND>(vreg17, vreg15, mask);
            } else if (roundMode == 2) {
                AscendC::Reg::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_FLOOR>(vreg17, vreg15, mask);
            } else if (roundMode == 3) {
                AscendC::Reg::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_CEIL>(vreg17, vreg15, mask);
            } else if (roundMode == 4) {
                AscendC::Reg::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_TRUNC>(vreg17, vreg15, mask);
            } else {
                AscendC::Reg::Cast<fp4x2_e1m2_t, bfloat16_t, CAST_BF16_TO_FP4_RINT>(vreg17, vreg15, mask);
            }
            AscendC::Reg::StoreAlign<uint8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(
                yFp4Addr, (AscendC::Reg::RegTensor<uint8_t>&)vreg17, maskFull8);
        } else if constexpr (IsHiFloat8) {
            AscendC::Reg::Cast<hifloat8_t, float, CAST_FP32_TO_HI8>(vreg18, vreg8, mask);
            AscendC::Reg::StoreAlign<hifloat8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(yAddr, vreg18, mask);
        } else {
            AscendC::Reg::Cast<int16_t, float, CAST_FP32_TO_INT16>(vreg9, vreg8, mask);
            AscendC::Reg::Cast<half, int16_t, CAST_INT16_TO_FP16>(vreg10, vreg9, mask);
            AscendC::Reg::Cast<int8_t, half, CAST_FP16_TO_INT8>(vreg11, vreg10, mask);
            AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::StoreDist::DIST_PACK4_B32>(yAddr, vreg11, mask);
        }
    }
}
} // namespace DequantSwigluQuantV35Ops
#endif
