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
 * \file arg_max_grad_d_nd.h
 * \brief ArgMaxGradD arch35 内核: 沿 dimension 轴按 "序号是否等于 indices" 做条件选择
 *
 * 布局归一 (outer, D, inner): var/assist/y 为 (outer, D, inner), indices/updates 为 (outer, 1, inner)。
 *   y[o,k,i] = (assist[o,k,i] == indices[o,i]) ? updates[o,i] : var[o,k,i]
 * INNER_IS_ONE=false: 任务=一行(o,k), 沿 inner 向量化, indices/updates 是等长向量。
 * INNER_IS_ONE=true : 任务=一个 o, 沿被选轴 D 向量化(此时 var 在该方向连续), indices/updates 退化为标量。
 */

#ifndef ARG_MAX_GRAD_D_VF_H
#define ARG_MAX_GRAD_D_VF_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace ArgMaxGradD {
using namespace AscendC;

using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UpdateMask;

// fp16/bf16 <-> fp32 的随路 cast trait(与 activation/clipped_swiglu 同款)
constexpr static AscendC::Reg::CastTrait CAST_B16_TO_FP32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};
constexpr static AscendC::Reg::CastTrait CAST_FP32_TO_B16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr static int64_t B16_BYTES = 2;

// bfloat16_t 没有到 float 的隐式转换, 必须走 ToFloat
template <typename T>
__aicore__ inline float ScalarToFloat(const T& v)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        return ToFloat(v);
    } else {
        return static_cast<float>(v);
    }
}

// int8_t 不在 arch35 Select 的支持类型里(half/int16/uint16/int32/uint32/float/bfloat16),
// 借 half 中转: |int8| <= 127 在 half 上可精确表示, 转一来一回无损。
template <typename T>
struct SelectTypeTrait {
    using Type = T;
};
template <>
struct SelectTypeTrait<int8_t> {
    using Type = half;
};

// DataCopyPad 的 paddingValue 需要一个 T 类型的零值
template <typename T>
__aicore__ inline T ScalarZero()
{
    return static_cast<T>(0);
}

// 三种车道宽度各自的"选一次"实现, 拆出来是为了让 VF 主体保持短小(CodeCheck 函数长度阈值 50 行)。
// 从 __simd_vf__ 里调用的被调函数必须标 __simd_callee__。
template <typename T, bool IDX_IS_SCALAR>
__simd_callee__ inline void SelectLane32(__ubuf__ T* outAddr, __ubuf__ T* varAddr, __ubuf__ T* updAddr, T updScalar,
                                         AscendC::Reg::MaskReg& hit, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> varReg;
    AscendC::Reg::RegTensor<T> updReg;
    AscendC::Reg::RegTensor<T> outReg;
    AscendC::Reg::LoadAlign(varReg, varAddr);
    if constexpr (IDX_IS_SCALAR) {
        AscendC::Reg::Duplicate(updReg, updScalar);
    } else {
        AscendC::Reg::LoadAlign(updReg, updAddr);
    }
    AscendC::Reg::Select<T>(outReg, updReg, varReg, hit);
    AscendC::Reg::StoreAlign(outAddr, outReg, mask);
}

// fp16/bf16: 拆包到 fp32 域选, 选完打包回去(纯搬运, 拆/打包互逆, 逐位无损)
template <typename T, bool IDX_IS_SCALAR>
__simd_callee__ inline void SelectLaneB16(__ubuf__ T* outAddr, __ubuf__ T* varAddr, __ubuf__ T* updAddr,
                                          float updScalarF, AscendC::Reg::MaskReg& hit, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> rawVar;
    AscendC::Reg::RegTensor<T> rawUpd;
    AscendC::Reg::RegTensor<T> rawOut;
    AscendC::Reg::RegTensor<float> varF;
    AscendC::Reg::RegTensor<float> updF;
    AscendC::Reg::RegTensor<float> outF;
    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(rawVar, varAddr);
    AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(varF, rawVar, mask);
    if constexpr (IDX_IS_SCALAR) {
        AscendC::Reg::Duplicate(updF, updScalarF);
    } else {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(rawUpd, updAddr);
        AscendC::Reg::Cast<float, T, CAST_B16_TO_FP32>(updF, rawUpd, mask);
    }
    AscendC::Reg::Select<float>(outF, updF, varF, hit);
    AscendC::Reg::Cast<T, float, CAST_FP32_TO_B16>(rawOut, outF, mask);
    AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(outAddr, rawOut, mask);
}

// int8: 拆到 32bit 车道按 int32 选, 再打包回 b8
template <typename T, bool IDX_IS_SCALAR>
__simd_callee__ inline void SelectLaneB8(__ubuf__ T* outAddr, __ubuf__ T* varAddr, __ubuf__ T* updAddr, T updScalar,
                                         AscendC::Reg::MaskReg& hit, AscendC::Reg::MaskReg& mask)
{
    AscendC::Reg::RegTensor<T> rawVar;
    AscendC::Reg::RegTensor<T> rawUpd;
    AscendC::Reg::RegTensor<T> rawOut;
    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(rawVar, varAddr);
    if constexpr (IDX_IS_SCALAR) {
        AscendC::Reg::Duplicate(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(rawUpd),
                                static_cast<int32_t>(updScalar));
    } else {
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK4_B8>(rawUpd, updAddr);
    }
    AscendC::Reg::Select<int32_t>(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(rawOut),
                                  reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(rawUpd),
                                  reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(rawVar), hit);
    AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK4_B32>(outAddr, rawOut, mask);
}

// 非对齐行长的 tile 铺设: 把 base[0, copyElems) 追加到 base[curElems, ...)。
// 行长不是 32B 整数倍时目的偏移必然非对齐, UB→UB 的 DataCopy 用不了, 改用非对齐流式写
// (vstus: StoreUnAlign 累积 + StoreUnAlignPost 收尾), 范式照 activation/log_softmax_grad。
// 配合倍增调用, log2(rows) 次即可铺满, 且每次都是满车道搬运。
template <typename T>
__simd_vf__ inline void TileAppendUnalignVF(__ubuf__ T* base, uint32_t curElems, uint32_t copyElems, uint32_t lane,
                                            uint16_t loops)
{
    AscendC::Reg::UnalignRegForStore ureg;
    __ubuf__ T* dst = base + curElems;
    AscendC::Reg::MaskReg mask;
    uint32_t remain = copyElems;
    for (uint16_t t = 0; t < loops; ++t) {
        uint32_t before = remain;
        mask = AscendC::Reg::UpdateMask<T>(remain);
        uint32_t step = before - remain; // 本轮实际元素数(尾轮不足一个 VL)
        AscendC::Reg::RegTensor<T> reg;
        AscendC::Reg::LoadAlign(reg, base + t * lane);
        AscendC::Reg::StoreUnAlign(dst, reg, ureg, step);
    }
    AscendC::Reg::StoreUnAlignPost<T>(dst, ureg, 0);
}

// ── regbase VF: mask = (assist == idx), out = mask ? updates : var ──────────────
// 掩码全程留在 MaskReg 里(不落 UB), 比较恒在 int32 的 64 车道上做; T 宽度不等于 4 字节时
// 随路拆包到 32bit 车道、选完再打包回去。范式照 loss/chamfer_distance 的 ChamferDistVF。
// 写法要点: 必须是 __simd_vf__ 自由函数 + 裸 __ubuf__ 指针推进; 塞进成员函数会触发后端
// "Unsupported Inst must be hoisted"。
template <typename T, bool IDX_IS_SCALAR>
__simd_vf__ inline void ArgMaxGradSelectVF(__ubuf__ T* outAddr, __ubuf__ T* varAddr, __ubuf__ T* updAddr,
                                           __ubuf__ int32_t* assistAddr, __ubuf__ int32_t* idxAddr, int32_t idxScalar,
                                           T updScalar, float updScalarF, uint32_t count, uint32_t lane,
                                           uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<int32_t> assistReg;
    AscendC::Reg::RegTensor<int32_t> idxReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::MaskReg hit;
    uint32_t remain = count; // UpdateMask 每轮自减, 不用外部算余量

    if constexpr (IDX_IS_SCALAR) {
        AscendC::Reg::Duplicate(idxReg, idxScalar);
    }
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::Reg::UpdateMask<int32_t>(remain);
        AscendC::Reg::LoadAlign(assistReg, assistAddr + i * lane);
        if constexpr (!IDX_IS_SCALAR) {
            AscendC::Reg::LoadAlign(idxReg, idxAddr + i * lane);
        }
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::EQ>(hit, assistReg, idxReg, mask);

        if constexpr (IsSameType<T, float>::value || IsSameType<T, int32_t>::value) {
            SelectLane32<T, IDX_IS_SCALAR>(outAddr + i * lane, varAddr + i * lane,
                                           IDX_IS_SCALAR ? updAddr : updAddr + i * lane, updScalar, hit, mask);
        } else if constexpr (sizeof(T) == B16_BYTES) {
            SelectLaneB16<T, IDX_IS_SCALAR>(outAddr + i * lane, varAddr + i * lane,
                                            IDX_IS_SCALAR ? updAddr : updAddr + i * lane, updScalarF, hit, mask);
        } else {
            SelectLaneB8<T, IDX_IS_SCALAR>(outAddr + i * lane, varAddr + i * lane,
                                           IDX_IS_SCALAR ? updAddr : updAddr + i * lane, updScalar, hit, mask);
        }
    }
}

} // namespace ArgMaxGradD

#endif // ARG_MAX_GRAD_D_VF_H
