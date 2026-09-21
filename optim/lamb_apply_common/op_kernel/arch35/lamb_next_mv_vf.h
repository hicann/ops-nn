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
 * \file lamb_next_mv_vf.h
 * \brief LambNextMV 的 regbase 计算段(13 入 4 出), 搬运/广播/分核见 lamb_apply_common/lamb_brc_kernel.h
 *
 * 与 golden 同运算序(先乘后加两步, 不用 FMA 融合形式):
 *   next_v = v*b2 + g2*(1-b2)                      -> y3
 *   next_m = m*b1 + g *(1-b1)                      -> y2
 *   v_unb  = next_v/rd1 ; m_unb = next_m/rd0
 *   y1 = param*wd + m_unb / sqrt(v_unb + eps)
 *   y4 = m_unb / (sqrt(v_unb) + eps)
 * 中间一律 fp32(A2 的 TBE compute dtype='float32'), 出口窄回 T。
 */

#ifndef LAMB_NEXT_MV_VF_H
#define LAMB_NEXT_MV_VF_H

#include "lamb_brc_kernel.h"

namespace LambNextMVOp {
using namespace AscendC;

// 输入下标(与原型入参顺序一致)
enum : uint32_t {
    IN_G2 = 0,    // input_mul3 = g^2
    IN_V = 1,     // input_mul2 = v
    IN_RD1 = 2,   // input_realdiv1 = 1 - b2^t
    IN_G = 3,     // input_mul1 = g
    IN_M = 4,     // input_mul0 = m
    IN_RD0 = 5,   // input_realdiv0 = 1 - b1^t
    IN_PARAM = 6, // input_mul4 = param
    IN_B1 = 7,
    IN_OMB1 = 8,
    IN_B2 = 9,
    IN_OMB2 = 10,
    IN_WD = 11,
    IN_EPS = 12
};

template <bool WITH_DECAY>
struct LambNextMVVf {
    template <typename T>
    static __aicore__ inline void Run(__local_mem__ T** in, __local_mem__ T** out, __local_mem__ float** scratch,
                                      uint32_t count)
    {
        uint16_t vfTimes = static_cast<uint16_t>(count / LambBrc::LAMB_VF_LEN);
        uint32_t tail = count % LambBrc::LAMB_VF_LEN;
        uint16_t tailTimes = (tail > 0) ? 1 : 0;

        __local_mem__ T* pG2 = in[IN_G2];
        __local_mem__ T* pV = in[IN_V];
        __local_mem__ T* pRd1 = in[IN_RD1];
        __local_mem__ T* pG = in[IN_G];
        __local_mem__ T* pM = in[IN_M];
        __local_mem__ T* pRd0 = in[IN_RD0];
        __local_mem__ T* pParam = in[IN_PARAM];
        __local_mem__ T* pB1 = in[IN_B1];
        __local_mem__ T* pOmB1 = in[IN_OMB1];
        __local_mem__ T* pB2 = in[IN_B2];
        __local_mem__ T* pOmB2 = in[IN_OMB2];
        __local_mem__ T* pWd = in[IN_WD];
        __local_mem__ T* pEps = in[IN_EPS];
        __local_mem__ T* y1 = out[0];
        __local_mem__ T* y2 = out[1];
        __local_mem__ T* y3 = out[2];
        __local_mem__ T* y4 = out[3];
        // 中间量以 fp32 暂存, 不经 T 的舍入 —— 与单段实现数值完全一致。
        __local_mem__ float* sNextV = scratch[0];
        __local_mem__ float* sNextM = scratch[1];

        __local_mem__ float* sVUnb = scratch[2];
        __local_mem__ float* sMUnb = scratch[3];

        // 分 4 段: 全量精确档(0ULP/FTZ_FALSE)展开后指令很多, 单段的循环回边偏移会超出
        // scbzi 立即数范围 [-512,511](--cce-long-scbz=true 压不住)。每段最多 2 个精确档调用,
        // 中间量一律以 fp32 暂存传递, 不经 T 的舍入, 与单段实现数值完全一致。

        // ---- 段1: next_v(y3) / next_m(y2) ----
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vA, vB, vNextV, vNextM;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
            for (uint16_t vfIdx = 0; vfIdx < vfTimes + tailTimes; vfIdx++) {
                uint32_t off = vfIdx * LambBrc::LAMB_VF_LEN;
                Reg::MaskReg preg = (vfIdx < vfTimes) ? maskAll : maskT;
                LambBrc::Load<T>(pV, vA, preg, off);
                LambBrc::Load<T>(pB2, vB, preg, off);
                Reg::Mul(vNextV, vA, vB, preg);
                LambBrc::Load<T>(pG2, vA, preg, off);
                LambBrc::Load<T>(pOmB2, vB, preg, off);
                Reg::Mul(vA, vA, vB, preg);
                Reg::Add(vNextV, vNextV, vA, preg);
                LambBrc::Store<T>(y3, vNextV, preg, off);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sNextV + off, vNextV, preg);
                LambBrc::Load<T>(pM, vA, preg, off);
                LambBrc::Load<T>(pB1, vB, preg, off);
                Reg::Mul(vNextM, vA, vB, preg);
                LambBrc::Load<T>(pG, vA, preg, off);
                LambBrc::Load<T>(pOmB1, vB, preg, off);
                Reg::Mul(vA, vA, vB, preg);
                Reg::Add(vNextM, vNextM, vA, preg);
                LambBrc::Store<T>(y2, vNextM, preg, off);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sNextM + off, vNextM, preg);
            }
        }

        // ---- 段2: 去偏 (2 次精确除法) ----
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vA, vT, vU;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
            for (uint16_t vfIdx = 0; vfIdx < vfTimes + tailTimes; vfIdx++) {
                uint32_t off = vfIdx * LambBrc::LAMB_VF_LEN;
                Reg::MaskReg preg = (vfIdx < vfTimes) ? maskAll : maskT;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vT, sNextV + off);
                LambBrc::Load<T>(pRd1, vA, preg, off);
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vU, vT, vA, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sVUnb + off, vU, preg);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vT, sNextM + off);
                LambBrc::Load<T>(pRd0, vA, preg, off);
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vU, vT, vA, preg);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sMUnb + off, vU, preg);
            }
        }

        // ---- 段3: y1 = param*wd + m_unb / sqrt(v_unb + eps) ----
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vA, vB, vPw, vVU, vMU;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
            for (uint16_t vfIdx = 0; vfIdx < vfTimes + tailTimes; vfIdx++) {
                uint32_t off = vfIdx * LambBrc::LAMB_VF_LEN;
                Reg::MaskReg preg = (vfIdx < vfTimes) ? maskAll : maskT;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vVU, sVUnb + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vMU, sMUnb + off);
                LambBrc::Load<T>(pParam, vA, preg, off);
                LambBrc::Load<T>(pWd, vB, preg, off);
                Reg::Mul(vPw, vA, vB, preg);
                LambBrc::Load<T>(pEps, vB, preg, off);
                Reg::Add(vA, vVU, vB, preg);
                Reg::Sqrt<float, &LambBrc::LAMB_PRECISE_SQRT>(vA, vA, preg);
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vA, vMU, vA, preg);
                Reg::Add(vA, vPw, vA, preg);
                LambBrc::Store<T>(y1, vA, preg, off);
            }
        }

        // ---- 段4: y4 = [param*wd +] m_unb / (sqrt(v_unb) + eps) ----
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vA, vB, vPw, vVU, vMU;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
            for (uint16_t vfIdx = 0; vfIdx < vfTimes + tailTimes; vfIdx++) {
                uint32_t off = vfIdx * LambBrc::LAMB_VF_LEN;
                Reg::MaskReg preg = (vfIdx < vfTimes) ? maskAll : maskT;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vVU, sVUnb + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vMU, sMUnb + off);
                Reg::Sqrt<float, &LambBrc::LAMB_PRECISE_SQRT>(vA, vVU, preg);
                LambBrc::Load<T>(pEps, vB, preg, off);
                Reg::Add(vA, vA, vB, preg);
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vA, vMU, vA, preg);
                if constexpr (WITH_DECAY) {
                    LambBrc::Load<T>(pParam, vPw, preg, off);
                    LambBrc::Load<T>(pWd, vB, preg, off);
                    Reg::Mul(vPw, vPw, vB, preg);
                    Reg::Add(vA, vPw, vA, preg);
                }
                LambBrc::Store<T>(y4, vA, preg, off);
            }
        }
    }
};
} // namespace LambNextMVOp
#endif // LAMB_NEXT_MV_VF_H
