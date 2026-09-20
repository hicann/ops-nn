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
 * \file lamb_apply_optimizer_assign_vf.h
 * \brief LambApplyOptimizerAssign 的 regbase 计算段(12 入 3 出)
 *
 * 与 golden 同运算序:
 *   next_v  = v*b2 + (g*g)*(1-b2)                 -> out1
 *   next_m  = m*b1 + g*(1-b1)                     -> out2
 *   b_corr  = 1 + (-1)*exp(ln(b)*steps)   (Log/Mul/Exp/Muls/Adds 五步, 不用幂运算:
 *             b 接近 1 时 ln(b) 有相消损失, 直接幂得不到同一中间量)
 *   update  = (next_m/b1corr)/(sqrt(next_v/b2corr)+eps) + param*wd*do_use  -> out0
 */

#ifndef LAMB_APPLY_OPTIMIZER_ASSIGN_VF_H
#define LAMB_APPLY_OPTIMIZER_ASSIGN_VF_H

#include "../lamb_apply_common/arch35/lamb_brc_kernel.h"

namespace LambApplyOptimizerAssignOp {
using namespace AscendC;

enum : uint32_t {
    IN_GRAD = 0,
    IN_V = 1,
    IN_M = 2,
    IN_PARAM = 3,
    IN_B1 = 4,
    IN_OMB1 = 5,
    IN_B2 = 6,
    IN_OMB2 = 7,
    IN_EPS = 8,
    IN_STEPS = 9,
    IN_DOUSE = 10,
    IN_WD = 11
};

struct LambApplyOptimizerAssignVf {
    template <typename T>
    static __aicore__ inline void Run(__local_mem__ T** in, __local_mem__ T** out, __local_mem__ float** scratch,
                                      uint32_t count)
    {
        uint16_t vfTimes = static_cast<uint16_t>(count / LambBrc::LAMB_VF_LEN);
        uint32_t tail = count % LambBrc::LAMB_VF_LEN;
        uint16_t tailTimes = (tail > 0) ? 1 : 0;

        __local_mem__ T* pG = in[IN_GRAD];
        __local_mem__ T* pV = in[IN_V];
        __local_mem__ T* pM = in[IN_M];
        __local_mem__ T* pParam = in[IN_PARAM];
        __local_mem__ T* pB1 = in[IN_B1];
        __local_mem__ T* pOmB1 = in[IN_OMB1];
        __local_mem__ T* pB2 = in[IN_B2];
        __local_mem__ T* pOmB2 = in[IN_OMB2];
        __local_mem__ T* pEps = in[IN_EPS];
        __local_mem__ T* pSteps = in[IN_STEPS];
        __local_mem__ T* pDoUse = in[IN_DOUSE];
        __local_mem__ T* pWd = in[IN_WD];
        __local_mem__ T* update = out[0];
        __local_mem__ T* nextVOut = out[1];
        __local_mem__ T* nextMOut = out[2];
        // 中间量以 fp32 暂存, 不经 T 的舍入 —— 与单段实现数值完全一致。
        __local_mem__ float* sNextV = scratch[0];
        __local_mem__ float* sNextM = scratch[1];

        // ---- 段1: next_v / next_m ----
        // 拆成两段是因为全量精确档(0ULP/FTZ_FALSE)展开后单段指令过多, 循环回边偏移超出
        // scbzi 立即数范围 [-512,511]。拆段后每段都在范围内, 且不必降低任何一处的精度档。
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vA, vB, vC, vNextV, vNextM;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
            for (uint16_t vfIdx = 0; vfIdx < vfTimes + tailTimes; vfIdx++) {
                uint32_t off = vfIdx * LambBrc::LAMB_VF_LEN;
                Reg::MaskReg preg = (vfIdx < vfTimes) ? maskAll : maskT;
                // next_v = v*b2 + (g*g)*(1-b2)
                LambBrc::Load<T>(pV, vA, preg, off);
                LambBrc::Load<T>(pB2, vB, preg, off);
                Reg::Mul(vNextV, vA, vB, preg);
                LambBrc::Load<T>(pG, vA, preg, off);
                Reg::Mul(vC, vA, vA, preg);
                LambBrc::Load<T>(pOmB2, vB, preg, off);
                Reg::Mul(vC, vC, vB, preg);
                Reg::Add(vNextV, vNextV, vC, preg);
                LambBrc::Store<T>(nextVOut, vNextV, preg, off);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sNextV + off, vNextV, preg);
                // next_m = m*b1 + g*(1-b1)   (vA 仍是 g)
                LambBrc::Load<T>(pOmB1, vB, preg, off);
                Reg::Mul(vC, vA, vB, preg);
                LambBrc::Load<T>(pM, vA, preg, off);
                LambBrc::Load<T>(pB1, vB, preg, off);
                Reg::Mul(vNextM, vA, vB, preg);
                Reg::Add(vNextM, vNextM, vC, preg);
                LambBrc::Store<T>(nextMOut, vNextM, preg, off);
                Reg::StoreAlign<float, Reg::StoreDist::DIST_NORM>(sNextM + off, vNextM, preg);
            }
        }

        // ---- 段2: 偏差校正 + update ----
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vA, vB, vC, vNextV, vNextM, vB1c, vB2c, vT1, vT2;
            Reg::MaskReg cmpExpm1;
            Reg::MaskReg maskAll = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg maskT = Reg::UpdateMask<float>(tail);
            for (uint16_t vfIdx = 0; vfIdx < vfTimes + tailTimes; vfIdx++) {
                uint32_t off = vfIdx * LambBrc::LAMB_VF_LEN;
                Reg::MaskReg preg = (vfIdx < vfTimes) ? maskAll : maskT;
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vNextV, sNextV + off);
                Reg::LoadAlign<float, Reg::LoadDist::DIST_NORM>(vNextM, sNextM + off);
                // Ln/Exp 取 1ULP + FTZ_TRUE: 默认 INTRINSIC 是硬件近似档, 而 b_corr 的抵消会把 exp 的误差
                // 放大 1/(1-exp) 倍; 取 FTZ_TRUE 是因为 ln(b)*steps 低于 exp 下溢边界时真值本就应为 0。
                // 1 - b^steps 由 OneMinusExp 计算(见 lamb_brc_kernel.h)。
                // b1corr = 1 - exp(ln(b1)*steps)
                LambBrc::Load<T>(pSteps, vC, preg, off);
                LambBrc::Load<T>(pB1, vB, preg, off);
                Ln<float, &LambBrc::LAMB_PRECISE_LN_FTZT>(vB1c, vB, preg);
                Reg::Mul(vB1c, vB1c, vC, preg);
                LambBrc::OneMinusExp(vB1c, vB1c, vT1, vT2, cmpExpm1, preg);
                LambBrc::Load<T>(pB2, vB, preg, off);
                Ln<float, &LambBrc::LAMB_PRECISE_LN_FTZT>(vB2c, vB, preg);
                Reg::Mul(vB2c, vB2c, vC, preg);
                LambBrc::OneMinusExp(vB2c, vB2c, vT1, vT2, cmpExpm1, preg);
                // update = (next_m/b1corr)/(sqrt(next_v/b2corr)+eps) + param*wd*do_use
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vA, vNextM, vB1c, preg);
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vB, vNextV, vB2c, preg);
                Reg::Sqrt<float, &LambBrc::LAMB_PRECISE_SQRT>(vB, vB, preg);
                LambBrc::Load<T>(pEps, vC, preg, off);
                Reg::Add(vB, vB, vC, preg);
                Reg::Div<float, &LambBrc::LAMB_PRECISE_DIV>(vA, vA, vB, preg);
                LambBrc::Load<T>(pParam, vB, preg, off);
                LambBrc::Load<T>(pWd, vC, preg, off);
                Reg::Mul(vB, vB, vC, preg);
                LambBrc::Load<T>(pDoUse, vC, preg, off);
                Reg::Mul(vB, vB, vC, preg);
                Reg::Add(vA, vA, vB, preg);
                LambBrc::Store<T>(update, vA, preg, off);
            }
        }
    }
};
} // namespace LambApplyOptimizerAssignOp
#endif // LAMB_APPLY_OPTIMIZER_ASSIGN_VF_H
