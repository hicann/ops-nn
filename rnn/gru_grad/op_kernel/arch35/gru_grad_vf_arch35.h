/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_vf_arch35.h
 * \brief 门梯度链 VF 寄存器级实现（GateGradsVF/GateGradsScatterVF/GateGradsBandVF），类外自由函数。
 */
#ifndef _GRU_GRAD_VF_ARCH35_
#define _GRU_GRAD_VF_ARCH35_

// GateGradsVF 实现。寄存器调度（与 基线 UB 链逐操作对应）：
//   aVal = gradH·(1−z)（先算后双消费：dhn / dzp）；t1 = dhn = aVal·(1−n²)；
//   t2 = dzp = aVal·(hp−n)·z；t3 = drp = dhn·hn·r·(1−r)；aVal 复用持 dGh_n = r·dhn；
//   t4 = dhFromH = gradH·z。
// ubHN(hn→dGh_n) / ubHPrev(hp→dhFromH) 就地覆写：repeat 内先载后存、跨 repeat 地址带不相交
// （thnn HyTailVF ng→hy 同构，生产验证）。UpdateMask 消费 remaining（尾拍掩码，bn 同款）。
template <typename T>
__simd_vf__ inline void GateGradsVF(__ubuf__ T* drpAddr, __ubuf__ T* dzpAddr, __ubuf__ T* dhnAddr, __ubuf__ T* dghnAddr,
                                    __ubuf__ T* dhfhAddr, __ubuf__ T* gradHAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr,
                                    __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t count,
                                    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    AscendC::Reg::RegTensor<T> gradHReg, zReg, nReg, hpReg, hnReg, rReg;
    AscendC::Reg::RegTensor<T> oneReg, aVal, t1, t2, t3, t4;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::AddrReg aReg;
    uint32_t remaining = count;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        aReg = AscendC::Reg::CreateAddrReg<T>(i, oneRepeatSize);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(gradHReg, gradHAddr, aReg);
        AscendC::Reg::LoadAlign(zReg, zAddr, aReg);
        AscendC::Reg::LoadAlign(nReg, nAddr, aReg);
        AscendC::Reg::LoadAlign(hpReg, hpAddr, aReg);
        AscendC::Reg::LoadAlign(hnReg, hnAddr, aReg);
        AscendC::Reg::LoadAlign(rReg, rAddr, aReg);
        AscendC::Reg::Duplicate(oneReg, (T)1.0, mask);
        AscendC::Reg::Sub(aVal, oneReg, zReg, mask);   // 1−z
        AscendC::Reg::Mul(aVal, gradHReg, aVal, mask); // A = gradH·(1−z)
        AscendC::Reg::Mul(t1, nReg, nReg, mask);       // n²
        AscendC::Reg::Sub(t1, oneReg, t1, mask);       // 1−n²
        AscendC::Reg::Mul(t1, aVal, t1, mask);         // dhn = A·(1−n²)
        AscendC::Reg::Sub(t2, hpReg, nReg, mask);      // hp−n
        AscendC::Reg::Mul(t2, t2, zReg, mask);         // (hp−n)·z
        AscendC::Reg::Mul(t2, aVal, t2, mask);         // dzp = A·(hp−n)·z
        AscendC::Reg::Mul(t3, t1, hnReg, mask);        // dhn·hn
        AscendC::Reg::Mul(t3, t3, rReg, mask);         // ·r
        AscendC::Reg::Sub(aVal, oneReg, rReg, mask);   // 1−r（A 已消费）
        AscendC::Reg::Mul(t3, t3, aVal, mask);         // drp = dhn·hn·r·(1−r)
        AscendC::Reg::Mul(aVal, rReg, t1, mask);       // dGh_n = r·dhn
        AscendC::Reg::Mul(t4, gradHReg, zReg, mask);   // dhFromH = gradH·z
        AscendC::Reg::StoreAlign(drpAddr, t3, aReg, mask);
        AscendC::Reg::StoreAlign(dzpAddr, t2, aReg, mask);
        AscendC::Reg::StoreAlign(dhnAddr, t1, aReg, mask);
        AscendC::Reg::StoreAlign(dghnAddr, aVal, aReg, mask);
        AscendC::Reg::StoreAlign(dhfhAddr, t4, aReg, mask);
    }
}

// GateGradsScatterVF 实现。行外层 × 拍内层（chunks = hAligned/VL）：
//   输入/dhFromH 按线性偏移 row*hA + c*VL 寻址；staging 按行基 row*3hA + 门偏 g*hA + c*VL。
//   计算链与 GateGradsVF 逐操作一致（数值逐位等价）；drp/dzp 寄存器双消费双平面直写。
//   全满拍（hA%VL==0 由分派保证）→ CreateMask 常量全满掩码，无 UpdateMask 尾处理。
template <typename T>
__simd_vf__ inline void GateGradsScatterVF(__ubuf__ T* dGiAddr, __ubuf__ T* dGhAddr, __ubuf__ T* dhfhAddr,
                                           __ubuf__ T* hpSaveAddr, __ubuf__ T* gradHAddr, __ubuf__ T* zAddr,
                                           __ubuf__ T* nAddr, __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr,
                                           uint32_t bRows, uint32_t hAligned, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> gradHReg, zReg, nReg, hpReg, hnReg, rReg;
    AscendC::Reg::RegTensor<T> oneReg, aVal, t1, t2, t3, t4;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
    AscendC::Reg::Duplicate(oneReg, (T)1.0);
    const uint32_t chunks = hAligned / oneRepeatSize;
    for (uint32_t row = 0; row < bRows; ++row) {
        const uint32_t rowOff = row * hAligned;
        __ubuf__ T* dGiR = dGiAddr + row * GRU_GATE_SIZE * hAligned;
        __ubuf__ T* dGhR = dGhAddr + row * GRU_GATE_SIZE * hAligned;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t off = rowOff + c * oneRepeatSize;
            const uint32_t cOff = c * oneRepeatSize;
            AscendC::Reg::LoadAlign(gradHReg, gradHAddr + off);
            AscendC::Reg::LoadAlign(zReg, zAddr + off);
            AscendC::Reg::LoadAlign(nReg, nAddr + off);
            AscendC::Reg::LoadAlign(hpReg, hpAddr + off);
            AscendC::Reg::LoadAlign(hnReg, hnAddr + off);
            AscendC::Reg::LoadAlign(rReg, rAddr + off);
            AscendC::Reg::Sub(aVal, oneReg, zReg, mask);                                  // 1−z
            AscendC::Reg::Mul(aVal, gradHReg, aVal, mask);                                // A = gradH·(1−z)
            AscendC::Reg::Mul(t1, nReg, nReg, mask);                                      // n²
            AscendC::Reg::Sub(t1, oneReg, t1, mask);                                      // 1−n²
            AscendC::Reg::Mul(t1, aVal, t1, mask);                                        // dhn = A·(1−n²)
            AscendC::Reg::Sub(t2, hpReg, nReg, mask);                                     // hp−n
            AscendC::Reg::Mul(t2, t2, zReg, mask);                                        // (hp−n)·z
            AscendC::Reg::Mul(t2, aVal, t2, mask);                                        // dzp = A·(hp−n)·z
            AscendC::Reg::Mul(t3, t1, hnReg, mask);                                       // dhn·hn
            AscendC::Reg::Mul(t3, t3, rReg, mask);                                        // ·r
            AscendC::Reg::Sub(aVal, oneReg, rReg, mask);                                  // 1−r（A 已消费）
            AscendC::Reg::Mul(t3, t3, aVal, mask);                                        // drp = dhn·hn·r·(1−r)
            AscendC::Reg::Mul(aVal, rReg, t1, mask);                                      // dGh_n = r·dhn
            AscendC::Reg::Mul(t4, gradHReg, zReg, mask);                                  // dhFromH = gradH·z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_RESET * hAligned + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_UPDATE * hAligned + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_NEW * hAligned + cOff, t1, mask);    // 门 n
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_RESET * hAligned + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_UPDATE * hAligned + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_NEW * hAligned + cOff, aVal, mask);  // 门 n
            AscendC::Reg::StoreAlign(dhfhAddr + off, t4, mask);
            AscendC::Reg::StoreAlign(hpSaveAddr + off, hpReg, mask); // hp 保留（Tail 免 GM 重读）
        }
    }
}

// GateGradsBandVF 实现。地址口径：
//   全宽输入/输出（dy/dhfhIn/z/n/hp/hn/r/dhfhOut/hpSave，行距 hA）: off = row*hA + bandOff + cOff
//   gemmRes 带紧凑（行距 hAb）: off = row*hAb + cOff
//   staging 全宽（行距 3hA）: 列区 [g*hA + bandOff + cOff)
// 折叠序与现行 Accumulate+LoadGradH 逐位一致：((gemmRes+dhFromH)+dy)，IEEE 加法交换律保证
// 加数对调无损。band0 与 band1 列区/hp 带列均不相交，跨带无 WAR（gemmRes 的 ubTmp3 复用由
// 带间 SyncVtoM2 覆盖——见 VecProcessBand）。
template <typename T>
__simd_vf__ inline void GateGradsBandVF(__ubuf__ T* dGiAddr, __ubuf__ T* dGhAddr, __ubuf__ T* dhfhOutAddr,
                                        __ubuf__ T* hpSaveAddr, __ubuf__ T* gemmResAddr, __ubuf__ T* dyAddr,
                                        __ubuf__ T* dhfhInAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr,
                                        __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t bRows,
                                        uint32_t hA, uint32_t hAb, uint32_t bandOff, uint32_t oneRepeatSize)
{
    AscendC::Reg::RegTensor<T> gradHReg, zReg, nReg, hpReg, hnReg, rReg;
    AscendC::Reg::RegTensor<T> gResReg, dyReg, dhInReg;
    AscendC::Reg::RegTensor<T> oneReg, aVal, t1, t2, t3, t4;
    AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<T>();
    AscendC::Reg::Duplicate(oneReg, (T)1.0);
    const uint32_t chunks = hAb / oneRepeatSize;
    for (uint32_t row = 0; row < bRows; ++row) {
        const uint32_t offFull = row * hA + bandOff;
        const uint32_t offComp = row * hAb;
        __ubuf__ T* dGiR = dGiAddr + row * GRU_GATE_SIZE * hA;
        __ubuf__ T* dGhR = dGhAddr + row * GRU_GATE_SIZE * hA;
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t cOff = c * oneRepeatSize;
            AscendC::Reg::LoadAlign(gResReg, gemmResAddr + offComp + cOff); // dhPrevF32(t+1) 带
            AscendC::Reg::LoadAlign(dyReg, dyAddr + offFull + cOff);        // dy(t) 带
            AscendC::Reg::LoadAlign(dhInReg, dhfhInAddr + offFull + cOff);  // dhFromH(t+1) 带
            AscendC::Reg::Add(gradHReg, gResReg, dhInReg, mask);            // gemmRes + dhFromH
            AscendC::Reg::Add(gradHReg, gradHReg, dyReg, mask);             // + dy = gradH(t)
            AscendC::Reg::LoadAlign(zReg, zAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(nReg, nAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(hpReg, hpAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(hnReg, hnAddr + offFull + cOff);
            AscendC::Reg::LoadAlign(rReg, rAddr + offFull + cOff);
            AscendC::Reg::Sub(aVal, oneReg, zReg, mask);   // 1−z
            AscendC::Reg::Mul(aVal, gradHReg, aVal, mask); // A = gradH·(1−z)
            AscendC::Reg::Mul(t1, nReg, nReg, mask);       // n²
            AscendC::Reg::Sub(t1, oneReg, t1, mask);       // 1−n²
            AscendC::Reg::Mul(t1, aVal, t1, mask);         // dhn = A·(1−n²)
            AscendC::Reg::Sub(t2, hpReg, nReg, mask);      // hp−n
            AscendC::Reg::Mul(t2, t2, zReg, mask);         // (hp−n)·z
            AscendC::Reg::Mul(t2, aVal, t2, mask);         // dzp = A·(hp−n)·z
            AscendC::Reg::Mul(t3, t1, hnReg, mask);        // dhn·hn
            AscendC::Reg::Mul(t3, t3, rReg, mask);         // ·r
            AscendC::Reg::Sub(aVal, oneReg, rReg, mask);   // 1−r（A 已消费）
            AscendC::Reg::Mul(t3, t3, aVal, mask);         // drp = dhn·hn·r·(1−r)
            AscendC::Reg::Mul(aVal, rReg, t1, mask);       // dGh_n = r·dhn
            AscendC::Reg::Mul(t4, gradHReg, zReg, mask);   // dhFromH(t) = gradH·z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_RESET * hA + bandOff + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_UPDATE * hA + bandOff + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGiR + GATE_IDX_NEW * hA + bandOff + cOff, t1, mask);    // 门 n
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_RESET * hA + bandOff + cOff, t3, mask);  // 门 r
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_UPDATE * hA + bandOff + cOff, t2, mask); // 门 z
            AscendC::Reg::StoreAlign(dGhR + GATE_IDX_NEW * hA + bandOff + cOff, aVal, mask);  // 门 n
            AscendC::Reg::StoreAlign(dhfhOutAddr + offFull + cOff, t4, mask);
            AscendC::Reg::StoreAlign(hpSaveAddr + offFull + cOff, hpReg, mask);
        }
    }
}

#endif // _GRU_GRAD_VF_ARCH35_
