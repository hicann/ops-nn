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
 * \file non_zero_with_value_general.h
 * \brief NonZeroWithValue 通用路径(唯一非空计算路径)—— 2D 坐标主序 + value 通道。TilingKey 1020。
 *        多核 + 按 tile 循环,任意 numel 均可;输出遍重算 mask(不依赖存下的 mask),
 *        故单核小张量(免同步)与超大张量(mask 存不下)都由本路径覆盖,无需 full_load/big_mask 分路。
 *        输出遍:Arange 线性 idx → GatherMask 压缩 idx + 同掩码压缩 value →
 *                线性 idx 拆 (row=idx/col, col=idx%col) → 坐标主序 [2,numel] 写 index + 写 value。
 */
#ifndef NON_ZERO_WITH_VALUE_GENERAL_H
#define NON_ZERO_WITH_VALUE_GENERAL_H

#include "non_zero_with_value_base.h"

namespace NonZeroWithValue {
using namespace AscendC;

template <typename TValue, typename TIndex>
class NonZeroWithValueGeneral : public NonZeroWithValueBase<TValue, TIndex> {
public:
    __aicore__ inline NonZeroWithValueGeneral() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR index, GM_ADDR count, GM_ADDR workspace,
                                const NonZeroWithValueTilingData* tilingData)
    {
        this->InitBase(x, value, index, count, workspace, tilingData);
        // idx buffer 放 3 份 int32 平面(压缩线性 idx + row + col),**恒 int32**、与 value dtype 无关:
        // 必须按 int32 分配(不是 xInputSize——那是 value-dtype 字节数,对 sub-32bit 会小 4×,idx 平面越写越界
        // 踩进 valOutBuf_,dense+多tile 时用 int32 步长的 col 覆盖 int8 value → 每 4 个坏 3 个)。与 tiling
        // 反推的 VALUE_CH_NUM*INT32_BYTES 记账一致;+V32 补 alignNum 向上取整裕量(int64 的 ubFactor 非 V32 整数倍)。
        this->pipe_.InitBuffer(idxOutBuf_,
                               (tilingData->ubFactorNum + (int64_t)NZV_VF_LEN_INT32) * (int64_t)sizeof(int32_t) * 3);
        this->pipe_.InitBuffer(valOutBuf_, tilingData->valueBufSize); // 压缩后 value
    }

    __aicore__ inline void Process()
    {
        LocalTensor<uint32_t> addUb = this->addUb_.template Get<uint32_t>();
        CleanAddUb(addUb);

        bool isTail = (this->blockIdx_ == this->tilingData_->realCoreNum - 1);
        int32_t loopNum = isTail ? this->tilingData_->loopNumTailCore : this->tilingData_->loopNumPerCore;
        int32_t tailNum = isTail ? this->tilingData_->loopTailTailCore : this->tilingData_->loopTailPerCore;
        int32_t beforeNum = this->tilingData_->ubFactorNum;

        // 计数遍(找非零 + 跨核前缀和 + 写 count)
        this->ComputeCount(addUb, loopNum, beforeNum, tailNum);

        // 输出遍(逐 tile 压缩坐标+value,坐标主序写出)
        int32_t loopNumO = isTail ? this->tilingData_->loopNumTo : this->tilingData_->loopNumO;
        int32_t loopTailO = isTail ? this->tilingData_->loopTailTo : this->tilingData_->loopTailO;
        int32_t beforeNumO = this->tilingData_->beforeNumO;
        uint64_t loopGm = 0;
        for (int32_t loop = 0; loop < loopNumO; loop++) {
            OutputTile(loop, beforeNumO, loopGm);
        }
        if (loopTailO != 0) {
            OutputTile(loopNumO, loopTailO, loopGm);
        }
    }

private:
    static constexpr int32_t NZV_DIM8L = 8;
    TBuf<QuePosition::VECCALC> idxOutBuf_;
    TBuf<QuePosition::VECCALC> valOutBuf_;

    __aicore__ inline void CleanAddUb(LocalTensor<uint32_t>& addUb)
    {
        auto ptr = (__ubuf__ uint32_t*)addUb.GetPhyAddr();
        uint32_t mask = NZV_VF_LEN_INT32;
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> zeroReg;
            MaskReg mreg = UpdateMask<uint32_t>(mask);
            Duplicate(zeroReg, (uint32_t)0);
            DataCopy(ptr, zeroReg, mreg);
        }
    }

    // 处理一个输出 tile:压缩线性 idx + value,拆坐标,坐标主序写。
    __aicore__ inline void OutputTile(int32_t loop, int32_t num, uint64_t& loopGm)
    {
        // 1) 载入 x tile
        LocalTensor<TValue> xUb = this->inQueX_.template AllocTensor<TValue>();
        this->copyParams_.blockCount = 1;
        this->copyParams_.blockLen = num * sizeof(TValue);
        this->copyParams_.srcStride = 0;
        this->copyParams_.dstStride = 0;
        int64_t gmBase = this->blockIdx_ * this->tilingData_->numPerCore + loop * this->tilingData_->beforeNumO;
        DataCopyPad(xUb, this->xGm_[gmBase], this->copyParams_, this->padParams_);
        this->inQueX_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(xUb);
        LocalTensor<TValue> xUbC = this->inQueX_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, TValue>();

        LocalTensor<int32_t> idxUb = idxOutBuf_.template Get<int32_t>();
        LocalTensor<TValue> valUb = valOutBuf_.template Get<TValue>();
        uint64_t arNum = 0;
        // 2) 压缩:线性 idx(Arange+GatherMask) + value(同掩码 GatherMask)
        CompactIdxAndValue(loop, num, gmBase, xUbC, idxUb, valUb, arNum);
        this->inQueX_.FreeTensor(xUbC);
        if (arNum == 0) {
            return;
        }
        loopGm += arNum;

        // 3) 线性 idx 拆 (row, col),坐标主序写 index;value 写 value
        WriteCoordMajorAndValue(idxUb, valUb, arNum, loopGm);
    }

    // Arange 线性位置 → GatherMask(mask) 压缩线性 idx;同掩码 GatherMask x 值压缩 value。
    // 输出遍:压缩 idx(scope1,推进 AR → arNum)+ 压缩 value(scope2)。两个 __VEC_SCOPE__ 各拆一个辅助函数
    // (硬件 unalign 累加器单例,每 scope 只做一路 GatherMask + DataCopyUnAlign;拆函数不跨 __VEC_SCOPE__,VF
    // 融合不受影响)。
    __aicore__ inline void CompactIdxAndValue(int32_t loop, int32_t num, int64_t gmBase, LocalTensor<TValue>& xUb,
                                              LocalTensor<int32_t>& idxUb, LocalTensor<TValue>& valUb, uint64_t& arNum)
    {
        (void)loop;
        CompactIdx(num, gmBase, xUb, idxUb, arNum);
        // 8 字节 value 复用 CompactIdx 已压缩的连续线性位置(idxUb[0:arNum]);≤4 字节忽略这几个参数。
        CompactValue(num, gmBase, arNum, idxUb, xUb, valUb);
    }

    // sub-32bit 索引压缩:一个 VF 寄存器装 4×/2× 于 int32 的元素,int32 idsReg 一次只能放 V32 个索引。
    // 重算 int8/int16 mask 后用 MaskUnPack 拆成 4/2 个 int32 宽度 mask,逐块 Arange(推进 V32)+ GatherMask
    // 生成压缩索引(照搬母算子 non_zero VfComputeIds/VfPerCoreNonZeroNum 的宽度膨胀处理)。4 字节不走此函数。
    __aicore__ inline void CompactIdxSub32(int32_t num, int64_t gmBase, LocalTensor<TValue>& xUb,
                                           LocalTensor<int32_t>& idxUb, uint64_t& arNum)
    {
        auto xPtr = (__ubuf__ TValue*)xUb.GetPhyAddr();
        auto idxPtr = (__ubuf__ int32_t*)idxUb.GetPhyAddr();
        uint16_t repeatElm = (uint16_t)this->vfLenV_;
        uint16_t repeatTimes = CeilDivision(num, repeatElm);
        int32_t scalar = static_cast<int32_t>(gmBase);
        uint32_t sreg1 = (uint32_t)num;
        int32_t vfLenInt32 = (int32_t)NZV_VF_LEN_INT32;
        __VEC_SCOPE__
        {
            AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
            AscendC::Reg::UnalignReg uregIdx;
            RegTensor<int32_t> idsReg;
            RegTensor<int32_t> sqzIdx;
            RegTensor<TValue> xReg;
            RegTensor<TValue> zeroXReg;
            MaskReg cmpReg;
            MaskReg preg;
            MaskReg mL, mH, m0, m1, m2, m3;
            preg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
            Duplicate(zeroXReg, (TValue)0);
            Arange(idsReg, scalar);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                MaskReg pnum = UpdateMask<TValue>(sreg1);
                AscendC::Reg::AddrReg xOffset = AscendC::Reg::CreateAddrReg<TValue>(i, repeatElm);
                DataCopy(xReg, xPtr, xOffset);
                Select(xReg, xReg, zeroXReg, pnum);
                CompareScalar<TValue, CMPMODE::NE>(cmpReg, xReg, (TValue)0, pnum);
                using GM = AscendC::Reg::GatherMaskMode;
                using HL = AscendC::Reg::HighLowPart;
                using PL = AscendC::Reg::PostLiteral;
                if constexpr (sizeof(TValue) == 1) {
                    AscendC::Reg::MaskUnPack<HL::LOWEST>(mL, cmpReg);
                    AscendC::Reg::MaskUnPack<HL::HIGHEST>(mH, cmpReg);
                    AscendC::Reg::MaskUnPack<HL::LOWEST>(m0, mL);
                    AscendC::Reg::MaskUnPack<HL::HIGHEST>(m1, mL);
                    AscendC::Reg::MaskUnPack<HL::LOWEST>(m2, mH);
                    AscendC::Reg::MaskUnPack<HL::HIGHEST>(m3, mH);
                    AscendC::Reg::GatherMask<int32_t, GM::STORE_REG>(sqzIdx, idsReg, m0);
                    AscendC::Reg::DataCopyUnAlign<int32_t, PL::POST_MODE_UPDATE>(idxPtr, sqzIdx, uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, vfLenInt32, preg);
                    AscendC::Reg::GatherMask<int32_t, GM::STORE_REG>(sqzIdx, idsReg, m1);
                    AscendC::Reg::DataCopyUnAlign<int32_t, PL::POST_MODE_UPDATE>(idxPtr, sqzIdx, uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, vfLenInt32, preg);
                    AscendC::Reg::GatherMask<int32_t, GM::STORE_REG>(sqzIdx, idsReg, m2);
                    AscendC::Reg::DataCopyUnAlign<int32_t, PL::POST_MODE_UPDATE>(idxPtr, sqzIdx, uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, vfLenInt32, preg);
                    AscendC::Reg::GatherMask<int32_t, GM::STORE_REG>(sqzIdx, idsReg, m3);
                    AscendC::Reg::DataCopyUnAlign<int32_t, PL::POST_MODE_UPDATE>(idxPtr, sqzIdx, uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, vfLenInt32, preg);
                } else {
                    AscendC::Reg::MaskUnPack<HL::LOWEST>(m0, cmpReg);
                    AscendC::Reg::MaskUnPack<HL::HIGHEST>(m1, cmpReg);
                    AscendC::Reg::GatherMask<int32_t, GM::STORE_REG>(sqzIdx, idsReg, m0);
                    AscendC::Reg::DataCopyUnAlign<int32_t, PL::POST_MODE_UPDATE>(idxPtr, sqzIdx, uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, vfLenInt32, preg);
                    AscendC::Reg::GatherMask<int32_t, GM::STORE_REG>(sqzIdx, idsReg, m1);
                    AscendC::Reg::DataCopyUnAlign<int32_t, PL::POST_MODE_UPDATE>(idxPtr, sqzIdx, uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, vfLenInt32, preg);
                }
            }
            AscendC::Reg::DataCopyUnAlignPost(idxPtr, uregIdx);
        }
        arNum = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / sizeof(int32_t);
    }

    // 8 字节索引压缩:reinterpret int64,CompareScalar<int64>(double 左移丢符号位),MaskPack 把 int64 mask
    // 压到 int32,GatherMask<int32> 压缩线性 idx;idsReg 每 iter 推进 vfLenV_(=V32/2 个 int64 元素)。
    __aicore__ inline void CompactIdxB64(int32_t num, int64_t gmBase, LocalTensor<TValue>& xUb,
                                         LocalTensor<int32_t>& idxUb, uint64_t& arNum)
    {
        constexpr bool isFp = AscendC::IsSameType<TValue, double>::value;
        auto xUbPtr = (__ubuf__ int64_t*)xUb.GetPhyAddr();
        auto idxPtr = (__ubuf__ int32_t*)idxUb.GetPhyAddr();
        uint16_t repeatElm = (uint16_t)this->vfLenV_;
        uint16_t repeatTimes = CeilDivision(num, repeatElm);
        int32_t scalar = static_cast<int32_t>(gmBase);
        uint32_t sreg1 = (uint32_t)num;
        int32_t advance = (int32_t)this->vfLenV_;
        __VEC_SCOPE__
        {
            AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
            AscendC::Reg::UnalignReg uregIdx;
            RegTensor<int32_t> idsReg;
            RegTensor<int32_t> sqzIdx;
            RegTensor<int64_t> xReg;
            RegTensor<int64_t> shiftReg;
            MaskReg cmpReg;
            MaskReg packedMask;
            MaskReg preg;
            preg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
            Arange(idsReg, scalar);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                MaskReg pnum = UpdateMask<int64_t>(sreg1);
                AscendC::Reg::AddrReg xOffset = AscendC::Reg::CreateAddrReg<int64_t>(i, repeatElm);
                DataCopy(xReg, xUbPtr, xOffset);
                if constexpr (isFp) {
                    AscendC::Reg::ShiftLefts(shiftReg, xReg, (int16_t)1, pnum);
                    CompareScalar<int64_t, CMPMODE::NE>(cmpReg, shiftReg, (int64_t)0, pnum);
                } else {
                    CompareScalar<int64_t, CMPMODE::NE>(cmpReg, xReg, (int64_t)0, pnum);
                }
                AscendC::Reg::MaskPack<AscendC::Reg::HighLowPart::LOWEST>(packedMask, cmpReg);
                AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(sqzIdx, idsReg, packedMask);
                AscendC::Reg::DataCopyUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(idxPtr, sqzIdx,
                                                                                                    uregIdx);
                AscendC::Reg::Adds(idsReg, idsReg, advance, preg);
            }
            AscendC::Reg::DataCopyUnAlignPost(idxPtr, uregIdx);
        }
        arNum = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / sizeof(int32_t);
    }

    // 8 字节值(double/int64/uint64):CompactIdx 已把非零的全局线性位置压缩进 idxUb[0:count](连续)。
    // 照 gather_v2 两段式:逐寄存器读连续位置 → 减 gmBase 得 tile 局部 → DataCopyGather 从 xUb gather 原始
    // 8 字节值(目标寄存器 RegTraitNumTwo:int64 占两个 32 位槽,否则只搬对低 32 位)→ 对齐 DataCopy 写。
    // 因位置已压缩连续,输出位=输入位(1:1),不需 DataCopyUnAlign(其也不支持 RegTraitNumTwo);位置已排除
    // -0.0(计数遍 ShiftLefts 处理),直接 gather 原值即可(不必再动符号位)。
    __aicore__ inline void CompactValueB64(int64_t gmBase, uint64_t count, LocalTensor<int32_t>& idxUb,
                                           LocalTensor<TValue>& xUb, LocalTensor<TValue>& valUb)
    {
        auto xUbPtr = (__ubuf__ int64_t*)xUb.GetPhyAddr();
        auto valPtr = (__ubuf__ int64_t*)valUb.GetPhyAddr();
        auto idxPtr = (__ubuf__ int32_t*)idxUb.GetPhyAddr();
        int32_t gmBaseNeg = -(int32_t)gmBase;
        uint16_t vfLen32 = (uint16_t)NZV_VF_LEN_INT32;
        uint32_t cntReg = (uint32_t)count;
        uint16_t loops = CeilDivision((int32_t)count, (int32_t)vfLen32);
        __VEC_SCOPE__
        {
            RegTensor<int32_t> posReg;
            RegTensor<int32_t> localPos;
            RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> valReg;
            for (uint16_t i = 0; i < loops; i++) {
                MaskReg preg = UpdateMask<int32_t>(cntReg);
                AscendC::Reg::AddrReg off = AscendC::Reg::CreateAddrReg<int32_t>(i, vfLen32);
                DataCopy(posReg, idxPtr, off);
                AscendC::Reg::Adds(localPos, posReg, gmBaseNeg, preg); // 全局位置 → tile 局部
                AscendC::Reg::DataCopyGather(valReg, xUbPtr, (RegTensor<uint32_t>&)localPos, preg);
                DataCopy(valPtr + (uint64_t)i * vfLen32, valReg, preg); // 1:1 对齐写(gather_v2 模式)
            }
        }
    }

    // scope1:压缩线性 idx,推进 AR → arNum
    __aicore__ inline void CompactIdx(int32_t num, int64_t gmBase, LocalTensor<TValue>& xUb,
                                      LocalTensor<int32_t>& idxUb, uint64_t& arNum)
    {
        if constexpr (sizeof(TValue) < 4) {
            CompactIdxSub32(num, gmBase, xUb, idxUb, arNum);
            return;
        }
        if constexpr (sizeof(TValue) == 8) {
            CompactIdxB64(num, gmBase, xUb, idxUb, arNum);
            return;
        }
        if constexpr (sizeof(TValue) == 4) {
            auto xPtr = (__ubuf__ TValue*)xUb.GetPhyAddr();
            auto idxPtr = (__ubuf__ int32_t*)idxUb.GetPhyAddr();
            uint16_t repeatElm = (uint16_t)this->vfLenV_;
            uint16_t repeatTimes = CeilDivision(num, repeatElm);
            int32_t scalar = static_cast<int32_t>(gmBase);
            uint32_t sreg1 = (uint32_t)num;
            __VEC_SCOPE__
            {
                AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
                AscendC::Reg::UnalignReg uregIdx;
                RegTensor<int32_t> idsReg;
                RegTensor<int32_t> sqzIdx;
                RegTensor<TValue> xReg;
                RegTensor<TValue> zeroXReg;
                MaskReg cmpReg;
                MaskReg preg;
                preg = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
                Duplicate(zeroXReg, (TValue)0);
                Arange(idsReg, scalar);
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    MaskReg pnum = UpdateMask<TValue>(sreg1);
                    AscendC::Reg::AddrReg xOffset = AscendC::Reg::CreateAddrReg<uint32_t>(i, repeatElm);
                    DataCopy(xReg, xPtr, xOffset);
                    Select(xReg, xReg, zeroXReg, pnum);                                // 尾部脏 UB 清 0
                    CompareScalar<TValue, CMPMODE::NE>(cmpReg, xReg, (TValue)0, pnum); // nan!=0 → nan 计非零
                    AscendC::Reg::GatherMask<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(sqzIdx, idsReg, cmpReg);
                    AscendC::Reg::DataCopyUnAlign<int32_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(idxPtr, sqzIdx,
                                                                                                        uregIdx);
                    AscendC::Reg::Adds(idsReg, idsReg, (int32_t)repeatElm, preg);
                }
                AscendC::Reg::DataCopyUnAlignPost(idxPtr, uregIdx);
            }
            arNum = (AscendC::Reg::GetSpr<SpecialPurposeReg::AR>()) / sizeof(int32_t);
        }
    }

    // scope2:同掩码压缩 value(重载 x + 同 Select/compare,与 idx 位置对齐)
    __aicore__ inline void CompactValue(int32_t num, int64_t gmBase, uint64_t count, LocalTensor<int32_t>& idxUb,
                                        LocalTensor<TValue>& xUb, LocalTensor<TValue>& valUb)
    {
        if constexpr (sizeof(TValue) == 8) {
            CompactValueB64(gmBase, count, idxUb, xUb, valUb);
            return;
        }
        // sub-32bit(1/2 字节)与 4 字节共用:GatherMask<TValue> 按 TValue 宽度压缩值,CreateAddrReg<TValue>
        // 修步长即可。仅 8 字节(GatherMask 不支持 int64)才走上面的 B64 gather 路。
        if constexpr (sizeof(TValue) <= 4) {
            auto xPtr = (__ubuf__ TValue*)xUb.GetPhyAddr();
            auto valPtr = (__ubuf__ TValue*)valUb.GetPhyAddr();
            uint16_t repeatElm = (uint16_t)this->vfLenV_;
            uint16_t repeatTimes = CeilDivision(num, repeatElm);
            uint32_t sreg2 = (uint32_t)num;
            __VEC_SCOPE__
            {
                AscendC::Reg::ClearSpr<SpecialPurposeReg::AR>();
                AscendC::Reg::UnalignReg uregVal;
                RegTensor<TValue> sqzVal;
                RegTensor<TValue> xReg;
                RegTensor<TValue> zeroXReg;
                MaskReg cmpReg;
                Duplicate(zeroXReg, (TValue)0);
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    MaskReg pnum = UpdateMask<TValue>(sreg2);
                    // AddrReg 用 TValue 宽度(sub-32bit 时步长按 dtype 字节;4 字节等价原 uint32 无变化)
                    AscendC::Reg::AddrReg xOffset = AscendC::Reg::CreateAddrReg<TValue>(i, repeatElm);
                    DataCopy(xReg, xPtr, xOffset);
                    Select(xReg, xReg, zeroXReg, pnum);
                    CompareScalar<TValue, CMPMODE::NE>(cmpReg, xReg, (TValue)0, pnum);
                    AscendC::Reg::GatherMask<TValue, AscendC::Reg::GatherMaskMode::STORE_REG>(sqzVal, xReg, cmpReg);
                    AscendC::Reg::DataCopyUnAlign<TValue, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(valPtr, sqzVal,
                                                                                                       uregVal);
                }
                AscendC::Reg::DataCopyUnAlignPost(valPtr, uregVal);
            }
        }
    }

    // 线性 idx → row=idx/col, col=idx%col(quick-div);坐标主序:index[0..]=row, index[numel..]=col;value 顺写。
    __aicore__ inline void WriteCoordMajorAndValue(LocalTensor<int32_t>& idxUb, LocalTensor<TValue>& valUb,
                                                   uint64_t arNum, uint64_t loopGm)
    {
        int32_t alignNum = (int32_t)CeilDivision(arNum, NZV_VF_LEN_INT32) * NZV_VF_LEN_INT32;
        auto idxPtr = (__ubuf__ uint32_t*)idxUb.GetPhyAddr();
        auto rowPtr = (__ubuf__ uint32_t*)idxUb[alignNum].GetPhyAddr();     // row 平面
        auto colPtr = (__ubuf__ uint32_t*)idxUb[alignNum * 2].GetPhyAddr(); // col 平面
        uint32_t sreg = (uint32_t)arNum;
        uint16_t repeatTimes = CeilDivision(arNum, NZV_VF_LEN_INT32);
        uint32_t qm = (uint32_t)this->tilingData_->quickDivColM;
        int16_t qk = (int16_t)this->tilingData_->quickDivColK; // 移位量须 int16(设备约束)
        uint32_t colVal = (uint32_t)this->tilingData_->col;
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> idsReg;
            RegTensor<uint32_t> qmulHi;
            RegTensor<uint32_t> tmp;
            RegTensor<uint32_t> rowReg;
            RegTensor<uint32_t> colReg;
            RegTensor<uint32_t> qmReg;
            MaskReg preg;
            Duplicate(qmReg, qm);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = UpdateMask<uint32_t>(sreg);
                AscendC::Reg::AddrReg off = AscendC::Reg::CreateAddrReg<uint32_t>(i, NZV_VF_LEN_INT32);
                DataCopy(idsReg, idxPtr, off);
                // row = idx / col:  hi = mulhi(idx, qm); t = idx + hi; row = t >> qk
                Mull(tmp, qmulHi, idsReg, qmReg, preg);
                Add(tmp, idsReg, qmulHi, preg);
                ShiftRights(rowReg, tmp, qk, preg);
                // col = idx - row*col
                Muls(tmp, rowReg, colVal, preg);
                Sub(colReg, idsReg, tmp, preg);
                DataCopy(rowPtr, rowReg, off, preg);
                DataCopy(colPtr, colReg, off, preg);
            }
        }
        // 坐标主序 CopyOut:index[gmOffset+loopGm-arNum .. ] = row 平面;index[numel + ...] = col 平面
        event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evVMte3);
        WaitFlag<HardEvent::V_MTE3>(evVMte3);
        uint64_t dstBase = this->gmOffset_ + loopGm - arNum;
        DataCopyExtParams rowParams{1, static_cast<uint32_t>(arNum * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(this->indexGm_[dstBase], idxUb[alignNum], rowParams);
        DataCopyPad(this->indexGm_[this->tilingData_->numel + dstBase], idxUb[alignNum * 2], rowParams);
        // value 顺写
        DataCopyExtParams valParams{1, static_cast<uint32_t>(arNum * sizeof(TValue)), 0, 0, 0};
        DataCopyPad(this->valueGm_[dstBase], valUb, valParams);
        // 多 tile:idxOutBuf_/valOutBuf_ 每 tile 复用。下一 tile 的 CompactIdxAndValue(V)会覆写它们,
        // 必须等本 tile 上面三条 DataCopyPad(MTE3 读 idxUb/valUb)全部完成,否则 WAR 冲突读到半写数据。
        event_t evMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(evMte3V);
        WaitFlag<HardEvent::MTE3_V>(evMte3V);
    }
};
} // namespace NonZeroWithValue
#endif // NON_ZERO_WITH_VALUE_GENERAL_H
