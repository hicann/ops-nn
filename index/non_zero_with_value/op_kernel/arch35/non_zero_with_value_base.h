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
 * \file non_zero_with_value_base.h
 * \brief NonZeroWithValue arch35 共享基类 —— 2D/fp32/transpose=true(坐标主序)。
 *        提供:Init / 计数遍(找非零+建掩码+per-core count) / 跨核 SyncAll+前缀和 / 写 count。
 *        参照 index/non_zero non_zero_base.h,裁为 2D 单路径 + value 通道。
 */
#ifndef NON_ZERO_WITH_VALUE_BASE_H
#define NON_ZERO_WITH_VALUE_BASE_H

#include "op_kernel/platform_util.h"
#include "kernel_operator.h"

namespace NonZeroWithValue {
using namespace Ops::Base;
using namespace AscendC;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

constexpr int32_t NZV_ONE_BLOCK = 32;
constexpr int32_t NZV_ADD_UB_SIZE = 72 * 32;
constexpr int32_t NZV_DB_BUFFER = 2;
constexpr int32_t NZV_DIM8 = 8;
const uint32_t NZV_VF_LEN_INT32 = GetVRegSize() / sizeof(int32_t);

// TValue = float(x/value dtype),TIndex = int32(index dtype)
template <typename TValue, typename TIndex>
class NonZeroWithValueBase {
public:
    __aicore__ inline NonZeroWithValueBase() {}

protected:
    const NonZeroWithValueTilingData* tilingData_ = nullptr;
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, NZV_DB_BUFFER> inQueX_;
    GlobalTensor<TValue> xGm_;
    GlobalTensor<TValue> valueGm_;
    GlobalTensor<TIndex> indexGm_;
    GlobalTensor<int32_t> countGm_;
    GlobalTensor<uint32_t> workspaceGm_;
    TBuf<QuePosition::VECCALC> addUb_;

    DataCopyExtParams copyParams_;
    DataCopyPadExtParams<TValue> padParams_{false, 0, 0, 0};
    uint32_t blockIdx_ = 0;
    // 本核输出全局起始(前缀和)。tiling 已守卫 numel ≤ INT32_MAX,
    // 而 count/offset 上限为 numel,故 uint32 前缀和不会回绕(恒 < UINT32_MAX)。
    uint32_t gmOffset_ = 0;
    uint64_t allNum_ = 0; // 全局非零总数
    uint32_t vfLenV_ = GetVRegSize() / sizeof(TValue);

    __aicore__ inline void InitBase(GM_ADDR x, GM_ADDR value, GM_ADDR index, GM_ADDR count, GM_ADDR workspace,
                                    const NonZeroWithValueTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        xGm_.SetGlobalBuffer((__gm__ TValue*)x);
        valueGm_.SetGlobalBuffer((__gm__ TValue*)value);
        indexGm_.SetGlobalBuffer((__gm__ TIndex*)index);
        countGm_.SetGlobalBuffer((__gm__ int32_t*)count, 1);
        // 各核 count 落 workspace[blockIdx * NZV_ONE_BLOCK],buffer 上界按实际核数(tiling)动态取,
        // 不硬编码 72(核数 > 72 时硬编码会越界)。
        workspaceGm_.SetGlobalBuffer((__gm__ uint32_t*)workspace, tilingData->realCoreNum * NZV_ONE_BLOCK);
        tilingData_ = tilingData;
        pipe_.InitBuffer(inQueX_, NZV_DB_BUFFER, tilingData_->xInputSize);
        pipe_.InitBuffer(addUb_, NZV_ADD_UB_SIZE);
    }

    // 找非零 + 累加 per-core count(fp32,复用 non_zero VfPerCoreNonZeroNum3 模式)
    // 注:输出遍(general.h CompactIdxAndValue)按 tile 重算 mask,不读这里存的 mask,
    //     故不落 mask 到 UB —— 存下的 mask 是死代码,且多 tile 时 loop*maskLoopNum 偏移
    //     既未对齐 32B 又越界(buffer 只按单 tile 尺寸分配)→ VEC_ERROR,直接去掉。
    // sub-32bit(1/2 字节)计数:一个 VF 寄存器装 4×/2× 于 int32 的元素,CompareScalar 出的 mask 也是
    // TValue 宽度。Select 到 TValue 宽度 0/1,再 UnPack 展宽到 int32(1字节拆4、2字节拆2)分别累加,
    // 正确统计所有车道(照搬母算子 non_zero VfPerCoreNonZeroNum 的 UnPack 写法)。4 字节路径不走此函数。
    __aicore__ inline void VfFindNonZeroSub32(int32_t computeNum, LocalTensor<TValue>& xUb,
                                              LocalTensor<uint32_t>& addUb)
    {
        using USel = typename AscendC::Conditional<sizeof(TValue) == 1, uint8_t, uint16_t>::type;
        auto xUbPtr = (__ubuf__ TValue*)xUb.GetPhyAddr();
        auto dstUbPtr = (__ubuf__ uint32_t*)addUb.GetPhyAddr();
        uint32_t sreg = (uint32_t)computeNum;
        uint16_t repeatElm = (uint16_t)vfLenV_;
        uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
        uint32_t addMask = NZV_VF_LEN_INT32;
        __VEC_SCOPE__
        {
            MaskReg preg;
            MaskReg addComReg;
            MaskReg cmpReg;
            RegTensor<TValue> xSrcReg;
            RegTensor<TValue> zeroXReg;
            RegTensor<USel> src1Reg;
            RegTensor<USel> src0Reg;
            RegTensor<USel> selectReg;
            RegTensor<uint16_t> u16a;
            RegTensor<uint16_t> u16b;
            RegTensor<uint32_t> u32a;
            RegTensor<uint32_t> u32b;
            RegTensor<uint32_t> u32c;
            RegTensor<uint32_t> u32d;
            RegTensor<uint32_t> addReg;
            DataCopy(addReg, dstUbPtr);
            Duplicate(src1Reg, (USel)1);
            Duplicate(src0Reg, (USel)0);
            Duplicate(zeroXReg, (TValue)0);
            addComReg = UpdateMask<uint32_t>(addMask);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = UpdateMask<TValue>(sreg);
                AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<TValue>(i, repeatElm);
                DataCopy(xSrcReg, xUbPtr, srcOffset);
                Select(xSrcReg, xSrcReg, zeroXReg, preg);
                CompareScalar<TValue, CMPMODE::NE>(cmpReg, xSrcReg, (TValue)0, preg);
                Select(selectReg, src1Reg, src0Reg, cmpReg);
                if constexpr (sizeof(TValue) == 1) {
                    AscendC::MicroAPI::UnPack<uint16_t, uint8_t, AscendC::MicroAPI::HighLowPart::LOWEST>(u16a,
                                                                                                         selectReg);
                    AscendC::MicroAPI::UnPack<uint16_t, uint8_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(u16b,
                                                                                                          selectReg);
                    AscendC::MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(u32a, u16a);
                    AscendC::MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(u32b, u16a);
                    AscendC::MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(u32c, u16b);
                    AscendC::MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(u32d, u16b);
                    Add(addReg, u32a, addReg, addComReg);
                    Add(addReg, u32b, addReg, addComReg);
                    Add(addReg, u32c, addReg, addComReg);
                    Add(addReg, u32d, addReg, addComReg);
                } else {
                    AscendC::MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(u32a,
                                                                                                          selectReg);
                    AscendC::MicroAPI::UnPack<uint32_t, uint16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(u32b,
                                                                                                           selectReg);
                    Add(addReg, u32a, addReg, addComReg);
                    Add(addReg, u32b, addReg, addComReg);
                }
            }
            DataCopy(dstUbPtr, addReg, addComReg);
        }
    }

    // 8 字节(double/int64/uint64)计数:一个 VF 寄存器只装 V32/2 个元素,mask 也是 int64 宽度。
    // reinterpret 成 int64 后 CompareScalar<int64>(double 先左移 1 位丢符号位,使 -0.0 与 +0.0 都判零);
    // MaskPack 把 int64 mask 压到 int32 宽度,再 Select uint32 + Add(复用 uint32 累加器)。GatherMask 不支持
    // 64 位,故不能像 4 字节那样直接压;照母算子 non_zero B64 套路。
    __aicore__ inline void VfFindNonZeroB64(int32_t computeNum, LocalTensor<TValue>& xUb, LocalTensor<uint32_t>& addUb)
    {
        constexpr bool isFp = AscendC::IsSameType<TValue, double>::value;
        auto xUbPtr = (__ubuf__ int64_t*)xUb.GetPhyAddr();
        auto dstUbPtr = (__ubuf__ uint32_t*)addUb.GetPhyAddr();
        uint32_t sreg = (uint32_t)computeNum;
        uint16_t repeatElm = (uint16_t)vfLenV_;
        uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
        uint32_t addMask = NZV_VF_LEN_INT32;
        __VEC_SCOPE__
        {
            MaskReg preg;
            MaskReg addComReg;
            MaskReg cmpReg;
            MaskReg packedMask;
            RegTensor<int64_t> xSrcReg;
            RegTensor<int64_t> shiftReg;
            RegTensor<uint32_t> src1Reg;
            RegTensor<uint32_t> src0Reg;
            RegTensor<uint32_t> selectReg;
            RegTensor<uint32_t> addReg;
            DataCopy(addReg, dstUbPtr);
            Duplicate(src1Reg, (uint32_t)1);
            Duplicate(src0Reg, (uint32_t)0);
            addComReg = UpdateMask<uint32_t>(addMask);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = UpdateMask<int64_t>(sreg);
                AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<int64_t>(i, repeatElm);
                DataCopy(xSrcReg, xUbPtr, srcOffset);
                if constexpr (isFp) {
                    AscendC::MicroAPI::ShiftLefts(shiftReg, xSrcReg, (int16_t)1, preg); // 丢符号位:-0.0/+0.0 都→0
                    CompareScalar<int64_t, CMPMODE::NE>(cmpReg, shiftReg, (int64_t)0, preg);
                } else {
                    CompareScalar<int64_t, CMPMODE::NE>(cmpReg, xSrcReg, (int64_t)0, preg);
                }
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(packedMask, cmpReg);
                Select(selectReg, src1Reg, src0Reg, packedMask);
                Add(addReg, selectReg, addReg, addComReg);
            }
            DataCopy(dstUbPtr, addReg, addComReg);
        }
    }

    __aicore__ inline void VfFindNonZero(int32_t computeNum, LocalTensor<TValue>& xUb, LocalTensor<uint32_t>& addUb)
    {
        if constexpr (sizeof(TValue) < 4) {
            VfFindNonZeroSub32(computeNum, xUb, addUb);
            return;
        }
        if constexpr (sizeof(TValue) == 8) {
            VfFindNonZeroB64(computeNum, xUb, addUb);
            return;
        }
        if constexpr (sizeof(TValue) == 4) {
            auto xUbPtr = (__ubuf__ TValue*)xUb.GetPhyAddr();
            auto dstUbPtr = (__ubuf__ uint32_t*)addUb.GetPhyAddr();
            uint32_t sreg = (uint32_t)computeNum;
            uint16_t repeatElm = (uint16_t)vfLenV_;
            uint16_t repeatTimes = CeilDivision(computeNum, repeatElm);
            uint32_t addMask = NZV_VF_LEN_INT32;
            __VEC_SCOPE__
            {
                MaskReg preg;
                MaskReg addComReg;
                RegTensor<TValue> xSrcReg;
                RegTensor<TValue> zeroXReg;
                RegTensor<uint32_t> src1Reg;
                RegTensor<uint32_t> src0Reg;
                RegTensor<uint32_t> selectReg;
                MaskReg cmpReg;
                RegTensor<uint32_t> addReg;
                DataCopy(addReg, dstUbPtr);
                Duplicate(src1Reg, (uint32_t)1);
                Duplicate(src0Reg, (uint32_t)0);
                Duplicate(zeroXReg, (TValue)0);
                addComReg = UpdateMask<uint32_t>(addMask);
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<TValue>(sreg);
                    AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<uint32_t>(i, repeatElm);
                    DataCopy(xSrcReg, xUbPtr, srcOffset);
                    // DataCopyPad 仅载入 computeNum 个元素,VF 读满整寄存器 → 尾部车道为脏 UB。
                    // 显式将非活跃车道(超出本次 preg)清 0,避免脏非零值被误计数/误压缩。
                    Select(xSrcReg, xSrcReg, zeroXReg, preg);
                    CompareScalar<TValue, CMPMODE::NE>(cmpReg, xSrcReg, (TValue)0, preg); // nan!=0 为真 → nan 计非零
                    Select(selectReg, src1Reg, src0Reg, cmpReg);
                    Add(addReg, selectReg, addReg, addComReg);
                }
                DataCopy(dstUbPtr, addReg, addComReg);
            }
        }
    }

    // per-core count 归约到单值
    __aicore__ inline void VfReduceSum(LocalTensor<uint32_t>& addUb)
    {
        auto addUbPtr = (__ubuf__ uint32_t*)addUb.GetPhyAddr();
        uint32_t mask = NZV_VF_LEN_INT32;
        uint32_t oneMask = 1;
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> addReg;
            RegTensor<uint32_t> dstReg;
            MaskReg addMask;
            MaskReg oneMaskReg;
            addMask = UpdateMask<uint32_t>(mask);
            oneMaskReg = UpdateMask<uint32_t>(oneMask);
            DataCopy(addReg, addUbPtr);
            ReduceSum(dstReg, addReg, addMask);
            DataCopy(addUbPtr, dstReg, oneMaskReg);
        }
    }

    // 独占前缀和:offset = Σ 前面核 count;allNum = Σ 全部核 count。
    // 跨核 count 经 DataCopyPad gather 后按 32 字节 UB block 对齐落位(stride 8 个 uint32),
    // 必须按 addUbPtr + i*8 逐核读(对齐 non_zero VfAllNum 已验证写法),不能当连续 lane 用 ReduceSum。
    __aicore__ inline void VfPrefixSum(LocalTensor<uint32_t>& addUb)
    {
        auto addUbPtr = (__ubuf__ uint32_t*)addUb.GetPhyAddr();
        uint16_t coreNum = tilingData_->realCoreNum;
        uint16_t blockNum = blockIdx_;
        uint16_t addAllNum = coreNum - blockNum;
        uint32_t oneMask = 1;
        __VEC_SCOPE__
        {
            RegTensor<uint32_t> addReg;
            RegTensor<uint32_t> addReg2;
            RegTensor<uint32_t> offReg;  // Σ 前 blockIdx 核 = 独占前缀 offset
            RegTensor<uint32_t> restReg; // Σ 本核及之后
            MaskReg oneMaskReg;
            oneMaskReg = UpdateMask<uint32_t>(oneMask);
            Duplicate(offReg, (uint32_t)0);
            Duplicate(restReg, (uint32_t)0);
            for (uint16_t i = 0; i < blockNum; i++) {
                DataCopy(addReg, addUbPtr + i * NZV_DIM8);
                Add(offReg, offReg, addReg, oneMaskReg);
            }
            for (uint16_t j = 0; j < addAllNum; j++) {
                DataCopy(addReg2, addUbPtr + (j + blockNum) * NZV_DIM8);
                Add(restReg, restReg, addReg2, oneMaskReg);
            }
            Add(restReg, offReg, restReg, oneMaskReg); // allNum
            DataCopy(addUbPtr, restReg, oneMaskReg);
            DataCopy(addUbPtr + NZV_DIM8, offReg, oneMaskReg);
        }
    }

    // 计数遍:逐 tile 找非零,per-core count 写 workspace,SyncAll,前缀和求 gmOffset_/allNum_,写 count
    __aicore__ inline void ComputeCount(LocalTensor<uint32_t>& addUb, int32_t loopNum, int32_t beforeNum,
                                        int32_t tailNum)
    {
        for (int32_t loop = 0; loop <= loopNum; loop++) {
            int32_t curNum = (loop < loopNum) ? beforeNum : tailNum;
            if (curNum <= 0) {
                break;
            }
            LocalTensor<TValue> xUb = inQueX_.AllocTensor<TValue>();
            copyParams_.blockCount = 1;
            copyParams_.blockLen = curNum * sizeof(TValue);
            copyParams_.srcStride = 0;
            copyParams_.dstStride = 0;
            DataCopyPad(xUb, xGm_[blockIdx_ * tilingData_->numPerCore + loop * tilingData_->ubFactorNum], copyParams_,
                        padParams_);
            inQueX_.template EnQue<QuePosition::VECIN, QuePosition::VECCALC>(xUb);
            LocalTensor<TValue> xUbC = inQueX_.template DeQue<QuePosition::VECIN, QuePosition::VECCALC, TValue>();
            VfFindNonZero(curNum, xUbC, addUb);
            inQueX_.template EnQue<QuePosition::VECCALC, QuePosition::VECOUT>(xUbC);
            LocalTensor<TValue> xUbO = inQueX_.template DeQue<QuePosition::VECCALC, QuePosition::VECOUT, TValue>();
            inQueX_.FreeTensor(xUbO);
        }

        VfReduceSum(addUb);
        event_t evVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evVMte3);
        WaitFlag<HardEvent::V_MTE3>(evVMte3);
        DataCopyPad(workspaceGm_[blockIdx_ * NZV_ONE_BLOCK], addUb, {1, 4, 0, 0});
        SyncAll();
        copyParams_.blockCount = tilingData_->realCoreNum;
        copyParams_.blockLen = 4;
        copyParams_.srcStride = 31 * 4;
        copyParams_.dstStride = 0;
        DataCopyPad(addUb, workspaceGm_, copyParams_, {false, 0, 0, 0});
        event_t evMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        VfPrefixSum(addUb);
        event_t evVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evVS);
        WaitFlag<HardEvent::V_S>(evVS);
        allNum_ = addUb.GetValue(0);
        gmOffset_ = addUb.GetValue(NZV_DIM8);

        // 写显式 count 输出(仅 0 号核)
        if (blockIdx_ == 0) {
            LocalTensor<int32_t> cntUb = addUb.template ReinterpretCast<int32_t>();
            cntUb.SetValue(NZV_DIM8 * 2, static_cast<int32_t>(allNum_));
            event_t evSMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(evSMte3);
            WaitFlag<HardEvent::S_MTE3>(evSMte3);
            DataCopyPad(countGm_, cntUb[NZV_DIM8 * 2], {1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0});
        }
    }
};
} // namespace NonZeroWithValue
#endif // NON_ZERO_WITH_VALUE_BASE_H
