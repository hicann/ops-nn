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
 * \file broadcast_gradient_args_perf.h
 * \brief
 */

#ifndef BROADCAST_GRADIENT_ARGS_PERF_H_
#define BROADCAST_GRADIENT_ARGS_PERF_H_
#include "broadcast_gradient_args_base.h"

namespace BroadcastGradientArgs {
using namespace AscendC;

template <typename T>
class BroadcastGradientArgsPerf : public BroadcastGradientArgsBase<T> {
public:
    __aicore__ inline BroadcastGradientArgsPerf(){};

    __aicore__ inline uint16_t CEIL_DIV(uint16_t x, uint16_t y) { return (x + y - 1) / y; }

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR outShape,
                                const BroadcastGradientArgsTilingData* __restrict tilingData)
    {
        // init global memory
        tilingData_ = tilingData;
        alignT = this->blockSize / sizeof(T);
        vlInput = this->vlLen / sizeof(T);
        vlInt64 = this->vlLen / sizeof(int64_t);
        vLoopT = CEIL_DIV(static_cast<uint16_t>(tilingData_->maxRank), vlInput);
        vLoopInt64 = CEIL_DIV(static_cast<uint16_t>(tilingData_->maxRank), vlInt64);
        this->x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        this->x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        this->y1Gm.SetGlobalBuffer((__gm__ T*)y1);
        this->y2Gm.SetGlobalBuffer((__gm__ T*)y2);
        this->outShapeGm.SetGlobalBuffer((__gm__ uint64_t*)outShape);
    }

    __aicore__ inline void CopyInputAndPad(const LocalTensor<T>& inUb, const GlobalTensor<T>& inGm, int64_t inLen)
    {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = 0;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        if (inLen == tilingData_->maxRank) {
            copyInParams.blockLen = inLen * sizeof(T);
            if (copyInParams.blockLen > 0) {
                DataCopyPad(inUb, inGm, copyInParams, padParams);
            }
        } else {
            int64_t padOneNum = tilingData_->maxRank - inLen;
            Duplicate<T>(inUb, 1, padOneNum);
            int64_t alignOffset = (padOneNum + alignT - 1) / alignT * alignT;
            int64_t setValueNum = alignOffset - padOneNum > inLen ? inLen : alignOffset - padOneNum;
            for (int64_t i = 0; i < setValueNum; ++i) {
                inUb.SetValue(padOneNum + i, inGm.GetValue(i));
            }
            copyInParams.blockLen = (inLen - setValueNum) * sizeof(T);
            if (copyInParams.blockLen > 0) {
                DataCopyPad(inUb[alignOffset], inGm[setValueNum], copyInParams, padParams);
            }
        }
    }

    __aicore__ inline int64_t CalcReduceAxes(__ubuf__ T* x, __ubuf__ int32_t* y, uint32_t xLen)
    {
        __VEC_SCOPE__
        {
            uint32_t srg0 = xLen;
            Reg::RegTensor<int32_t> init_index_reg, calc_index_reg, x_int32_reg, dst_reg;
            Reg::RegTensor<T> in_reg;
            Reg::UnalignRegForStore ureg;
            Reg::MaskReg maskCompare;
            Reg::MaskReg maskCalc;
            Reg::MaskReg maskAll = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
            Reg::ClearSpr<AscendC::SpecialPurposeReg::AR>();
            Reg::Arange(init_index_reg, 0);
            if constexpr (IsSameType<T, int64_t>::value) {
                for (uint16_t i = 0; i < vLoopInt64; i++) {
                    Reg::Adds(calc_index_reg, init_index_reg, i * vlInt64, maskAll);
                    Reg::LoadAlign(in_reg, x + i * vlInt64);
                    maskCalc = Reg::UpdateMask<int64_t>(srg0);
                    // mask未选择位置会置0，对尾块无影响
                    Reg::Cast<int32_t, int64_t, castTraitB642B32>(x_int32_reg, in_reg, maskCalc);
                    Reg::Pack((Reg::RegTensor<uint32_t>&)x_int32_reg, (Reg::RegTensor<uint64_t>&)x_int32_reg);
                    Reg::Compares<int32_t, CMPMODE::EQ>(maskCompare, x_int32_reg, 1, maskAll);
                    Reg::Squeeze<int32_t, Reg::GatherMaskMode::STORE_REG>(dst_reg, calc_index_reg, maskCompare);
                    Reg::StoreUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(y, dst_reg, ureg);
                }
            } else {
                for (uint16_t i = 0; i < vLoopT; i++) {
                    maskCalc = Reg::UpdateMask<int32_t>(srg0);
                    Reg::Adds(calc_index_reg, init_index_reg, i * vlInput, maskAll);
                    Reg::LoadAlign(in_reg, x + i * vlInput);
                    Reg::Compares<int32_t, CMPMODE::EQ>(maskCompare, in_reg, 1, maskCalc);
                    Reg::Squeeze<int32_t, Reg::GatherMaskMode::STORE_REG>(dst_reg, calc_index_reg, maskCompare);
                    Reg::StoreUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(y, dst_reg, ureg);
                }
            }
            Reg::StoreUnAlignPost(y, ureg);
        }
        return ((Reg::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t));
    }

    __aicore__ inline void CheckInput(__ubuf__ T* x1, __ubuf__ T* x2, __ubuf__ T* invalidFlag, __ubuf__ T* equalFlag,
                                      uint32_t calcLen)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<T> x1_reg, x2_reg, one_reg, zero_reg, eq_flag_reg, bool_reg1, bool_reg2, out_eq_reg,
                out_invalid_reg;
            Reg::MaskReg maskCompare0, maskCompare1, maskCompare2;
            Reg::MaskReg maskCalc;
            Reg::MaskReg maskMerge = Reg::CreateMask<T, Reg::MaskPattern::VL1>();
            Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::Duplicate(one_reg, 1, maskAll);
            Reg::Duplicate(zero_reg, 0, maskAll);
            Reg::Duplicate(out_eq_reg, 0, maskAll);
            Reg::Duplicate(out_invalid_reg, 0, maskAll);
            for (uint16_t i = 0; i < vLoopT; i++) {
                maskCalc = Reg::UpdateMask<T>(calcLen);
                Reg::LoadAlign(x1_reg, x1 + i * vlInput);
                Reg::LoadAlign(x2_reg, x2 + i * vlInput);
                // x1 x2不等的维度标识为true
                Reg::Compare<T, CMPMODE::NE>(maskCompare0, x1_reg, x2_reg, maskCalc);
                // x1 维度值不等于1维度标识为true
                Reg::Compares<T, CMPMODE::NE>(maskCompare1, x1_reg, 1, maskCalc);
                // x2 维度值不等于1维度标识为true
                Reg::Compares<T, CMPMODE::NE>(maskCompare2, x2_reg, 1, maskCalc);
                // maskReg转RegTensor
                // 可修改为Mask计算
                Reg::Select(eq_flag_reg, one_reg, zero_reg, maskCompare0);
                Reg::Select(bool_reg1, one_reg, zero_reg, maskCompare1);
                Reg::Select(bool_reg2, one_reg, zero_reg, maskCompare2);
                // 将未筛选的置0
                Reg::Adds(eq_flag_reg, eq_flag_reg, 0, maskCalc);
                // 两次与筛选出(x1 != x2) && (x1 != 1) && (x2 != 1)
                Reg::And(bool_reg1, bool_reg1, eq_flag_reg, maskCalc);
                Reg::And(bool_reg2, bool_reg2, bool_reg1, maskCalc);
                // 与之前vl循环的结果Max
                Reg::Max(out_eq_reg, out_eq_reg, eq_flag_reg, maskAll);
                Reg::Max(out_invalid_reg, out_invalid_reg, bool_reg2, maskAll);
            }
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(x1_reg, out_invalid_reg, maskAll);
            Reg::Reduce<AscendC::Reg::ReduceType::SUM>(x2_reg, out_eq_reg, maskAll);
            Reg::StoreAlign(invalidFlag, x1_reg, maskMerge);
            Reg::StoreAlign(equalFlag, x2_reg, maskMerge);
        }
    }

    __aicore__ inline void Process()
    {
        int64_t addrOffset = 0;
        int64_t singleUbSize = tilingData_->ubMaxRank * sizeof(T);
        LocalTensor<T> x1Ub(TPosition::VECCALC, addrOffset, singleUbSize);
        addrOffset += singleUbSize;
        LocalTensor<T> x2Ub(TPosition::VECCALC, addrOffset, singleUbSize);
        addrOffset += singleUbSize;
        LocalTensor<T> y1Ub(TPosition::VECCALC, addrOffset, singleUbSize);
        addrOffset += singleUbSize;
        LocalTensor<T> y2Ub(TPosition::VECCALC, addrOffset, singleUbSize);
        addrOffset += singleUbSize;
        LocalTensor<T> invalidFlagUb(TPosition::VECCALC, addrOffset, this->blockSize);
        addrOffset += this->blockSize;
        LocalTensor<T> equalFlagUb(TPosition::VECCALC, addrOffset, this->blockSize);
        addrOffset += this->blockSize;
        LocalTensor<uint64_t> y1ShapeUb(TPosition::VECCALC, addrOffset, this->blockSize);
        addrOffset += this->blockSize;
        LocalTensor<uint64_t> y2ShapeUb(TPosition::VECCALC, addrOffset, this->blockSize);
        addrOffset += this->blockSize;

        y1ShapeUb.SetValue(SHAPE_SIZE_IDX, FIRST_UINT64_SHAPE_DIM_ONE);
        y2ShapeUb.SetValue(SHAPE_SIZE_IDX, UINT64_SHAPE_DIM_ONE);

        CopyInputAndPad(x1Ub, this->x1Gm, tilingData_->x1Len);
        CopyInputAndPad(x2Ub, this->x2Gm, tilingData_->x2Len);
        // 后续Vector计算依赖搬入完成
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        // 后续Vector计算依赖Scalar结束
        SetFlag<HardEvent::S_V>(4);
        WaitFlag<HardEvent::S_V>(4);

        __ubuf__ T* x1UbAddr = (__ubuf__ T*)x1Ub.GetPhyAddr();
        __ubuf__ T* x2UbAddr = (__ubuf__ T*)x2Ub.GetPhyAddr();
        __ubuf__ T* y1UbAddr = (__ubuf__ T*)y1Ub.GetPhyAddr();
        __ubuf__ T* y2UbAddr = (__ubuf__ T*)y2Ub.GetPhyAddr();
        __ubuf__ T* invalidFlagUbAddr = (__ubuf__ T*)invalidFlagUb.GetPhyAddr();
        __ubuf__ T* equalFlagUbAddr = (__ubuf__ T*)equalFlagUb.GetPhyAddr();

        CheckInput(x1UbAddr, x2UbAddr, invalidFlagUbAddr, equalFlagUbAddr, tilingData_->maxRank);
        // invalidFlagUb.GetValue Scalar操作依赖CheckInput VF计算完成
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        assert(invalidFlagUb.GetValue(0) == 0, "Inputs x1 and x2 do not satisfy broadcasting rules !\n");

        // 长度相同并且值完全相同，y1 y2为空
        if ((tilingData_->x1Len == tilingData_->x2Len) && (equalFlagUb.GetValue(0) == 0)) {
            y1ShapeUb.SetValue(SHAPE_DIM0_IDX, 0);
            y2ShapeUb.SetValue(SHAPE_DIM0_IDX, 0);
            SetFlag<HardEvent::S_MTE3>(2);
            WaitFlag<HardEvent::S_MTE3>(2);
            DataCopyExtParams copyOutShapeParams{1, 2 * sizeof(uint64_t), 0, 0, 0};
            DataCopyPad(this->outShapeGm, y1ShapeUb, copyOutShapeParams);
            DataCopyPad(this->outShapeGm[SHAPE1_GM_IDX], y2ShapeUb, copyOutShapeParams);
            return;
        }
        // 计算reduce轴
        int64_t y1Num = 0;
        int64_t y2Num = 0;
        if constexpr (IsSameType<T, int64_t>::value) {
            LocalTensor<int32_t> y1UbInt32 = y1Ub.template ReinterpretCast<int32_t>()[tilingData_->ubMaxRank];
            LocalTensor<int32_t> y2UbInt32 = y2Ub.template ReinterpretCast<int32_t>()[tilingData_->ubMaxRank];
            __ubuf__ int32_t* y1UbInt32Addr = (__ubuf__ int32_t*)y1UbInt32.GetPhyAddr();
            __ubuf__ int32_t* y2UbInt32Addr = (__ubuf__ int32_t*)y2UbInt32.GetPhyAddr();
            y1Num = CalcReduceAxes(x1UbAddr, y1UbInt32Addr, tilingData_->maxRank);
            y2Num = CalcReduceAxes(x2UbAddr, y2UbInt32Addr, tilingData_->maxRank);
            Cast(y1Ub, y1UbInt32, RoundMode::CAST_NONE, tilingData_->maxRank);
            Cast(y2Ub, y2UbInt32, RoundMode::CAST_NONE, tilingData_->maxRank);
        } else {
            y1Num = CalcReduceAxes(x1UbAddr, y1UbAddr, tilingData_->maxRank);
            y2Num = CalcReduceAxes(x2UbAddr, y2UbAddr, tilingData_->maxRank);
        }

        SetFlag<HardEvent::V_MTE3>(3);
        WaitFlag<HardEvent::V_MTE3>(3);

        DataCopyExtParams copyOutY1Params;
        copyOutY1Params.blockCount = 1;
        copyOutY1Params.blockLen = y1Num * sizeof(T);
        copyOutY1Params.srcStride = 0;
        copyOutY1Params.dstStride = 0;

        DataCopyExtParams copyOutY2Params;
        copyOutY2Params.blockCount = 1;
        copyOutY2Params.blockLen = y2Num * sizeof(T);
        copyOutY2Params.srcStride = 0;
        copyOutY2Params.dstStride = 0;
        if (copyOutY1Params.blockLen > 0) {
            DataCopyPad(this->y1Gm, y1Ub, copyOutY1Params);
        }
        if (copyOutY2Params.blockLen > 0) {
            DataCopyPad(this->y2Gm, y2Ub, copyOutY2Params);
        }

        // 设置输出shape
        SetFlag<HardEvent::V_S>(2);
        WaitFlag<HardEvent::V_S>(2);
        y1ShapeUb.SetValue(SHAPE_DIM0_IDX, y1Num);
        y2ShapeUb.SetValue(SHAPE_DIM0_IDX, y2Num);
        SetFlag<HardEvent::S_MTE3>(2);
        WaitFlag<HardEvent::S_MTE3>(2);
        DataCopyExtParams copyOutShapeParams{1, 2 * sizeof(uint64_t), 0, 0, 0};
        DataCopyPad(this->outShapeGm, y1ShapeUb, copyOutShapeParams);
        DataCopyPad(this->outShapeGm[SHAPE1_GM_IDX], y2ShapeUb, copyOutShapeParams);
    }

private:
    const BroadcastGradientArgsTilingData* tilingData_;
    int64_t alignT = 0;
    uint16_t vlInput = 0;
    uint16_t vlInt64 = 0;
    uint16_t vLoopT = 0;
    uint16_t vLoopInt64 = 0;
};
} // namespace BroadcastGradientArgs
#endif // BROADCAST_GRADIENT_ARGS_PERF_H_
