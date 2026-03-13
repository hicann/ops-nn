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
 * \file broadcast_gradient_args_scalar.h
 * \brief
 */

#ifndef BROADCAST_GRADIENT_ARGS_SCALAR_H_
#define BROADCAST_GRADIENT_ARGS_SCALAR_H_
#include "broadcast_gradient_args_base.h"

namespace BroadcastGradientArgs {
using namespace AscendC;

template <typename T>
class BroadcastGradientArgsScalar : public BroadcastGradientArgsBase<T>
{
public:
    __aicore__ inline BroadcastGradientArgsScalar(){};

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, GM_ADDR outShape,
        const BroadcastGradientArgsTilingData* __restrict tilingData)
    {
        // init global memory
        tilingData_ = tilingData;
        this->x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        this->x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        this->y1Gm.SetGlobalBuffer((__gm__ T*)y1);
        this->y2Gm.SetGlobalBuffer((__gm__ T*)y2);
        this->outShapeGm.SetGlobalBuffer((__gm__ uint64_t*)outShape);
    }

    __aicore__ inline void Process()
    {
        this->outShapeGm.SetValue(FIRST_OUTPUT_DIMS_IDX, FIRST_UINT64_SHAPE_DIM_ONE);
        this->outShapeGm.SetValue(SECOND_OUTPUT_DIMS_IDX, 1);
        this->outShapeGm.SetValue(FIRST_OUTPUT_DIM0_IDX, 0);
        this->outShapeGm.SetValue(SECOND_OUTPUT_DIM0_IDX, 0);

        bool invalid_flag = false;
        bool equal_flag = true;
        bool now_equal_flag = true;
        for (int64_t i = 0; i < tilingData_->maxRank; i++) {
            if ((i >= tilingData_->x1Len) || (i >= tilingData_->x2Len)) {
                break;
            }
            int64_t x1Value = this->x1Gm.GetValue(tilingData_->x1Len - 1 - i);
            int64_t x2Value = this->x2Gm.GetValue(tilingData_->x2Len - 1 - i);
            now_equal_flag = x1Value == x2Value;
            equal_flag = equal_flag && now_equal_flag;
            if ((!now_equal_flag) && (x1Value != 1) && (x2Value != 1)) {
                invalid_flag = true;
                break;
            }
        }

        if (tilingData_->x1Len == tilingData_->x2Len && equal_flag) {
            AscendC::DataCacheCleanAndInvalid<
                uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(this->outShapeGm);
            return;
        }

        if (invalid_flag) {
            assert(!invalid_flag, "Inputs x1 and x2 do not satisfy broadcasting rules !\n");
            return;
        }

        int64_t x1Offset = tilingData_->maxRank - tilingData_->x1Len;
        int64_t x2Offset = tilingData_->maxRank - tilingData_->x2Len;
        int64_t y1Index = 0;
        int64_t y2Index = 0;
        T y1Value;
        T y2Value;
        for (int64_t i = 0; i < tilingData_->maxRank; i++) {
            if (i < x1Offset) {
                y1Value = 1;
            } else {
                y1Value = this->x1Gm.GetValue(i - x1Offset);
            }
            if (i < x2Offset) {
                y2Value = 1;
            } else {
                y2Value = this->x2Gm.GetValue(i - x2Offset);
            }
            if (y1Value == 1) {
                this->y1Gm.SetValue(y1Index, i);
                y1Index++;
            }
            if (y2Value == 1) {
                this->y2Gm.SetValue(y2Index, i);
                y2Index++;
            }
        }

        this->outShapeGm.SetValue(FIRST_OUTPUT_DIM0_IDX, y1Index);
        this->outShapeGm.SetValue(SECOND_OUTPUT_DIM0_IDX, y2Index);
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
            this->y1Gm);
        AscendC::DataCacheCleanAndInvalid<T, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(
            this->y2Gm);
        AscendC::DataCacheCleanAndInvalid<
            uint64_t, AscendC::CacheLine::ENTIRE_DATA_CACHE, AscendC::DcciDst::CACHELINE_OUT>(this->outShapeGm);
    }

private:
    const BroadcastGradientArgsTilingData* tilingData_;
};
} // namespace BroadcastGradientArgs
#endif // BROADCAST_GRADIENT_ARGS_SCALAR_H_