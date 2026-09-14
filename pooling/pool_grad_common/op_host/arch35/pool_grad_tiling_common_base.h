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
 * \file pool_grad_tiling_common_base.h
 * \brief 池化反向算子 NCHW/NHWC 通用 tiling 流程基类: 承载公共成员与公共流程方法
 *        (IsMeetUBSize / DynamicAdjustmentWH / DoOpTiling / 基础硬件信息初始化),
 *        差异环节 (DoBufferCalculate / DoUBTiling / DoBlockTiling / SetTilingData /
 *        Print* ) 由派生类实现。CPU 侧代码, 虚函数开销可忽略。
 */

#ifndef POOL_GRAD_TILING_COMMON_BASE_H_
#define POOL_GRAD_TILING_COMMON_BASE_H_

#include <algorithm>

#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "pool_grad_tiling_split_helper.h"

namespace optiling {

template <typename BaseInfoT, typename SplitInfoT, typename InputInfoT>
class PoolGradTilingCommonBase {
public:
    bool IsMeetUBSize()
    {
        DoBufferCalculate();
        return splitData.totalBufferSize <= baseData.availableUb;
    }

    void DynamicAdjustmentWH()
    {
        if (splitData.hOutputInner == 1) {
            splitData.wOutputOuter++;
            splitData.wOutputInner = Ops::Base::CeilDiv(inputData->wX, splitData.wOutputOuter);
        } else {
            splitData.hOutputOuter++;
            splitData.hOutputInner = Ops::Base::CeilDiv(inputData->hX, splitData.hOutputOuter);
        }
    }

    ge::graphStatus DoOpTiling(gert::TilingContext* context, uint64_t key)
    {
        DoUBTiling();
        DoBlockTiling();
        SetTilingData(context, key);
        PrintBaseData();
        PrintSplitData();
        return ge::GRAPH_SUCCESS;
    }

protected:
    PoolGradTilingCommonBase(InputInfoT* input) : inputData(input) {}
    ~PoolGradTilingCommonBase() = default;

    // 基础硬件/字节信息初始化 (NCHW 与 NHWC 公共部分)
    void InitCommonBaseInfo(gert::TilingContext* context, int64_t ubSize, int64_t coreNum)
    {
        baseData.vRegSize = Ops::Base::GetVRegSize(context);
        baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context);
        baseData.inputBytes = inputData->inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
        baseData.indexBytes = inputData->indexDtype == ge::DT_INT32 ? INT32_SIZE : INT64_SIZE;
        baseData.availableUb = ubSize - UB_RESVERVED_SIZE;
        baseData.totalCoreNum = coreNum;
        baseData.coreUsedForBestPerformance = baseData.totalCoreNum;

        int64_t oneBlockNumT1 = baseData.ubBlockSize / baseData.inputBytes;
        int64_t oneBlockNumT2 = baseData.ubBlockSize / baseData.indexBytes;

        baseData.maxDataNumInOneBlock = std::max(oneBlockNumT1, oneBlockNumT2);
        baseData.proDataNumInOneBeatT2 = baseData.vRegSize / baseData.ubBlockSize * oneBlockNumT2;
    }

    // 窗口重叠时的批处理大小与 overlap 标记 (NCHW 与 NHWC 公共部分)
    void InitOverlapBatchInfo(int64_t hKernel, int64_t wKernel, int64_t hStride, int64_t wStride)
    {
        baseData.hProBatchSize = PoolGradTiling::CalcProBatchSize(hKernel, hStride);
        baseData.wProBatchSize = PoolGradTiling::CalcProBatchSize(wKernel, wStride);
        baseData.isOverlap = (baseData.hProBatchSize != 1 || baseData.wProBatchSize != 1) ? 1 : 0;
    }

    virtual void DoBufferCalculate() = 0;
    virtual void DoUBTiling() = 0;
    virtual void DoBlockTiling() = 0;
    virtual void SetTilingData(gert::TilingContext* context, uint64_t key) = 0;
    virtual void PrintBaseData() const = 0;
    virtual void PrintSplitData() const = 0;

    static constexpr int64_t FLOAT16_SIZE = 2;
    static constexpr int64_t FLOAT32_SIZE = 4;
    static constexpr int64_t INT32_SIZE = 4;
    static constexpr int64_t INT64_SIZE = 8;
    static constexpr int64_t UB_RESVERVED_SIZE = 1024;

    BaseInfoT baseData;
    SplitInfoT splitData;
    InputInfoT* inputData;
};

} // namespace optiling
#endif // POOL_GRAD_TILING_COMMON_BASE_H_
