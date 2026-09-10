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
 * \file pool_grad_tiling_split_helper.h
 * \brief 池化反向算子 (MaxPoolGrad / AvgPoolV2Grad 系列) op_host 侧共用的
 *        UB 切分搜索 (TrySplit* / SplitUnalign*) 与核间分块 (DoBlockTiling) 模板实现。
 *        调用方将各自 InputInfo/BaseInfo 的维度与参数归一为标量/维度结构体传入,
 *        切分判定与动态调整通过 TilingClass 模板参数直接调用 tiling 类的
 *        IsMeetUBSize() / IsMeetTargetCoreNum() / DynamicAdjustmentWH() 成员,
 *        各算子的 tiling 类保持自身流程不变。
 *        约定 TilingClass 需公开上述三个成员函数。
 */

#ifndef POOL_GRAD_TILING_SPLIT_HELPER_H_
#define POOL_GRAD_TILING_SPLIT_HELPER_H_

#include <algorithm>
#include <cstdint>

#include "util/math_util.h"

namespace optiling {
namespace PoolGradTiling {

// NCHW 切分所需维度 (hX/wX 为输出 H/W; highAxis 为 NC 合并轴)
struct PoolGradNchwDims {
    int64_t hX;
    int64_t wX;
    int64_t hStride;
    int64_t wStride;
};

// NHWC 切分所需维度 (hX/wX/cX 为输入 H/W/C; nX 为 batch)
struct PoolGradNhwcDims {
    int64_t nX;
    int64_t cX;
    int64_t hX;
    int64_t wX;
    int64_t hStride;
    int64_t wStride;
};

/*!
 * \brief 按内层块大小计算某轴的 outer 与 tail: outer = CeilDiv(total, inner),
 *        tail 为最后一个不满块大小, 整除时为 inner。
 */
inline void CalcAxisOuterTail(int64_t totalSize, int64_t innerSize, int64_t& outerSize, int64_t& tailSize)
{
    outerSize = Ops::Base::CeilDiv(totalSize, innerSize);
    int64_t tempTail = totalSize % innerSize;
    tailSize = tempTail == 0 ? innerSize : tempTail;
}

/*!
 * \brief 计算池化窗口重叠时的批处理大小: kernel > stride 时为 CeilDiv(kernel, stride), 否则为 1。
 */
inline int64_t CalcProBatchSize(int64_t kernelSize, int64_t strideSize)
{
    if (kernelSize > strideSize) {
        return Ops::Base::CeilDiv(kernelSize, strideSize);
    }
    return 1;
}

/*!
 * \brief 在 [1, right] 内二分查找最大的 mid 使 splitData.*field = mid * multiplier 后判定通过。
 *        约定 mid = 1 已由调用方验证可行 (bestSplit 初值 1)。
 *        判定为 IsMeetUBSize() && IsMeetTargetCoreNum(), 与 TrySplit* 系列一致;
 *        checkCoreNum = false 时仅判定 IsMeetUBSize() (NHWC C 轴兜底阶段使用)。
 */
template <typename SplitInfo, typename TilingClass, typename MemberPtrT>
int64_t SearchMaxSplit(TilingClass& tiling, SplitInfo& splitData, MemberPtrT field, int64_t multiplier, int64_t right,
                       bool checkCoreNum = true)
{
    int64_t left = 1;
    int64_t bestSplit = 1;

    while (left <= right) {
        int64_t mid = left + (right - left) / 2;

        splitData.*field = mid * multiplier;
        if (tiling.IsMeetUBSize() && (!checkCoreNum || tiling.IsMeetTargetCoreNum())) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return bestSplit * multiplier;
}

// ---------------- NCHW ----------------

template <typename SplitInfo>
bool IsMeetTargetCoreNumNchw(const SplitInfo& splitData, const PoolGradNchwDims& d, int64_t inputNCSize,
                             int64_t coreUsedForBestPerformance)
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(d.wX, splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(d.hX, splitData.hOutputInner);
    int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(inputNCSize, splitData.highAxisInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >= coreUsedForBestPerformance;
}

/*!
 * \brief NCHW: 尝试沿 NC 合并轴整切, 失败后在 [1, inputNCSize] 二分。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitNc(SplitInfo& splitData, const PoolGradNchwDims& d, int64_t inputNCSize,
                int64_t coreUsedForBestPerformance, TilingClass& tiling)
{
    splitData.wOutputInner = d.wX;
    splitData.hOutputInner = d.hX;

    splitData.highAxisInner = Ops::Base::CeilDiv(inputNCSize, coreUsedForBestPerformance);
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.highAxisInner = 1;
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.highAxisInner = SearchMaxSplit(tiling, splitData, &SplitInfo::highAxisInner, 1, inputNCSize);
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief NCHW: 无 pad 无 overlap 时按 hStride 对齐二分切 H。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitAlignH(SplitInfo& splitData, const PoolGradNchwDims& d, TilingClass& tiling)
{
    splitData.wOutputInner = d.wX;
    splitData.hOutputInner = d.hStride;

    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.hOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::hOutputInner, d.hStride,
                                                Ops::Base::CeilDiv(d.hX / 2, d.hStride));
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief NCHW: 无 pad 无 overlap 时按 wStride 对齐二分切 W。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitAlignW(SplitInfo& splitData, const PoolGradNchwDims& d, TilingClass& tiling)
{
    splitData.hOutputInner = d.hStride;
    splitData.wOutputInner = d.wStride;

    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.wOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::wOutputInner, d.wStride,
                                                Ops::Base::CeilDiv(d.wX / 2, d.wStride));
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief NCHW: 非对齐兜底切分, 不满足条件时动态调整 W/H 直至满足或 H 切满。
 */
template <typename SplitInfo, typename TilingClass>
void SplitUnalignHw(SplitInfo& splitData, const PoolGradNchwDims& d, int64_t isPad, int64_t isOverlap,
                    int64_t proDataNumInOneBeat, TilingClass& tiling)
{
    splitData.highAxisInner = 1;
    if (isPad == 0 && isOverlap == 0) {
        splitData.hOutputInner = d.hStride;
        splitData.wOutputInner = d.wStride;
    } else {
        splitData.hOutputInner = d.hX;
        splitData.wOutputInner = d.wX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(d.wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(d.hX, splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner > proDataNumInOneBeat) {
        if (!tiling.IsMeetTargetCoreNum() || !tiling.IsMeetUBSize()) {
            tiling.DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    splitData.wOutputInner = std::min(d.wX, proDataNumInOneBeat);
    return;
}

template <typename SplitInfo>
void DoBlockTilingNchw(SplitInfo& splitData, int64_t totalCoreNum)
{
    splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

// ---------------- NHWC ----------------

template <typename SplitInfo>
bool IsMeetTargetCoreNumNhwc(const SplitInfo& splitData, const PoolGradNhwcDims& d, int64_t coreUsedForBestPerformance)
{
    // The calculation only involves inner.
    int64_t tmpWOutputOuter = Ops::Base::CeilDiv(d.wX, splitData.wOutputInner);
    int64_t tmpHOutputOuter = Ops::Base::CeilDiv(d.hX, splitData.hOutputInner);
    int64_t tmpNOutputOuter = Ops::Base::CeilDiv(d.nX, splitData.nOutputInner);
    int64_t tmpCOutputOuter = Ops::Base::CeilDiv(d.cX, splitData.cOutputInner);

    return tmpWOutputOuter * tmpHOutputOuter * tmpNOutputOuter * tmpCOutputOuter >= coreUsedForBestPerformance;
}

/*!
 * \brief NHWC: 尝试沿 N 轴整切, 失败后在 [1, nX] 二分。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitN(SplitInfo& splitData, const PoolGradNhwcDims& d, int64_t coreUsedForBestPerformance, TilingClass& tiling)
{
    splitData.wOutputInner = d.wX;
    splitData.hOutputInner = d.hX;
    splitData.cOutputInner = d.cX;

    splitData.nOutputInner = Ops::Base::CeilDiv(d.nX, coreUsedForBestPerformance);
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        return true;
    }

    splitData.nOutputInner = 1;
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.nOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::nOutputInner, 1, d.nX);
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief NHWC: 无 pad 无 overlap 时按 hStride 对齐二分切 H。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitAlignH(SplitInfo& splitData, const PoolGradNhwcDims& d, TilingClass& tiling)
{
    splitData.nOutputInner = 1;
    splitData.wOutputInner = d.wX;
    splitData.cOutputInner = d.cX;

    splitData.hOutputInner = d.hStride;
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.hOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::hOutputInner, d.hStride,
                                                Ops::Base::CeilDiv(d.hX / 2, d.hStride));
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief NHWC: 无 pad 无 overlap 时按 wStride 对齐二分切 W。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitAlignW(SplitInfo& splitData, const PoolGradNhwcDims& d, TilingClass& tiling)
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = d.hStride;
    splitData.cOutputInner = d.cX;

    splitData.wOutputInner = d.wStride;
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.wOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::wOutputInner, d.wStride,
                                                Ops::Base::CeilDiv(d.wX / 2, d.wStride));
        return true;
    } else {
        return false;
    }
}

/*!
 * \brief NHWC: 无 pad 无 overlap 时按 cacheLine 对齐二分切 C。
 */
template <typename SplitInfo, typename TilingClass>
bool TrySplitAlignC(SplitInfo& splitData, const PoolGradNhwcDims& d, int64_t moveDataNumCacheLine, TilingClass& tiling)
{
    splitData.nOutputInner = 1;
    splitData.hOutputInner = d.hStride;
    splitData.wOutputInner = d.wStride;

    int64_t tmpCAligned = d.cX < moveDataNumCacheLine ? d.cX : moveDataNumCacheLine;
    splitData.cOutputInner = tmpCAligned;
    if (tiling.IsMeetUBSize() && tiling.IsMeetTargetCoreNum()) {
        splitData.cOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::cOutputInner, moveDataNumCacheLine,
                                                Ops::Base::CeilDiv(d.cX / 2, moveDataNumCacheLine));
        return true;
    } else {
        // hw stride 较大场景 或者 nhwc超小场景  ---> 应该对hw做更小的切分
        return false;
    }
}

/*!
 * \brief NHWC: 非对齐兜底切分; H/W 调整至 1 后再对 C 二分 (C 轴阶段仅判定 IsMeetUBSize)。
 */
template <typename SplitInfo, typename TilingClass>
void SplitUnalignHwc(SplitInfo& splitData, const PoolGradNhwcDims& d, int64_t isPad, int64_t isOverlap,
                     int64_t moveDataNumCacheLine, int64_t proDataNumInOneBeat, TilingClass& tiling)
{
    splitData.nOutputInner = 1;
    if (isPad == 0 && isOverlap == 0) {
        splitData.hOutputInner = d.hStride;
        splitData.wOutputInner = d.wStride;
        int64_t tmpCAligned = d.cX < moveDataNumCacheLine ? d.cX : moveDataNumCacheLine;
        splitData.cOutputInner = tmpCAligned;
    } else {
        splitData.wOutputInner = d.wX;
        splitData.hOutputInner = d.hX;
        splitData.cOutputInner = d.cX;
    }

    splitData.wOutputOuter = Ops::Base::CeilDiv(d.wX, splitData.wOutputInner);
    splitData.hOutputOuter = Ops::Base::CeilDiv(d.hX, splitData.hOutputInner);

    while (splitData.hOutputInner != 1 || splitData.wOutputInner != 1) {
        if (!tiling.IsMeetTargetCoreNum() || !tiling.IsMeetUBSize()) {
            tiling.DynamicAdjustmentWH();
        } else {
            return;
        }
    }

    // NHW全切为1  C 超大场景 或者 NHW超小场景
    if (d.cX <= proDataNumInOneBeat) {
        return;
    } else if (tiling.IsMeetUBSize()) {
        splitData.cOutputInner = proDataNumInOneBeat;
        return;
    } else {
        splitData.cOutputInner = SearchMaxSplit(tiling, splitData, &SplitInfo::cOutputInner, proDataNumInOneBeat,
                                                Ops::Base::CeilDiv(d.cX / 2, proDataNumInOneBeat), false);
        return;
    }
}

template <typename SplitInfo>
void DoBlockTilingNhwc(SplitInfo& splitData, int64_t totalCoreNum)
{
    splitData.totalBaseBlockNum = splitData.nOutputOuter * splitData.cOutputOuter * splitData.hOutputOuter *
                                  splitData.wOutputOuter;
    splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, totalCoreNum);
    splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
    splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                   splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
}

} // namespace PoolGradTiling
} // namespace optiling

#endif // POOL_GRAD_TILING_SPLIT_HELPER_H_
