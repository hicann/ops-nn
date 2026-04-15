/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_filter_v2.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_FILTER_H
#define CONV3D_BACKPROP_FILTER_H

#include "basic_api/kernel_basic_intf.h"
#include "conv3d_bp_filter.h"
#include "kernel_type.h"
#include "lib/matmul_intf.h"
#include "conv3d_backprop_filter_v2_tiling_data.h"

namespace AscendC {
using Conv3ddwConfig = typename ConvolutionBackprop::Conv3ddwConfig;

__aicore__ inline constexpr ConvolutionBackprop::CubeFormat GetFormat(int format)
{
    if (format == FORMAT_NDHWC || format == FORMAT_NHWC) {
        return ConvolutionBackprop::CubeFormat::NDHWC;
    } else if (format == FORMAT_DHWCN || format == FORMAT_HWCN) {
        return ConvolutionBackprop::CubeFormat::DHWCN;
    } else {
        return ConvolutionBackprop::CubeFormat::NCDHW;
    }
}

template <typename xType, int xFormat, typename dedyType, int dedyFormat, typename yType, int yFormat,
    bool isSplitKernelHW, uint32_t l0cCondition = TPL_STREAM_DFL>
class Conv3dDw {
public:
    __aicore__ inline Conv3dDw()
    {
    };

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dedy, GM_ADDR y, GM_ADDR workSpace,
        const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
    }

    __aicore__ inline void Process()
    {
    }

protected:
    static constexpr ConvolutionBackprop::CubeFormat xCubeFormat = GetFormat(xFormat);
    static constexpr ConvolutionBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr ConvolutionBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    using xDwType = ConvolutionBackprop::ConvType<TPosition::GM, xCubeFormat, xType>;
    using filterSizeDwType = ConvolutionBackprop::ConvType<TPosition::GM, ConvolutionBackprop::CubeFormat::ND, int32_t>;
    using dedyDwType = ConvolutionBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDwType = ConvolutionBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    static constexpr Conv3ddwConfig conv3ddwConfig = {isSplitKernelHW, l0cCondition};
    ConvolutionBackprop::Conv3DBackpropFilter<xDwType, filterSizeDwType, dedyDwType, yDwType, conv3ddwConfig> dw_;
    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<float> workspaceGm_;
    const conv_bp_v2_kernel::TConv3DDwTiling* tiling_;
    uint64_t offsetA_ = 0;
    uint64_t offsetB_ = 0;
    uint64_t offsetC_ = 0;
    uint32_t mCoreIndx_ = 0;
    uint32_t nCoreIndx_ = 0;
    uint32_t kCoreIndx_ = 0;
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t k_ = 0;
    uint64_t singleShapeBatch_ = 0;
    uint64_t singleShapeNInCurrentHo_ = 0;
    uint64_t singleShapeMInCurrentHo_ = 0;
    uint32_t singleShapeCin_ = 0;
    uint64_t singleShapeM_ = 0;
    uint64_t singleShapeN_ = 0;
    uint32_t cinCoreIndx_ = 0;
    uint32_t tailCinG_ = 0;
    uint32_t tailCoutG_ = 0;
    bool groupFlag_ = 0;
    bool seperateDk_ = true;
    uint64_t offsetCubeWorkSpaceC_ = 0;

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        tiling_ = &(tilingData->dwTiling);
        singleShapeM_ = tilingData->basicBlockTiling.singleCoreM;
        singleShapeN_ = tilingData->basicBlockTiling.singleCoreN;

        k_ = static_cast<uint64_t>(tiling_->ho) * tiling_->wo;
        groupFlag_ = tiling_->group > 1;

        // TilingData init different for 1971 and 1982
#if defined(__DAV_C310__)
        // 判断条件为是否为分组卷积，而不是是否cin和dk分轴
        if (groupFlag_) {
            m_ = tiling_->cout1G;
            n_ = static_cast<uint64_t>(tiling_->cin1G) * tiling_->hk * tiling_->wk;
            // cinG等于cin1g乘以c0，cin1g等于扩维后的cin除以c0，又因为扩维系数必定对齐C0，因此，cinG就是扩维后的每组cin个数
            tailCinG_ = tiling_->cin - tiling_->cin1G * (tiling_->realGroup - 1);
            tailCoutG_ = tiling_->cout - tiling_->cout1G * (tiling_->realGroup - 1);
        } else {
            m_ = tiling_->cout;
            n_ = tiling_->cin * tiling_->hk * tiling_->wk;
            if (!seperateDk_) {
                n_ *= tiling_->dk;
            }
        }
#endif
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    __aicore__ inline void CalcBlockOffsetA(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx, uint64_t hoCal)
    {
        uint64_t groupOffsetA = 0;
        uint64_t hoOffset = 0;
        uint64_t doOffset = 0;
        uint64_t coutOffset = 0;
        uint64_t batchOffsetA = 0;
        if constexpr (dedyCubeFormat == ConvolutionBackprop::CubeFormat::NCDHW) {
            if (groupFlag_) {
                // group将cin和cout分为多个组，每轮处理一组cin和cout，因此，每轮group的偏移计算仅算分组扩维后的cinG或coutG
                groupOffsetA = static_cast<uint64_t>(groupIdx) * tiling_->cout1G * tiling_->dout * tiling_->ho * tiling_->wo;
            }
            hoOffset = hoCal * tiling_->wo;
            doOffset = static_cast<uint64_t>(doutIdx) * tiling_->ho * tiling_->wo;
            coutOffset = static_cast<uint64_t>(mCoreIndx_) * tiling_->singleCoreCout * tiling_->dout * tiling_->ho * tiling_->wo;
            batchOffsetA = static_cast<uint64_t>(batchIdx) * tiling_->cout * tiling_->dout * tiling_->ho * tiling_->wo;
        } else {
            // ndhwc, cout is last axis
            if (groupFlag_) {
                // group将cin和cout分为多个组，每轮处理一组cin和cout，因此，每轮group的偏移计算仅算分组扩维后的cinG或coutG
                groupOffsetA = static_cast<uint64_t>(groupIdx) * tiling_->cout1G;
            }
            hoOffset = hoCal * tiling_->wo * tiling_->cout;
            doOffset = static_cast<uint64_t>(doutIdx) * tiling_->ho * tiling_->wo * tiling_->cout;
            coutOffset = static_cast<uint64_t>(mCoreIndx_) * tiling_->singleCoreCout;
            batchOffsetA = static_cast<uint64_t>(batchIdx) * tiling_->dout * tiling_->ho * tiling_->wo * tiling_->cout;
        }
        offsetA_ = batchOffsetA + groupOffsetA + coutOffset + doOffset + hoOffset;
    }

    __aicore__ inline void CalcBlockOffsetB(uint64_t batchIdx, uint64_t doutIdx,
        uint32_t groupIdx, uint32_t dkIdx, uint64_t hoCal, uint64_t& dinCurIdx)
    {
        uint64_t groupOffsetB = 0;
        dinCurIdx = static_cast<uint64_t>(doutIdx) * tiling_->strideD;
        uint64_t dinIdx = static_cast<uint64_t>(dkIdx) * tiling_->dilationD;
        // 当dilationD_大于1或者分组卷积时，cin与dk分轴，dinIdx无需考虑pad，使用seperateDk_作为判断条件
        if (seperateDk_) {
            dinCurIdx += dinIdx;
            dinIdx = dinCurIdx - tiling_->padFront;
        } else {
            if (dinCurIdx > tiling_->padFront) {
                dinIdx += dinCurIdx - tiling_->padFront;
            }
        }

        uint64_t hiIdx = 0;
        if (hoCal * tiling_->strideH > tiling_->padUp) {
            hiIdx = hoCal * tiling_->strideH - tiling_->padUp;
        }

        uint64_t hiOffset = 0;
        uint64_t dinDkOffset = 0;
        uint64_t cinOffset = 0;
        uint64_t batchOffsetB = 0;
        if constexpr (xCubeFormat == ConvolutionBackprop::CubeFormat::NCDHW) {
            if (groupFlag_) {
                groupOffsetB = static_cast<uint64_t>(groupIdx) * tiling_->cin1G * tiling_->di * tiling_->hi * tiling_->wi;
            }
            hiOffset = hiIdx * tiling_->wi;
            dinDkOffset = dinIdx * tiling_->hi * tiling_->wi;
            //分组卷积时，需注意nCoreIndx_就是cinCoreIndx_，其他情况nCoreIndx_为cinIndx和dkIndx
            cinOffset = static_cast<uint64_t>(cinCoreIndx_) * tiling_->singleCoreCin * tiling_->di * tiling_->hi * tiling_->wi;
            batchOffsetB = static_cast<uint64_t>(batchIdx) * tiling_->cin * tiling_->di * tiling_->hi * tiling_->wi;
        } else {
            if (groupFlag_) {
                groupOffsetB = static_cast<uint64_t>(groupIdx) * tiling_->cin1G;
            }
            hiOffset = hiIdx * tiling_->wi * tiling_->cin;
            dinDkOffset = dinIdx * tiling_->hi * tiling_->wi * tiling_->cin;
            //分组卷积时，需注意nCoreIndx_就是cinCoreIndx_，其他情况nCoreIndx_为cinIndx和dkIndx
            cinOffset = static_cast<uint64_t>(cinCoreIndx_) * tiling_->singleCoreCin;
            batchOffsetB = static_cast<uint64_t>(batchIdx) * tiling_->di * tiling_->hi * tiling_->wi * tiling_->cin;
        }
        offsetB_ = batchOffsetB + groupOffsetB + cinOffset + dinDkOffset + hiOffset;
    }

    __aicore__ inline void CalcBlockOffsetC(uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx, uint64_t dinCurIdx)
    {
        uint64_t groupOffsetC = 0;
        uint64_t coutOffsetC = 0;
        if constexpr (yCubeFormat == ConvolutionBackprop::CubeFormat::DHWCN) {
            coutOffsetC = mCoreIndx_ * tiling_->singleCoreCout;
        } else {
            coutOffsetC = mCoreIndx_ * tiling_->singleCoreCout * tiling_->cin * tiling_->dk * tiling_->hk * tiling_->wk;
        }

        // 判断条件为是否为分组卷积
        if (groupFlag_) {
            if (yCubeFormat == ConvolutionBackprop::CubeFormat::DHWCN) {
                // 对于输出位DHWCN的C矩阵，GM的排布为group dk hk wk cin/group cout/group
                groupOffsetC = static_cast<uint64_t>(groupIdx) * tiling_->cout1G;
            } else {
                // C矩阵的GM排布为group cout/group cin/group dk hk wk，而每轮处理的group都是经过扩维的
                // 因此可以将enlarge_看成扩维前singleCoreGroup，而enlarge * cout/group等于coutG_，因此偏移计算如下
                // group与cout绑定，而ncdhw和ndhwc格式的排布cout并未发生变化，因此groupOffsetC和coutOffsetC都无需发生变化
                groupOffsetC = static_cast<uint64_t>(groupIdx) * tiling_->cout1G * (tiling_->cin / tiling_->group) * tiling_->dk * tiling_->hk * tiling_->wk;
                // 当分组卷积时，C矩阵的shape由cout cin dk hk wk变为cout cin/group dk hk wk，因此offset计算偏移也发生变化
                coutOffsetC = static_cast<uint64_t>(mCoreIndx_) * tiling_->singleCoreCout * (tiling_->cin / tiling_->group) * tiling_->dk * tiling_->hk * tiling_->wk;
            }
        }

        if constexpr (yCubeFormat == ConvolutionBackprop::CubeFormat::NCDHW) {
            uint64_t cinDkOffset = cinCoreIndx_ * tiling_->singleCoreCin * tiling_->dk * tiling_->hk * tiling_->wk + dkIdx * tiling_->hk * tiling_->wk;
            offsetC_ = groupOffsetC + cinDkOffset + coutOffsetC;
            if (dinCurIdx < tiling_->padFront && !seperateDk_) {
                offsetC_ += static_cast<uint64_t>(tiling_->padFront - dinCurIdx) * tiling_->hk * tiling_->wk;


            }
        } else if constexpr (yCubeFormat == ConvolutionBackprop::CubeFormat::NDHWC) {
            // NDHWC格式，C在最内轴，考虑到group，需要使用cinG计算偏移
            uint64_t cinDkOffset = static_cast<uint64_t>(cinCoreIndx_) * tiling_->singleCoreCin + dkIdx * tiling_->hk * tiling_->wk * tiling_->cin1G;
            offsetC_ = groupOffsetC + cinDkOffset + coutOffsetC;
            if (dinCurIdx < tiling_->padFront && !seperateDk_) {
                offsetC_ += static_cast<uint64_t>(tiling_->padFront - dinCurIdx) * tiling_->hk * tiling_->wk * tiling_->cin1G;
            }
        } else {
            // DHWCN
            uint64_t cinDkOffset = cinCoreIndx_ * tiling_->singleCoreCin * tiling_->cout + dkIdx * tiling_->hk * tiling_->wk * (tiling_->cin / tiling_->group) * tiling_->cout;
            offsetC_ = groupOffsetC + cinDkOffset + coutOffsetC;
            if (dinCurIdx < tiling_->padFront && !seperateDk_) {
                offsetC_ += (tiling_->padFront - dinCurIdx) * tiling_->hk * tiling_->wk * (tiling_->cin / tiling_->group) * tiling_->cout;
            }
        }
    }

    __aicore__ inline void CalcOffset(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx, uint64_t hoCal)
    {
        CalcBlockOffsetA(batchIdx, doutIdx, groupIdx, hoCal);
        uint64_t dinCurIdx = 0;
        CalcBlockOffsetB(batchIdx, doutIdx, groupIdx, dkIdx, hoCal, dinCurIdx);
        CalcBlockOffsetC(doutIdx, groupIdx, dkIdx, dinCurIdx);
    }

    __aicore__ inline void ReCalcSingleCoreBatchDout(uint64_t doutIdx, uint32_t dkIdx, bool& invalidDIndex)
    {
        for (uint64_t i = 0; i < singleShapeBatch_; i++) {
            uint64_t doutCurIdx = (doutIdx + i) % tiling_->dout;
            uint64_t dinCurIdx = doutCurIdx * tiling_->strideD + dkIdx * tiling_->dilationD;
            if ((dinCurIdx >= tiling_->padFront) && (dinCurIdx < (tiling_->di + tiling_->padFront))) {
                invalidDIndex = false;
                return;
            }
        }
    }

    __aicore__ inline void ReCalDkCinSingleCoreShape(uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx)
    {
        singleShapeNInCurrentHo_ = singleShapeN_;
        singleShapeMInCurrentHo_ = singleShapeM_;
        uint64_t dinIdx = doutIdx * tiling_->strideD;

        if (seperateDk_) {
            bool invalidDIndex = true;
            ReCalcSingleCoreBatchDout(doutIdx, dkIdx, invalidDIndex);
            if (invalidDIndex) {
                singleShapeNInCurrentHo_ = 0;
                return;
            }
            // 若group大于1，还需判断分组的cin和cout尾块
            if (groupFlag_) {
                ReCalSingleShapeWithGroup(groupIdx);
            }
            // 由于cin和dk分轴，每轮singlecoreDk都为1，因此无需使用下面复杂的代码来计算实际din应当载入多少
            return;
        }
        // 当D轴有pad，且nCoreIndx为dkIndx时，仅nCoreIndx为0时singleShapeNInCurrentHo不为0，即dk核仅0核运行，其他核不运行
        // 此时，singleCoreDk失效，新的singleCoreDk为newDk
        // 当D轴有pad，且nCoreIndx为cinIndx时，所有核的singleShapeNInCurrentHo均不为0，即所有核都参与计算，
        // 此时，singleCoreDk失效，新的singleCoreDk为newDk
        // 当D轴有pad，nCoreIndx既包含cinIndx又包含dkIndx时，当nCoreIndx小于cinIndx时，所有核参与运行，
        // 当nCoreIndx等于cinIndx, 此核为0核，当nCoreIndx大于cinIndx，所有核均不运行
        uint64_t newDk = tiling_->dk;
        if (dinIdx + newDk > tiling_->padFront + tiling_->di) {
            if (dinIdx < tiling_->padFront + tiling_->di) {
                newDk = tiling_->padFront + tiling_->di - dinIdx;
                if (n_ / tiling_->dk * newDk < nCoreIndx_ * tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                }
                uint64_t nRamin = n_ / tiling_->dk * newDk - nCoreIndx_ * tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk;
                singleShapeNInCurrentHo_ =
                    nRamin < tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk ? nRamin : tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk;
            } else {
                singleShapeNInCurrentHo_ = 0;
            }
        }

        if (dinIdx < tiling_->padFront) {
            if (dinIdx + newDk > tiling_->padFront) {
                newDk = dinIdx + newDk - tiling_->padFront;
                if (n_ / tiling_->dk * newDk < nCoreIndx_ * tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                }
                uint64_t nRamin = n_ / tiling_->dk * newDk - nCoreIndx_ * tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk;
                singleShapeNInCurrentHo_ =
                    nRamin < tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk ? nRamin : tiling_->singleCoreCin * tiling_->hk * tiling_->wk * newDk;
            } else {
                singleShapeNInCurrentHo_ = 0;
            }
        }
    }
#endif

    __aicore__ inline void ReCalSingleShapeWithGroup(uint32_t groupIdx)
    {
        if (groupIdx == (tiling_->realGroup - 1)) {
            if (tailCinG_ < tiling_->cin1G) {
                uint64_t tailN = tailCinG_ * tiling_->hk * tiling_->wk;
                uint64_t singleCoreN = tiling_->singleCoreCin * tiling_->hk * tiling_->wk;
                uint64_t currentN = nCoreIndx_ * singleCoreN;
                if (tailN <= currentN) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                } else {
                    uint64_t nRemain = tailN - currentN;
                    uint32_t cinRemain = tailCinG_ - nCoreIndx_ * tiling_->singleCoreCin;
                    singleShapeNInCurrentHo_ = nRemain < singleCoreN ? nRemain : singleCoreN;
                    // 310时，每个核的cin大小是由singleShapeCin_传递，因此，需计算singleShapeCin_
                    singleShapeCin_ = cinRemain < tiling_->singleCoreCin ? cinRemain : tiling_->singleCoreCin;
                }
            }

            if (tailCoutG_ < tiling_->cout1G) {
                uint32_t tailM = tailCoutG_;
                uint32_t currentM = mCoreIndx_ * tiling_->singleCoreCout;
                if (tailM <= currentM) {
                    singleShapeMInCurrentHo_ = 0;
                    return;
                } else {
                    uint32_t mRemain = tailM - currentM;
                    singleShapeMInCurrentHo_ = mRemain < tiling_->singleCoreCout ? mRemain : tiling_->singleCoreCout;
                }
            }
        }
    }

    __aicore__ inline void ClearWorkspace(const GlobalTensor<float>& workspace)
    {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

        int32_t clearSize = tiling_->baseM * tiling_->baseN;
        LocalTensor<float> tempBuf = dw_.ctx.vecBuf_.template Get<float>();
        Duplicate(tempBuf, 0.0f, clearSize);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = Ceil(clearSize * sizeof(float), ONE_BLOCK_SIZE);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopy(workspace[0], tempBuf, copyParams);
        event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }
};
}

#endif // CONV3D_BACKPROP_FILTER_H