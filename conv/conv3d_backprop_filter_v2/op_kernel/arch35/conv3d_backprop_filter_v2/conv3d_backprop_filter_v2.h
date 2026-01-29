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
    uint32_t l0cCondition>
class Conv3dDw {
public:
    __aicore__ inline Conv3dDw()
    {
    };
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR dedy,
        GM_ADDR y, GM_ADDR workSpace,
        const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
        InitTilingData(tilingData);
        // init global buffer
        xGm_.SetGlobalBuffer((__gm__ xType*)x);
        dedyGm_.SetGlobalBuffer((__gm__ dedyType*)dedy);
        yGm_.SetGlobalBuffer((__gm__ yType*)y);
        dw_.Init(&(tilingData->dwTiling), seperateDk_);
        if constexpr (conv3ddwConfig.l0cCondition == TPL_MN_STREAM_K) {
            workspaceGm_.SetGlobalBuffer((__gm__ float*)workSpace);
        }
    }

    /** main logical function
    */
    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            if constexpr (conv3ddwConfig.l0cCondition == TPL_STREAM_DFL) {
                return;
            }
            if (GetSubBlockIdx() > 0) {
                dw_.End();
                return;
            }
        }
        if ASCEND_IS_AIC {
            if (GetBlockIdx() >= GetBlockNum()) {
                dw_.End();
                return;
            }
        }
        CalSingleCoreShape();
        for (uint32_t i = 0; i < singleShapeGroup_; ++i) {
            uint32_t groupIdx = groupCoreIndx_ * singleCoreGroup_ + i;
            for (uint64_t j = 0; j < singleShapeBatch_; ++j) {
                uint64_t batchDoutIdx = batchCoreIndx_ * singleCoreBatch_ + j;
                uint64_t batchIdx = batchDoutIdx / dout_;
                uint64_t doutIdx = batchDoutIdx - batchIdx * dout_;
                for (uint32_t t = 0; t < singleShapeDk_; ++t) {
                    uint32_t dkIdx = dkCoreIndx_ * singleCoreDk_ + t;
                    ReCalDkCinSingleCoreShape(doutIdx, groupIdx, dkIdx);
                    if (singleShapeNInCurrentHo_ == 0 || singleShapeMInCurrentHo_ == 0) {
                        continue;
                    }
                    CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx);
                    dw_.SetFmap(xGm_[offsetB_]);
                    dw_.SetOutBackprop(dedyGm_[offsetA_]);
                    // cin和dk分轴，当cin有尾块且d轴有padding，将singleShapeNInCurrentHo作为参数传递，无法计算出正确的cin和dk
                    // 因此需传递一个singleShapeCin，以便获取singleShapeCin和singleShapeDk
                    dw_.SetSingleShape(
                        singleShapeMInCurrentHo_, singleShapeNInCurrentHo_,
                        singleShapeK_, singleShapeCin_, ONE_BATCH);
                    dw_.SetStartIdx(batchDoutIdx, hoStartIdx_, dkIdx);
                    dw_.IterateAll(yGm_[offsetC_], 1); // 1 means atomic add
                }
            }
        }
        dw_.End();
    }

    __aicore__ inline void ProcessWithDeterministic()
    {
        if constexpr (conv3ddwConfig.l0cCondition != TPL_MN_STREAM_K) {
            return;
        }

        if (block_idx >= GetBlockNum()) {
            return;
        }
        CalSingleCoreShape();
        for (uint32_t i = 0; i < singleShapeGroup_; ++i) {
            uint32_t groupIdx = groupCoreIndx_ * singleCoreGroup_ + i;
            uint64_t batchDoutIdx = batchCoreIndx_ * singleCoreBatch_;
            uint64_t batchIdx = batchDoutIdx / dout_;
            uint64_t doutIdx = batchDoutIdx - batchIdx * dout_;
            for (uint32_t t = 0; t < singleCoreDk_; ++t) {
                uint32_t dkIdx = dkCoreIndx_ * singleCoreDk_ + t;
                /* 使用singleCoreBatch进行循环，若尾块超出了有效数据范围，
                * 则设置singleShapeNInCurrentHo_和singleShapeMInCurrentHo_为零，不在IterateAll中计算 */
                ReCalDkCinSingleCoreShape(doutIdx, groupIdx, dkIdx);

                /* 在确定性计算当中，即使当前核没有有效数据进行计算，
                * 仍然需要进行workspace置零操作和同步操作，因此不能直接跳过计算 */
                if (singleShapeNInCurrentHo_ == 0 || singleShapeMInCurrentHo_ == 0) {
                    // 跳过场景只需要用到C矩阵运算，只传入group和dk偏移即可
                    CalcOffset(0, 0, groupIdx, dkIdx);
                    dw_.SetSingleShape(singleShapeM_, singleShapeN_, singleShapeK_, singleShapeCin_, singleShapeBatch_);
                } else {
                    /* 当且仅当当前核有有效数据进行计算时，需要计算Gm地址中的offset并进行设置 */
                    CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx);
                    dw_.SetFmap(xGm_[offsetB_]);
                    dw_.SetOutBackprop(dedyGm_[offsetA_]);
                    dw_.SetSingleShape(
                        singleShapeMInCurrentHo_, singleShapeNInCurrentHo_,
                        singleShapeK_, singleShapeCin_, singleShapeBatch_);
                }
                dw_.SetStartIdx(batchDoutIdx, hoStartIdx_, dkIdx);
                DeterministicIterateAll();
            }
        }
        dw_.End();
    }

protected:
    static constexpr ConvolutionBackprop::CubeFormat xCubeFormat = GetFormat(xFormat);
    static constexpr ConvolutionBackprop::CubeFormat dedyCubeFormat = GetFormat(dedyFormat);
    static constexpr ConvolutionBackprop::CubeFormat yCubeFormat = GetFormat(yFormat);
    static constexpr uint32_t ONE_BATCH = 1;
    using xDwType = ConvolutionBackprop::ConvType<TPosition::GM, xCubeFormat, xType>;
    using filterSizeDwType = ConvolutionBackprop::ConvType<TPosition::GM, ConvolutionBackprop::CubeFormat::ND, int32_t>;
    using dedyDwType = ConvolutionBackprop::ConvType<TPosition::GM, dedyCubeFormat, dedyType>;
    using yDwType = ConvolutionBackprop::ConvType<TPosition::GM, yCubeFormat, yType>;
    static constexpr Conv3ddwConfig conv3ddwConfig = {l0cCondition};
    ConvolutionBackprop::Conv3DBackpropFilter<xDwType, filterSizeDwType, dedyDwType, yDwType, conv3ddwConfig> dw_;
    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> dedyGm_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<float> workspaceGm_;
    uint64_t offsetA_ = 0;
    uint64_t offsetB_ = 0;
    uint64_t offsetC_ = 0;
    uint32_t batchDim_ = 0;
    uint32_t kDim_ = 0;
    uint32_t mDim_ = 0;
    uint32_t nDim_ = 0;
    uint32_t groupDim_ = 0;
    uint32_t batchCoreIndx_ = 0;
    uint32_t mCoreIndx_ = 0;
    uint32_t nCoreIndx_ = 0;
    uint32_t kCoreIndx_ = 0;
    uint32_t cout_ = 0;
    uint32_t cin_ = 0;
    uint32_t ho_ = 0;
    uint32_t wo_ = 0;
    uint32_t hi_ = 0;
    uint32_t wi_ = 0;
    uint32_t hk_ = 0;
    uint32_t wk_ = 0;
    uint32_t strideH_ = 0;
    uint32_t strideD_ = 0;
    uint32_t padUp_ = 0;
    uint32_t padFront_ = 0;
    uint32_t padBack_ = 0;
    uint32_t dout_ = 0;
    uint32_t di_ = 0;
    uint32_t dk_ = 0;
    uint32_t hoStartIdx_ = 0;
    uint32_t batch_ = 0;
    uint32_t c0_ = 0;
    uint64_t m_ = 0;
    uint64_t n_ = 0;
    uint64_t k_ = 0;
    uint64_t singleCoreBatch_ = 0;
    uint64_t singleCoreCout_ = 0;
    uint64_t singleCoreCin_ = 0;
    uint64_t singleCoreHo_ = 0;
    uint64_t singleShapeBatch_ = 0;
    uint64_t singleShapeM_ = 0;
    uint64_t singleShapeN_ = 0;
    uint64_t singleShapeNInCurrentHo_ = 0;
    uint64_t singleShapeMInCurrentHo_ = 0;
    uint64_t singleShapeK_ = 0;
    uint64_t coutG_ = 0;
    uint64_t cinG_ = 0;
    uint32_t dkDim_ = 0;
    uint32_t singleShapeCin_ = 0;
    uint32_t groupCoreIndx_ = 0;
    uint32_t dkCoreIndx_ = 0;
    uint32_t cinCoreIndx_ = 0;
    uint32_t group_ = 0;
    uint32_t singleCoreGroup_ = 0;
    uint32_t singleCoreDk_ = 0;
    uint32_t singleShapeGroup_ = 0;
    uint32_t singleShapeDk_ = 0;
    uint32_t dilationD_ = 0;
    uint32_t tailCinG_ = 0;
    uint32_t tailCoutG_ = 0;
    uint32_t realGroup_ = 0;
    bool groupFlag_ = 0;
    bool seperateDk_ = 0;
    bool basicBlockFlag_ = 0; // 后续默认基本块后，此标识和相关改动可以删除
    uint64_t offsetCubeWorkSpaceC_ = 0;
    static constexpr uint64_t SYNC_MODE0 = 0;
    static constexpr uint64_t SYNC_MODE2 = 2;
    static constexpr uint64_t SYNC_MODE4 = 4;
    static constexpr uint16_t SYNC_AIC_ONLY_ALL_DET_FLAG = 4;
    static constexpr uint16_t SYNC_AIC_AIV_DET_FLAG = 8;
    static constexpr uint64_t CUBE_WORKSPACE = AscendC::TOTAL_L0C_SIZE >> 2;

    __aicore__ inline void InitTilingData(
        const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData,
        bool isSeperateDk = false, bool isBasicBlock = false)
    {
        batchDim_ = tilingData->params.batchDim;
        mDim_ = tilingData->params.mDim;
        nDim_ = tilingData->params.nDim;
        kDim_ = tilingData->params.kDim;
        groupDim_ = tilingData->params.groupDim;
        batch_ = tilingData->dwTiling.batch;
        dk_ = tilingData->dwTiling.dk;
        c0_ = tilingData->dwTiling.k0;
        k_ = static_cast<uint64_t>(tilingData->dwTiling.ho) * tilingData->dwTiling.wo;
        singleCoreBatch_ = tilingData->dwTiling.singleCoreBatch;
        singleCoreCout_ = tilingData->dwTiling.singleCoreCout;
        singleCoreCin_ = tilingData->dwTiling.singleCoreCin;
        singleCoreDk_ = tilingData->dwTiling.singleCoreDk;
        singleCoreHo_ = tilingData->dwTiling.singleCoreHo;
        ho_ = tilingData->dwTiling.ho;
        wo_ = tilingData->dwTiling.wo;
        hi_ = tilingData->dwTiling.hi;
        wi_ = tilingData->dwTiling.wi;
        hk_ = tilingData->dwTiling.hk;
        wk_ = tilingData->dwTiling.wk;
        cout_ = tilingData->dwTiling.cout;
        cin_ = tilingData->dwTiling.cin;
        strideH_ = tilingData->dwTiling.strideH;
        padUp_ = tilingData->dwTiling.padUp;
        padFront_ = tilingData->dwTiling.padFront;
        padBack_ = tilingData->dwTiling.padBack;
        strideD_ = tilingData->dwTiling.strideD;
        dout_ = tilingData->dwTiling.dout;
        di_ = tilingData->dwTiling.di;
        hoStartIdx_ = 0;
        dkDim_ = tilingData->params.dkDim;
        group_ = tilingData->dwTiling.group;
        realGroup_ = tilingData->dwTiling.realGroup;
        singleCoreGroup_ = tilingData->dwTiling.singleCoreGroup;
        dilationD_ = tilingData->dwTiling.dilationD;
        groupCoreIndx_ = 0;
        singleShapeGroup_ = 1;
        groupFlag_ = group_ > 1;
        basicBlockFlag_ = isBasicBlock;
        // 当分组卷积或dilation时，cin和dk分轴，需标志位
        if constexpr (conv3ddwConfig.l0cCondition != TPL_MN_STREAM_K) {
            // 当分组卷积或dilation时，cin和dk分轴，需标志位
            seperateDk_ = groupFlag_ || (dilationD_ > 1) || isSeperateDk;
        } else {
            seperateDk_ = true;
        }
        InitTilingData_diff_1971_1982(tilingData);
    }

    __aicore__ inline void InitTilingData_diff_1971_1982(
        const conv_bp_v2_kernel::Conv3DBackpropFilterV2TilingData* tilingData)
    {
#if defined(__DAV_C310__)
        coutG_ = static_cast<uint64_t>(tilingData->dwTiling.cout1G);
        cinG_ = static_cast<uint64_t>(tilingData->dwTiling.cin1G);
        // 判断条件为是否为分组卷积，而不是是否cin和dk分轴
        if (groupFlag_) {
            m_ = coutG_;
            n_ = cinG_ * tilingData->dwTiling.hk * tilingData->dwTiling.wk;
            // cinG等于cin1g乘以c0，cin1g等于扩维后的cin除以c0，又因为扩维系数必定对齐C0，因此，cinG就是扩维后的每组cin个数
            tailCinG_ = cin_ - cinG_ * (realGroup_ - 1);
            tailCoutG_ = cout_ - coutG_ * (realGroup_ - 1);
        } else {
            m_ = cout_;
            n_ = cin_ * tilingData->dwTiling.hk * tilingData->dwTiling.wk;
            if (!seperateDk_) {
                n_ *= dk_;
            }
        }
#endif
    }

#if defined(__DAV_C310__)
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
                groupOffsetA = static_cast<uint64_t>(groupIdx) * coutG_ * dout_ * ho_ * wo_;
            }
            hoOffset = hoCal * wo_;
            doOffset = static_cast<uint64_t>(doutIdx) * ho_ * wo_;
            coutOffset = static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_ * dout_ * ho_ * wo_;
            batchOffsetA = static_cast<uint64_t>(batchIdx) * cout_ * dout_ * ho_ * wo_;
        } else {
            // ndhwc, cout is last axis
            if (groupFlag_) {
                // group将cin和cout分为多个组，每轮处理一组cin和cout，因此，每轮group的偏移计算仅算分组扩维后的cinG或coutG
                groupOffsetA = static_cast<uint64_t>(groupIdx) * coutG_;
            }
            hoOffset = hoCal * wo_ * cout_;
            doOffset = static_cast<uint64_t>(doutIdx) * ho_ * wo_ * cout_;
            coutOffset = static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_;
            batchOffsetA = static_cast<uint64_t>(batchIdx) * dout_ * ho_ * wo_ * cout_;
        }
        offsetA_ = batchOffsetA + groupOffsetA + coutOffset + doOffset + hoOffset;
    }

    __aicore__ inline void CalcBlockOffsetB(
        uint64_t batchIdx, uint64_t doutIdx,
        uint32_t groupIdx, uint32_t dkIdx, uint64_t hoCal, uint64_t& dinCurIdx)
    {
        uint64_t groupOffsetB = 0;
        dinCurIdx = static_cast<uint64_t>(doutIdx) * strideD_;
        uint64_t dinIdx = static_cast<uint64_t>(dkIdx) * dilationD_;
        // 当dilationD_大于1或者分组卷积时，cin与dk分轴，dinIdx无需考虑pad，使用seperateDk_作为判断条件
        if (seperateDk_) {
            dinCurIdx += dinIdx;
            dinIdx = dinCurIdx - padFront_;
        } else {
            if (dinCurIdx > padFront_) {
                dinIdx += dinCurIdx - padFront_;
            }
        }

        uint64_t hiIdx = 0;
        if (hoCal * strideH_ > padUp_) {
            hiIdx = hoCal * strideH_ - padUp_;
        }

        uint64_t hiOffset = 0;
        uint64_t dinDkOffset = 0;
        uint64_t cinOffset = 0;
        uint64_t batchOffsetB = 0;
        if constexpr (xCubeFormat == ConvolutionBackprop::CubeFormat::NCDHW) {
            if (groupFlag_) {
                groupOffsetB = static_cast<uint64_t>(groupIdx) * cinG_ * di_ * hi_ * wi_;
            }
            hiOffset = hiIdx * wi_;
            dinDkOffset = dinIdx * hi_ * wi_;
            //分组卷积时，需注意nCoreIndx_就是cinCoreIndx_，其他情况nCoreIndx_为cinIndx和dkIndx
            cinOffset = static_cast<uint64_t>(cinCoreIndx_) * singleCoreCin_ * di_ * hi_ * wi_;
            batchOffsetB = static_cast<uint64_t>(batchIdx) * cin_ * di_ * hi_ * wi_;
        } else {
            if (groupFlag_) {
                groupOffsetB = static_cast<uint64_t>(groupIdx) * cinG_;
            }
            hiOffset = hiIdx * wi_ * cin_;
            dinDkOffset = dinIdx * hi_ * wi_ * cin_;
            //分组卷积时，需注意nCoreIndx_就是cinCoreIndx_，其他情况nCoreIndx_为cinIndx和dkIndx
            cinOffset = static_cast<uint64_t>(cinCoreIndx_) * singleCoreCin_;
            batchOffsetB = static_cast<uint64_t>(batchIdx) * di_ * hi_ * wi_ * cin_;
        }
        offsetB_ = batchOffsetB + groupOffsetB + cinOffset + dinDkOffset + hiOffset;
    }

    __aicore__ inline void CalcBlockOffsetC(
        uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx,
        uint64_t dinCurIdx)
    {
        uint64_t groupOffsetC = 0;
        uint64_t coutOffsetC = 0;
        if constexpr (yCubeFormat == ConvolutionBackprop::CubeFormat::DHWCN) {
            coutOffsetC = mCoreIndx_ * singleCoreCout_;
        } else {
            coutOffsetC = mCoreIndx_ * singleCoreCout_ * cin_ * dk_ * hk_ * wk_;
        }

        // 判断条件为是否为分组卷积
        if (groupFlag_) {
            if (yCubeFormat == ConvolutionBackprop::CubeFormat::DHWCN) {
                // 对于输出位DHWCN的C矩阵，GM的排布为group dk hk wk cin/group cout/group
                groupOffsetC = static_cast<uint64_t>(groupIdx) * coutG_;
            } else {
                // C矩阵的GM排布为group cout/group cin/group dk hk wk，而每轮处理的group都是经过扩维的
                // 因此可以将enlarge_看成扩维前singleCoreGroup，而enlarge * cout/group等于coutG_，因此偏移计算如下
                // group与cout绑定，而ncdhw和ndhwc格式的排布cout并未发生变化，因此groupOffsetC和coutOffsetC都无需发生变化
                groupOffsetC = static_cast<uint64_t>(groupIdx) * coutG_ * (cin_ / group_) * dk_ * hk_ * wk_;
                // 当分组卷积时，C矩阵的shape由cout cin dk hk wk变为cout cin/group dk hk wk，因此offset计算偏移也发生变化
                coutOffsetC = static_cast<uint64_t>(mCoreIndx_) * singleCoreCout_ * (cin_ / group_) * dk_ * hk_ * wk_;
            }
        }

        if constexpr (yCubeFormat == ConvolutionBackprop::CubeFormat::NCDHW) {
            uint64_t cinDkOffset = cinCoreIndx_ * singleCoreCin_ * dk_ * hk_ * wk_ + dkIdx * hk_ * wk_;
            offsetC_ = groupOffsetC + cinDkOffset + coutOffsetC;
            if constexpr (conv3ddwConfig.l0cCondition != TPL_MN_STREAM_K) {
                if (dinCurIdx < padFront_ && !seperateDk_) {
                    offsetC_ += static_cast<uint64_t>(padFront_ - dinCurIdx) * hk_ * wk_;
                }
            }
        } else if constexpr (yCubeFormat == ConvolutionBackprop::CubeFormat::NDHWC) {
            // NDHWC格式，C在最内轴，考虑到group，需要使用cinG计算偏移
            uint64_t cinDkOffset = static_cast<uint64_t>(cinCoreIndx_) * singleCoreCin_ + dkIdx * hk_ * wk_ * cinG_;
            offsetC_ = groupOffsetC + cinDkOffset + coutOffsetC;
            if (dinCurIdx < padFront_ && !seperateDk_) {
                offsetC_ += static_cast<uint64_t>(padFront_ - dinCurIdx) * hk_ * wk_ * cinG_;
            }
        } else {
            // DHWCN
            uint64_t cinDkOffset = cinCoreIndx_ * singleCoreCin_ * cout_ + dkIdx * hk_ * wk_ * (cin_ / group_) * cout_;
            offsetC_ = groupOffsetC + cinDkOffset + coutOffsetC;
            if (dinCurIdx < padFront_ && !seperateDk_) {
                offsetC_ += (padFront_ - dinCurIdx) * hk_ * wk_ * (cin_ / group_) * cout_;
            }
        }
    }

    __aicore__ inline void CalcOffset(uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx)
    {
        //该函数当前仅为模板0保留
        uint64_t hoCal = basicBlockFlag_ ? (kCoreIndx_ * singleShapeK_) / wo_ : kCoreIndx_ * singleCoreHo_;
        CalcOffset(batchIdx, doutIdx, groupIdx, dkIdx, hoCal);
    }

    __aicore__ inline void CalcOffset(
        uint64_t batchIdx, uint64_t doutIdx, uint32_t groupIdx, uint32_t dkIdx,
        uint64_t hoCal)
    {
        CalcBlockOffsetA(batchIdx, doutIdx, groupIdx, hoCal);
        uint64_t dinCurIdx = 0;
        CalcBlockOffsetB(batchIdx, doutIdx, groupIdx, dkIdx, hoCal, dinCurIdx);
        CalcBlockOffsetC(doutIdx, groupIdx, dkIdx, dinCurIdx);
    }

    __aicore__ inline void ReCalcSingleCoreBatchDout(uint64_t doutIdx, uint32_t dkIdx, bool& invalidDIndex)
    {
        uint64_t doutStartIdx = doutIdx;
        // 非基本块+group场景目前不支持batchDout内移，因此使用ONE_BATCH作为singleShapeBatch
        uint64_t curSingleShapeBatch = (!basicBlockFlag_ && groupFlag_) ? ONE_BATCH : singleShapeBatch_;
        for (uint64_t i = 0; i < curSingleShapeBatch; i++) {
            uint64_t doutCurIdx = (doutStartIdx + i) % dout_;
            uint64_t dinCurIdx = doutCurIdx * strideD_ + dkIdx * dilationD_;
            if ((dinCurIdx >= padFront_) && (dinCurIdx < (di_ + padFront_))) {
                invalidDIndex = false;
                return;
            }
        }
    }

    __aicore__ inline void ReCalDkCinSingleCoreShape(
        uint64_t doutIdx, uint32_t groupIdx,
        uint32_t dkIdx)
    {
        singleShapeNInCurrentHo_ = singleShapeN_;
        singleShapeMInCurrentHo_ = singleShapeM_;
        uint64_t dinIdx = doutIdx * strideD_;

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
        uint64_t newDk = dk_;
        if (dinIdx + newDk > padFront_ + di_) {
            if (dinIdx < padFront_ + di_) {
                newDk = padFront_ + di_ - dinIdx;
                if (n_ / dk_ * newDk < nCoreIndx_ * singleCoreCin_ * hk_ * wk_ * newDk) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                }
                uint64_t nRamin = n_ / dk_ * newDk - nCoreIndx_ * singleCoreCin_ * hk_ * wk_ * newDk;
                singleShapeNInCurrentHo_ =
                    nRamin < singleCoreCin_ * hk_ * wk_ * newDk ? nRamin : singleCoreCin_ * hk_ * wk_ * newDk;
            } else {
                singleShapeNInCurrentHo_ = 0;
            }
        }

        if (dinIdx < padFront_) {
            if (dinIdx + newDk > padFront_) {
                newDk = dinIdx + newDk - padFront_;
                if (n_ / dk_ * newDk < nCoreIndx_ * singleCoreCin_ * hk_ * wk_ * newDk) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                }
                uint64_t nRamin = n_ / dk_ * newDk - nCoreIndx_ * singleCoreCin_ * hk_ * wk_ * newDk;
                singleShapeNInCurrentHo_ =
                    nRamin < singleCoreCin_ * hk_ * wk_ * newDk ? nRamin : singleCoreCin_ * hk_ * wk_ * newDk;
            } else {
                singleShapeNInCurrentHo_ = 0;
            }
        }
    }
#endif

    __aicore__ inline void CalCoreIndex()
    {
        kCoreIndx_ = block_idx % kDim_;
        nCoreIndx_ = (block_idx / kDim_) % nDim_;
        mCoreIndx_ = (block_idx / (kDim_ * nDim_)) % mDim_;
        batchCoreIndx_ = (block_idx / (kDim_ * nDim_ * mDim_)) % batchDim_;

        uint64_t batchRamin = static_cast<uint64_t>(batch_) * dout_ - batchCoreIndx_ * singleCoreBatch_;
        uint64_t mRamin = m_ - mCoreIndx_ * singleCoreCout_;
        uint64_t kRamin = k_ - kCoreIndx_ * singleCoreHo_ * wo_;
        singleShapeBatch_ = batchRamin < singleCoreBatch_ ? batchRamin : singleCoreBatch_;
        singleShapeM_ = mRamin < singleCoreCout_ ? mRamin : singleCoreCout_;
        singleShapeK_ = kRamin < singleCoreHo_ * wo_ ? kRamin : singleCoreHo_ * wo_;
        hoStartIdx_ = kCoreIndx_ * singleCoreHo_;
    }

    __aicore__ inline void CalSingleCoreShape()
    {
        CalCoreIndex();
        // n_、nCoreIndx_、singleCoreCin_在不同的分支下表示含义不同
        // 一共四个分支220普通卷积和seperateDk_卷积、310普通卷积和seperateDk_卷积
        uint64_t nRamin = n_ - nCoreIndx_ * singleCoreCin_ * hk_ * wk_;

        // 310/220 seperateDk_分支，包含group或dilation
        if (seperateDk_) {
            // seperateDk_时，nCoreIndx_为cinIndx，cin和dk分轴，cin内循环，dk外循环
            cinCoreIndx_ = nCoreIndx_;
            // 310时，singleShapeN_为传递每个核上dk的大小，singleShapeCin_为传递每个核上cin的大小
            singleShapeN_ = nRamin < singleCoreCin_ * hk_ * wk_ ? nRamin : singleCoreCin_ * hk_ * wk_;
            singleShapeCin_ = singleShapeN_ / (hk_ * wk_);

            uint64_t totalDim = static_cast<uint64_t>(batchDim_) * kDim_ * nDim_ * mDim_;
            dkCoreIndx_ = (block_idx / totalDim) % dkDim_;
            // dk外循环
            singleShapeDk_ = (dk_ - dkCoreIndx_ * singleCoreDk_) > singleCoreDk_ ?
                                 singleCoreDk_ :
                                 (dk_ - dkCoreIndx_ * singleCoreDk_);
            if (groupFlag_) {
                groupCoreIndx_ = (block_idx / (totalDim * dkDim_)) % groupDim_;
                uint32_t groupRamin = realGroup_ - groupCoreIndx_ * singleCoreGroup_;
                singleShapeGroup_ = groupRamin < singleCoreGroup_ ? groupRamin : singleCoreGroup_;
            }
        } else {
            // 普通卷积，group循环初始化为1，index初始化为0；
            groupCoreIndx_ = 0;
            singleShapeGroup_ = 1;
#if defined(__DAV_C310__)
            // 310 普通卷积分支
            // 普通卷积 cin和dk分轴，且nCoreIndx_为cinIndx和dkIndx，各自求取Index
            // 在D轴有padding时，如果dkIndx取余得到（先dk后cin），那么逻辑与计算padding逻辑相反，具体参考上面注释
            dkCoreIndx_ = (singleCoreDk_ != dk_) ? (nCoreIndx_ / DivCeil(cin_, singleCoreCin_)) : 0;
            cinCoreIndx_ = (singleCoreDk_ != dk_) ? (nCoreIndx_ % DivCeil(cin_, singleCoreCin_)) : nCoreIndx_;
            singleShapeCin_ = (cin_ - cinCoreIndx_ * singleCoreCin_) > singleCoreCin_ ?
                                  singleCoreCin_ :
                                  (cin_ - cinCoreIndx_ * singleCoreCin_);
            singleShapeDk_ = (dk_ - dkCoreIndx_ * singleCoreDk_) > singleCoreDk_ ?
                                 singleCoreDk_ :
                                 (dk_ - dkCoreIndx_ * singleCoreDk_);
            // 310时，singleShapeN_为传递每个核上dk的大小，singleShapeCin_为传递每个核上cin的大小
            singleShapeN_ = singleShapeCin_ * singleShapeDk_ * hk_ * wk_;
            singleShapeDk_ = 1; // dk在内部循环
#endif
        }
    }

    __aicore__ inline void ReCalSingleShapeWithGroup(uint32_t groupIdx)
    {
        if (groupIdx == (realGroup_ - 1)) {
            if (tailCinG_ < cinG_) {
                uint64_t tailN = tailCinG_ * hk_ * wk_;
                uint64_t singleCoreN = singleCoreCin_ * hk_ * wk_;
                uint64_t currentN = nCoreIndx_ * singleCoreN;
                if (tailN <= currentN) {
                    singleShapeNInCurrentHo_ = 0;
                    return;
                } else {
                    uint64_t nRemain = tailN - currentN;
                    uint32_t cinRemain = tailCinG_ - nCoreIndx_ * singleCoreCin_;
                    singleShapeNInCurrentHo_ = nRemain < singleCoreN ? nRemain : singleCoreN;
                    // 310时，每个核的cin大小是由singleShapeCin_传递，因此，需计算singleShapeCin_
                    singleShapeCin_ = cinRemain < singleCoreCin_ ? cinRemain : singleCoreCin_;
                }
            }

            if (tailCoutG_ < coutG_) {
                uint32_t tailM = tailCoutG_;
                uint32_t currentM = mCoreIndx_ * singleCoreCout_;
                if (tailM <= currentM) {
                    singleShapeMInCurrentHo_ = 0;
                    return;
                } else {
                    uint32_t mRemain = tailM - currentM;
                    singleShapeMInCurrentHo_ = mRemain < singleCoreCout_ ? mRemain : singleCoreCout_;
                }
            }
        }
    }

    __aicore__ inline void ClearL0C()
    {
        constexpr uint32_t FRACTAL_LEN_BITS = 9; // 512 = 2^9
        LocalTensor<half> l0a = dw_.ctx.l0aBuf_.template Get<half>();
        LocalTensor<half> l0b = dw_.ctx.l0bBuf_.template Get<half>();
        LocalTensor<float> l0c = dw_.ctx.l0cPingPongFlag_ ?
                                     dw_.ctx.l0cPing_.template AllocTensor<float>() :
                                     dw_.ctx.l0cPong_.template AllocTensor<float>();

        PipeBarrier<PIPE_MTE1>();
        InitConstValue(l0a, {1, static_cast<uint16_t>(TOTAL_L0A_SIZE >> FRACTAL_LEN_BITS), 0, 0U});
        InitConstValue(l0b, {1, static_cast<uint16_t>(TOTAL_L0B_SIZE >> FRACTAL_LEN_BITS), 0, 0U});
        MmadParams mmadParams;
        mmadParams.m = dw_.ctx.tiling_->baseM;
        mmadParams.n = dw_.ctx.tiling_->baseN;
        mmadParams.k = 1;
        mmadParams.cmatrixInitVal = true;

        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE1_M>();
        SetFlag<HardEvent::MTE1_M>(eventId);
        WaitFlag<HardEvent::MTE1_M>(eventId);
        MmadImpl(l0c, l0a, l0b, mmadParams);
        // MMAD计算量baseM*baseN小于一定阈值时需要添加PIPE_M同步,当前平台阈值为2560: 10*256
        if (mmadParams.m * mmadParams.n < 2560) {
            PipeBarrier<PIPE_M>();
        }
        eventId = GetTPipePtr()->FetchEventID<HardEvent::M_MTE1>();
        SetFlag<HardEvent::M_MTE1>(eventId);
        WaitFlag<HardEvent::M_MTE1>(eventId);
        if (dw_.ctx.l0cPingPongFlag_) {
            dw_.ctx.l0cPing_.EnQue(l0c);
        } else {
            dw_.ctx.l0cPong_.EnQue(l0c);
        }
    }

    __aicore__ inline void LoadL0CToWorkspace(const GlobalTensor<float>& output)
    {
        LocalTensor<float> l0c = dw_.ctx.l0cPingPongFlag_ ?
                                     dw_.ctx.l0cPing_.template DeQue<float>() :
                                     dw_.ctx.l0cPong_.template DeQue<float>();
        FixpipeParamsC310<CO2Layout::NZ> fixPipeParams;
        fixPipeParams.mSize = dw_.ctx.tiling_->baseM;
        fixPipeParams.nSize = dw_.ctx.tiling_->baseN;
        fixPipeParams.srcStride = ShiftCeilM0(dw_.ctx.tiling_->baseM, BLOCK_CUBE) * BLOCK_CUBE;
        fixPipeParams.dstStride = dw_.ctx.tiling_->baseM * BLOCK_CUBE;
        fixPipeParams.quantPre = QuantMode_t::QF322F32_PRE;
        fixPipeParams.deqScalar = 0;
        fixPipeParams.unitFlag = 0;

        if (dw_.ctx.l0cPingPongFlag_) {
            Fixpipe<float, float, CFG_NZ>(output, l0c, fixPipeParams);
            dw_.ctx.l0cPing_.FreeTensor(l0c);
        } else {
            Fixpipe<float, float, CFG_NZ>(output[CUBE_WORKSPACE >> 1], l0c, fixPipeParams);
            dw_.ctx.l0cPong_.FreeTensor(l0c);
        }
    }

    __aicore__ inline void ClearWorkspace(const GlobalTensor<float>& workspace)
    {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

        int32_t clearSize = dw_.ctx.tiling_->baseM * dw_.ctx.tiling_->baseN;
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

    __aicore__ inline void GetDeterAddCore()
    {
        uint32_t addCoreStartInx = (nCoreIndx_ + (mCoreIndx_ + (dkCoreIndx_ + groupCoreIndx_ * dkCoreIndx_ * dkDim_) *
                                                  mDim_) * nDim_) * kDim_ * batchDim_;
        offsetCubeWorkSpaceC_ = (kCoreIndx_ + batchCoreIndx_ * kDim_ + addCoreStartInx) * CUBE_WORKSPACE;
        dw_.SetDeterministicCoreInfo(batchDim_ * kDim_, kCoreIndx_ + batchCoreIndx_ * kDim_, addCoreStartInx, false);
    }

    __aicore__ inline void DeterministicIterateAll()
    {
        GetDeterAddCore();
        bool isCompute = (singleShapeNInCurrentHo_ != 0 && singleShapeMInCurrentHo_ != 0);
        for (uint64_t iter = 0; iter < dw_.ctx.mIter_ * dw_.ctx.nIter_; iter++) {
            if ASCEND_IS_AIC {
                // 确定性计算的Pingpong逻辑关闭，此处AIC启动需要来自AIV确定性计算结束
                CubeWaitVector();
                ClearL0C();
                LoadL0CToWorkspace(workspaceGm_[offsetCubeWorkSpaceC_]);
                if (isCompute) {
                    dw_.Iterate();
                    // 0: disable atomic add; true: enable sequential write
                    dw_.GetTensorC(workspaceGm_[offsetCubeWorkSpaceC_], 0, true);
                } else if (dw_.ctx.tiling_->cl0Pbuffer > 1) {
                    dw_.ctx.l0cPingPongFlag_ = !dw_.ctx.l0cPingPongFlag_;
                }
                CubeNotifyVector();
            }
            if ASCEND_IS_AIV {
                dw_.Iterate();
                VecWaitCube();
                dw_.DeterministicReduceKInUb(yGm_[offsetC_], workspaceGm_);
                VecNotifyCube();
                if (dw_.ctx.tiling_->cl0Pbuffer > 1) {
                    dw_.ctx.l0cPingPongFlag_ = !dw_.ctx.l0cPingPongFlag_;
                }
            }
        }
        dw_.ctx.isFirstIter_ = true;
    }

    __aicore__ inline void CubeNotifyVector()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_DET_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_DET_FLAG);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_DET_FLAG);
#endif
    }

    __aicore__ inline void VecWaitCube()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIC_AIV_DET_FLAG);
#endif
    }

    __aicore__ inline void CubeWaitVector()
    {
#ifndef __CCE_KT_TEST__
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_M>(ConvolutionBackpropFunc::SYNC_AIV_AIC_DET_FLAG);
#endif
    }

    static __aicore__ inline void VecNotifyCube()
    {
#ifndef __CCE_KT_TEST__
        static constexpr uint16_t SYNC_AIV_ONLY_ALL_DET_FLAG = 2;
        CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_ONLY_ALL_DET_FLAG);
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_DET_FLAG);

        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(ConvolutionBackpropFunc::SYNC_AIV_AIC_DET_FLAG);
#endif
    }
};
}

#endif // CONV3D_BACKPROP_FILTER_H