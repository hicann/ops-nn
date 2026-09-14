/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool3d_grad_ncdhw_tiling_common.h
 * \brief NCDHW big/small kernel tiling 公共基类: 承载公共常量/数据结构与公共流程方法
 *        (维度成员、基础信息初始化、IsMeetTargetCoreNum/IsMeetUBSize/DoUBTiling/DoBlockTiling、
 *        公共 tiling 字段下发), big/small 仅保留各自的切分搜索与 buffer 计算逻辑。
 *        注意: big/small 的 SplitInfo/tiling 结构不同, 本头文件不可同时被两个 tiling 头包含。
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_TILING_COMMON_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_TILING_COMMON_H_

#include "adaptive_avg_pool3d_grad_tiling_arch35.h"
#include "tiling/tiling_api.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../pool_grad_common/op_host/arch35/pool_grad_tiling_split_helper.h"

namespace optiling {

constexpr int64_t FLOAT16_SIZE = 2;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr int64_t INT32_SIZE = 4;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t UB_RESVERVED_SIZE = 2048;
constexpr int64_t UB_TEMP_BUFF_SIZE = 256 * 10;
constexpr int64_t T3_INT64 = 10;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t THRESHOLD = 2;
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int64_t ALIGN_NUM = 32;
constexpr int64_t MAX_INT32 = 2147483647;

struct AdaptiveAvgPool3dGradNCDHWBaseInfo {
    int64_t vRegSize{0};
    int64_t ubBlockSize{0};
    int64_t inputBytes{0};
    int64_t indexBytes{0};
    int64_t availableUb{0};
    int64_t totalCoreNum{0};
    int64_t coreUsedForBestPerformance{0};
    int64_t dProBatchSize{0};
    int64_t hProBatchSize{0};
    int64_t wProBatchSize{0};
    int64_t inputNCSize{0};
    int64_t maxDataNumInOneBlock{0};
    int64_t proDataNumInOneBeatT2{0};
    int64_t isPad{0};
    int64_t isOverlap{0};
};

// big/small SplitInfo 的公共字段部分
struct AdaptiveAvgPool3dGradNCDHWSplitCommon {
    // DoUBTiling
    int64_t isCheckRange{0};

    int64_t highAxisInner{0};
    int64_t highAxisTail{0};
    int64_t highAxisOuter{0};

    int64_t dOutputInner{0};
    int64_t dOutputTail{0};
    int64_t dOutputOuter{0};

    int64_t hOutputInner{0};
    int64_t hOutputTail{0};
    int64_t hOutputOuter{0};

    int64_t wOutputInner{0};
    int64_t wOutputTail{0};
    int64_t wOutputOuter{0};

    // DoBlockTiling
    int64_t normalCoreProcessNum{0};
    int64_t tailCoreProcessNum{0};
    int64_t usedCoreNum{0};
    int64_t totalBaseBlockNum{0};

    // DoBufferCalculate
    int64_t totalBufferSize{0};
};

template <typename SplitInfoT>
class AdaptiveAvgPool3dGradNCDHWTilingCommon : public AdaptiveAvgPool3dGradTilingBaseV35 {
public:
    explicit AdaptiveAvgPool3dGradNCDHWTilingCommon(gert::TilingContext* context)
        : AdaptiveAvgPool3dGradTilingBaseV35(context)
    {}

    ~AdaptiveAvgPool3dGradNCDHWTilingCommon() override {}

    int64_t gradInputN{0};
    int64_t gradInputC{0};
    int64_t gradInputD{0};
    int64_t gradInputH{0};
    int64_t gradInputW{0};

    int64_t gradOutputN{0};
    int64_t gradOutputC{0};
    int64_t gradOutputD{0};
    int64_t gradOutputH{0};
    int64_t gradOutputW{0};

    int64_t kernelD{0};
    int64_t kernelH{0};
    int64_t kernelW{0};

    AdaptiveAvgPool3dGradNCDHWBaseInfo baseData;
    SplitInfoT splitData;

public:
    // 约定: IsMeetTargetCoreNum / IsMeetUBSize 需公开, 供 pool_grad_tiling_split_helper 模板调用
    bool IsMeetTargetCoreNum()
    {
        int64_t tmpWOutputOuter = Ops::Base::CeilDiv(gradOutputW, splitData.wOutputInner);
        int64_t tmpHOutputOuter = Ops::Base::CeilDiv(gradOutputH, splitData.hOutputInner);
        int64_t tmpDOutputOuter = Ops::Base::CeilDiv(gradOutputD, splitData.dOutputInner);
        int64_t tmpHighAxisOutputOuter = Ops::Base::CeilDiv(baseData.inputNCSize, splitData.highAxisInner);

        return tmpDOutputOuter * tmpWOutputOuter * tmpHOutputOuter * tmpHighAxisOutputOuter >=
               baseData.coreUsedForBestPerformance;
    }

    bool IsMeetUBSize()
    {
        DoBufferCalculate();
        return splitData.totalBufferSize <= baseData.availableUb;
    }

protected:
    // 初始化公共维度成员与 baseData 公共字段 (proDataNumInOneBeatT2 的取整方式 big/small 不同, 由派生类补充)
    void InitCommonVars()
    {
        gradInputN = inputData.nGrad;
        gradInputC = inputData.cGrad;
        gradInputD = inputData.dGrad;
        gradInputH = inputData.hGrad;
        gradInputW = inputData.wGrad;

        gradOutputN = inputData.nX;
        gradOutputC = inputData.cX;
        gradOutputD = inputData.dX;
        gradOutputH = inputData.hX;
        gradOutputW = inputData.wX;

        baseData.vRegSize = Ops::Base::GetVRegSize(context_);
        baseData.ubBlockSize = Ops::Base::GetUbBlockSize(context_);
        baseData.inputBytes = inputData.inputDtype == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
        baseData.availableUb = ubSize_ - UB_RESVERVED_SIZE - UB_TEMP_BUFF_SIZE;
        baseData.totalCoreNum = coreNum_;
        baseData.coreUsedForBestPerformance = baseData.totalCoreNum;
        baseData.maxDataNumInOneBlock = baseData.ubBlockSize / baseData.inputBytes;
        baseData.inputNCSize = gradOutputN * gradOutputC;
    }

    void DoUBTiling()
    {
        SearchBestTiling();
        DoBufferCalculate();
        PoolGradTiling::CalcAxisOuterTail(gradOutputW, splitData.wOutputInner, splitData.wOutputOuter,
                                          splitData.wOutputTail);
        PoolGradTiling::CalcAxisOuterTail(gradOutputH, splitData.hOutputInner, splitData.hOutputOuter,
                                          splitData.hOutputTail);
        PoolGradTiling::CalcAxisOuterTail(gradOutputD, splitData.dOutputInner, splitData.dOutputOuter,
                                          splitData.dOutputTail);
        PoolGradTiling::CalcAxisOuterTail(baseData.inputNCSize, splitData.highAxisInner, splitData.highAxisOuter,
                                          splitData.highAxisTail);
    }

    void DoBlockTiling()
    {
        splitData.totalBaseBlockNum = splitData.highAxisOuter * splitData.hOutputOuter * splitData.wOutputOuter *
                                      splitData.dOutputOuter;
        splitData.normalCoreProcessNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, baseData.totalCoreNum);
        splitData.usedCoreNum = Ops::Base::CeilDiv(splitData.totalBaseBlockNum, splitData.normalCoreProcessNum);
        splitData.tailCoreProcessNum = splitData.totalBaseBlockNum -
                                       splitData.normalCoreProcessNum * (splitData.usedCoreNum - 1);
    }

    // 下发 dInput~usedCoreNum 共 21 个公共 tiling 字段 (buffer 类字段由派生类补充)
    template <typename TilingDataT>
    void SetCommonTilingData(TilingDataT* tilingData) const
    {
        tilingData->dInput = gradInputD;
        tilingData->hInput = gradInputH;
        tilingData->wInput = gradInputW;
        tilingData->dOutput = gradOutputD;
        tilingData->hOutput = gradOutputH;
        tilingData->wOutput = gradOutputW;
        tilingData->highAxisInner = splitData.highAxisInner;
        tilingData->highAxisTail = splitData.highAxisTail;
        tilingData->highAxisOuter = splitData.highAxisOuter;
        tilingData->dOutputInner = splitData.dOutputInner;
        tilingData->dOutputTail = splitData.dOutputTail;
        tilingData->dOutputOuter = splitData.dOutputOuter;
        tilingData->hOutputInner = splitData.hOutputInner;
        tilingData->hOutputTail = splitData.hOutputTail;
        tilingData->hOutputOuter = splitData.hOutputOuter;
        tilingData->wOutputInner = splitData.wOutputInner;
        tilingData->wOutputTail = splitData.wOutputTail;
        tilingData->wOutputOuter = splitData.wOutputOuter;
        tilingData->normalCoreProcessNum = splitData.normalCoreProcessNum;
        tilingData->tailCoreProcessNum = splitData.tailCoreProcessNum;
        tilingData->usedCoreNum = splitData.usedCoreNum;
    }

    ge::graphStatus DoLibApiTiling() override { return ge::GRAPH_SUCCESS; }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto workspaces = context_->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
        workspaces[0] = WORKSPACE_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    virtual void SearchBestTiling() = 0;
    virtual void DoBufferCalculate() = 0;
};

} // namespace optiling

#endif // ADAPTIVE_AVG_POOL3D_GRAD_NCDHW_TILING_COMMON_H_
