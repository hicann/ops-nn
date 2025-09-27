/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv3d_backprop_filter_v2_stream_k_tiling.cpp
 * \brief
 */

 #include "conv3d_backprop_filter_v2_stream_k_tiling.h"

 #include <map>
 #include <numeric>
 #include <util/math_util.h>
 #include <log/log.h>
 #include <graph/utils/type_utils.h>
 #include "tiling_base/tiling_templates_registry.h"

namespace Ops {
namespace NN {
namespace Conv {
constexpr uint64_t CONST_ZERO = 0;
constexpr int32_t KERNEL_SIZE_4 = 4;
constexpr uint32_t STREAM_K = 1;
constexpr uint32_t MN_STREAM_K = 2;
constexpr const uint32_t NO_STREAMK_CALC = 0;
constexpr const uint32_t STREAMK_BATCHDOUT = 1;
constexpr const uint32_t STREAMK_HWOUT = 2;
constexpr const uint64_t ONE_DIM = 1;
constexpr uint32_t L0C_DTYPE_BYTE = 4;  //注意，修改此处值需要联动修改其它cc文件中同名的定义，后续优化，一份定义

bool Conv3DBackpropFilterV2StreamKTiling::IsCapable() {
    if (!IsSocVersion91095()) {
        return false;
    }

    if (dtypeByte_ != ge::GetSizeByDataType(ge::DT_BF16) && dtypeByte_ != ge::GetSizeByDataType(ge::DT_FLOAT16) &&
        dtypeByte_ != ge::GetSizeByDataType(ge::DT_FLOAT)) {
        OP_LOGD(opName_, "Stream K tiling is only supported for bf16/fp16/fp32/hf32 dataType.");
        return false;
    }
    // DW默认支持确定性，不再使用开关
    return true;
}

void Conv3DBackpropFilterV2StreamKTiling::InitSplitWOI() {
    if (enableSplitW) {
         if (runInfo_.wo <= SPLIT_WO_THRESHOLD) {
             blockTiling_.splitWo = runInfo_.wo;
             blockTiling_.splitWi = runInfo_.wi;
         } else {
             blockTiling_.splitWo = SPLIT_WO_SIZE;
             blockTiling_.splitWi = GetWiCal(blockTiling_.splitWo, blockTiling_.isSplitKernelHW);
             blockTiling_.tailWo = runInfo_.wo % blockTiling_.splitWo;
             blockTiling_.tailWi = GetWiCal(blockTiling_.tailWo, blockTiling_.isSplitKernelHW);
         }
    } else {
        blockTiling_.splitWo = runInfo_.wo;
        blockTiling_.splitWi = runInfo_.wi;
    }

    return ;
}

ge::graphStatus Conv3DBackpropFilterV2StreamKTiling::DoOpTiling()
{
    InitSplitWOI();
    if (!MultiCoreSplitMN()) {
        std::stringstream ss;
        ss << "the output shape info follows ncdhw format. "
           << "x shape is [" << runInfo_.batch << ", " <<  runInfo_.ci << ", " << runInfo_.di << ", " << runInfo_.hi << ", " << runInfo_.wi << "]. "
           << "output_backprop shape is [" << runInfo_.batch << ", " << runInfo_.co << ", " << runInfo_.dout << ", " << runInfo_.ho << ", " << runInfo_.wo << "]. "
           << "filter shape is [" << runInfo_.co << ", "<< runInfo_.ci << ", " << runInfo_.kd << ", " << runInfo_.kh << ", " << runInfo_.kw << "]. "
           << "stride is [1, 1, " << runInfo_.stride_d << ", " << runInfo_.stride_h << ", " << runInfo_.stride_w << "]. "
           << "dilation is [1, 1, " << runInfo_.dilation_d << ", " << runInfo_.dilation_h << ", " << runInfo_.dilation_w << "]. ";
        OP_LOGE(opName_, "StreamK tiling template do optiling failed. Exceed L1 buffer size, please check the shape and attribute. %s", ss.str().c_str());
        return ge::GRAPH_FAILED;
    }
    AdjustSmallCaseBaseBlock();
    DoStreamKTiling();
    OP_LOGD(opName_, "Finish doing basic block stream k tiling, total blockCnt[%ld], coreStreamK[%d].",
        blockTiling_.totalCnt, blockTiling_.coreStreamK);
    return ge::GRAPH_SUCCESS;
}

void Conv3DBackpropFilterV2StreamKTiling::AdjustSmallCaseBaseBlock()
{
    // 扩维情况下，不允许调整基本块大小，否则和重排不兼容
    if (blockTiling_.groupEnlarge) {
        return;
    }
    uint64_t mBlockCnt = Ops::Base::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.singleCoreM));
    uint64_t nBlockCnt = Ops::Base::CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.singleCoreN));
    uint64_t totalCnt = mBlockCnt * nBlockCnt * runInfo_.kd * runInfo_.real_g;
    // 若基本块数量能够做一轮计算，则不调整基本块大小，保持最优基本块
    if ((totalCnt >= coreNum_)) {
        return;
    }
    // 若HWout和BatchDout不满足StreamK的分核，则不调整基本块
    uint64_t splitCoreNum = coreNum_ / totalCnt;
    if ((coreNum_ % totalCnt == 0) && ((static_cast<uint64_t>(runInfo_.batch * runInfo_.dout) >= splitCoreNum) ||
        (static_cast<uint64_t>(runInfo_.ho) >= splitCoreNum))) {
        return;
    }
    int64_t mCntMax = std::min(Ops::Base::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(BLOCK_CUBE)),
        static_cast<uint64_t>(coreNum_)) / runInfo_.kd / runInfo_.real_g / nBlockCnt;
    for (int64_t mCnt = mCntMax; mCnt >= 1; mCnt--) {
        // 在基本块配置时，singleShapeM最大为256，此时根据L0大小重新配置singleShapeM，使在StreamK下能分满核
        uint64_t singleShapeM = Ops::Base::CeilAlign(Ops::Base::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(mCnt)),
            static_cast<uint64_t>(BLOCK_CUBE));
        uint64_t mCntFixed = Ops::Base::CeilDiv(mmInfo_.mValue, singleShapeM);
        uint64_t blockTotalCnt = mCntFixed * nBlockCnt * runInfo_.kd * runInfo_.real_g;
        if ((coreNum_ % blockTotalCnt != 0) || (blockTotalCnt > coreNum_)) {
            continue;
        }

        uint64_t blockBaseK = GetBaseK(singleShapeM, blockTiling_.blockBaseN);
        uint64_t streamkCoreDim = coreNum_ / blockTotalCnt;
        if ((runInfo_.batch * runInfo_.dout < static_cast<int32_t>(streamkCoreDim)) &&
            (blockBaseK < streamkCoreDim * runInfo_.wo)) {
            continue;
        }
        OP_LOGD(opName_, "Adjust the block baseM from [%d] to [%ld]", blockTiling_.blockBaseM, singleShapeM);

        blockTiling_.blockBaseK = blockBaseK;
        blockTiling_.blockBaseM = singleShapeM;
        if (blockTiling_.blockBaseM * blockTiling_.blockBaseN * DB_ON * L0C_DTYPE_BYTE <= L0C_SIZE) {
            blockTiling_.dbL0C = DB_ON;
        }
        SetStepK4SplitMN();
        if (!IsCurBlockL1Invalid()) {
            break;
        }
    }
}

uint64_t Conv3DBackpropFilterV2StreamKTiling::GetSingleShapeKByStreamK()
{
    uint64_t singleShapeK = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.ho),
        static_cast<uint64_t>(blockTiling_.coreStreamK)) * runInfo_.wo;
    singleShapeK = Ops::Base::CeilAlign(singleShapeK, static_cast<uint64_t>(runInfo_.wo));
    return singleShapeK;
}

bool Conv3DBackpropFilterV2StreamKTiling::IsSplitBatchDoutBetter()
{
    uint64_t singleCoreBatchDout = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout;
    uint64_t singleShapeBatchDout = Ops::Base::CeilDiv(singleCoreBatchDout, static_cast<uint64_t>(blockTiling_.coreStreamK));
    uint64_t batchDoutTail = singleCoreBatchDout % singleShapeBatchDout;
    uint64_t streamkBatchDim = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout, singleShapeBatchDout);
    uint64_t singleShapeK = GetSingleShapeKByStreamK();

    // 优先选择切BatchDout或者HWout 分核数多者
    uint64_t streamkHWoutDim = Ops::Base::CeilDiv(blockTiling_.singleCoreK, singleShapeK);
    if (streamkBatchDim < streamkHWoutDim) {
        return false;
    }

    // 若分核数相同，则选择尾块比例大者
    uint64_t singleShapeKTail = blockTiling_.singleCoreK % singleShapeK;
    double batchDoutTailRatio = static_cast<double>(batchDoutTail) / static_cast<double>(singleShapeBatchDout);
    double kTailRatio = static_cast<double>(singleShapeKTail) / static_cast<double>(singleShapeK);
    if (batchDoutTailRatio != 0 && batchDoutTailRatio < kTailRatio) {
        return false;
    }
    // 若尾块比例相同，则选择切BatchDout
    return true;
}

void Conv3DBackpropFilterV2StreamKTiling::DoStreamkByBatchDout()
{
    blockTiling_.singleCoreBatchDout = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout,
        static_cast<uint64_t>(blockTiling_.coreStreamK));
    uint64_t streamkBatchDim = Ops::Base::CeilDiv(static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout,
        blockTiling_.singleCoreBatchDout);
    if (streamkBatchDim == ONE_DIM) {
        blockTiling_.streamkType = NO_STREAMK_CALC;
    } else {
        blockTiling_.streamkType = STREAMK_BATCHDOUT;
        blockTiling_.coreStreamK = streamkBatchDim;
    }
}

void Conv3DBackpropFilterV2StreamKTiling::DoStreamkByHWout()
{
    blockTiling_.singleCoreBatchDout = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout;

    // 当前切HWout只对H方向切块
    uint64_t singleShapeK = GetSingleShapeKByStreamK();
    uint64_t streamkHWoutDim = Ops::Base::CeilDiv(blockTiling_.singleCoreK, singleShapeK);
    if (streamkHWoutDim == ONE_DIM) {
        blockTiling_.streamkType = NO_STREAMK_CALC;
    } else {
        blockTiling_.streamkType = STREAMK_HWOUT;
        blockTiling_.singleCoreK = singleShapeK;
        blockTiling_.coreStreamK = streamkHWoutDim;
        uint64_t blockBaseK = std::min(static_cast<uint64_t>(blockTiling_.blockBaseK), blockTiling_.singleCoreK);
        blockBaseK = Ops::Base::CeilAlign(blockBaseK, static_cast<uint64_t>(BLOCK_CUBE));
        blockTiling_.blockBaseK = blockBaseK;
        blockTiling_.stepKa =
            std::min(blockTiling_.stepKa, static_cast<uint32_t>(Ops::Base::CeilDiv(singleShapeK, blockBaseK)));
        blockTiling_.stepKb =
            std::min(blockTiling_.stepKb, static_cast<uint32_t>(Ops::Base::CeilDiv(singleShapeK, blockBaseK)));
    }
}

/*
    主轮计算于核内完成，无确定性计算；
    对尾轮计算进一步做streamk分核，实现负载均衡，减少核间拖尾；
    当前streamk只对最后一轮尾轮做K方向切分，因此只有一轮确定性计算；
    SplitM_K和SplitN_K为streamk的子集，为保证只有一轮确定性计算，streamk对K的切分数暂时不大.
*/
void Conv3DBackpropFilterV2StreamKTiling::DoStreamKTiling()
{
    uint64_t mCnt = Ops::Base::CeilDiv(mmInfo_.mValue, static_cast<uint64_t>(blockTiling_.singleCoreM));
    uint64_t nCnt = Ops::Base::CeilDiv(mmInfo_.nValue, static_cast<uint64_t>(blockTiling_.singleCoreN));
    blockTiling_.totalCnt = mCnt * nCnt * runInfo_.kd * runInfo_.real_g;
    blockTiling_.coreBindDirection = blockTiling_.totalCnt < coreNum_ ? STREAM_K : MN_STREAM_K;
    blockTiling_.singleCoreBatchDout = static_cast<uint64_t>(runInfo_.batch) * runInfo_.dout;
    blockTiling_.usedCoreNum = coreNum_;
    uint64_t streamkCnt = blockTiling_.totalCnt % coreNum_;
    if (streamkCnt == CONST_ZERO) { // 没有尾轮基本块
        blockTiling_.streamkType = NO_STREAMK_CALC;
        blockTiling_.coreStreamK = 0;
        OP_LOGD(opName_, "The basic block streamk template does not need to process the tail block.");
        return;
    } else if (streamkCnt > (coreNum_ >> 1)) { // 尾轮的基本块超过一半的核，不做streamk
        blockTiling_.streamkType = NO_STREAMK_CALC;
        blockTiling_.coreStreamK = 0;
        OP_LOGD(opName_, "The basic block streamk template does not process the tail block.");
        return;
    } else if (deterNotSupportFormat_) {
        blockTiling_.streamkType = NO_STREAMK_CALC;
        blockTiling_.coreStreamK = 0;
        OP_LOGD(opName_, "The basic block streamk template does not process the group > 1.");
        return;
    }

    blockTiling_.coreStreamK = coreNum_ / streamkCnt;

    /*
        StreamK可选择切BatchDout或者HWout，此时优先级为：
        1. 优先选择切BatchDout或者HWout 分核数多者；
        2. 若分核数相同，则选择尾块比例大者；
        3. 若尾块比例相同，则选择切BatchDout，保持HWout连续，对并包、stepk的加载数据量更大。
    */
    if (IsSplitBatchDoutBetter()) {
        DoStreamkByBatchDout();
    } else {
        DoStreamkByHWout();
    }
}

ge::graphStatus Conv3DBackpropFilterV2StreamKTiling::GetWorkspaceSize()
{
    constexpr uint64_t WORKSPACE = 16777216; // 16777216 : 16 * 1024 * 1024 libapiworkspace
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    size_t userWorkSpaceSize = 0;
    if (blockTiling_.streamkType != NO_STREAMK_CALC) {
        size_t singleSize = 0;
        if (tilingData_.dwTiling.get_cl0Pbuffer() > 1) {
            singleSize = L0C_SIZE >> 1;
        } else {
            singleSize = L0C_SIZE;
        }
        constexpr uint64_t WORKSPACE_PIECE_NUM = 2; // 2: 1 space for Cube and 1 space for Vector
        userWorkSpaceSize = WORKSPACE_PIECE_NUM * DB_ON * blockTiling_.usedCoreNum * singleSize;
    }
    workspaces[0] = WORKSPACE + userWorkSpaceSize;
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Conv3DBackpropFilterV2", Conv3DBackpropFilterV2StreamKTiling, 3);
}
}
}
