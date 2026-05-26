/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_ext_tiling_ar_full_load.cc
 * \brief
 */
#include "softmax_grad_ext_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace optiling {

static constexpr int64_t UB_RESERVED_BYTE = 1024 * 8;  //保留空间
static constexpr int64_t R_MAX_VALUE = 16384; // 支持的最大r值
static constexpr int64_t BINARY_TMP_LOCAL_SHAPE = 512;  //临时局部内存大小

bool SoftmaxGradExtTilingAR::IsCapable()
{
    OP_TILING_CHECK(
        a0_ != DIM_NUM_ONE, OP_LOGI(context_->GetNodeName(), "AR full load template is not capable. "), return false);

    OP_TILING_CHECK(
        r_ > R_MAX_VALUE,
        OP_LOGI(
            context_->GetNodeName(), "AR full load template is not capable. actual r is %ld, larger than %ld", r_,
            R_MAX_VALUE),
        return false);
    return true;
}

ge::graphStatus SoftmaxGradExtTilingAR::DoOpTiling()
{
    int64_t rAligned =
        CeilAlign(r_, (blockSize_ / xDtypeSize_)); // rAligned是将r_向上去整到blockSize_ /
                                                   // xDtypeSize_的倍数；blocksize是块大小，xDtypeSize是输入数据的字节数
    int64_t ubFactor = (aicoreParams_.ubSize - UB_RESERVED_BYTE - BINARY_TMP_LOCAL_SHAPE) /
                       (DOUBLE_BUFFER * rAligned * (xDtypeSize_ * CONST_THREE + yDtypeSize_));
    // ubFactor:UB中可以容纳的块数；aicoreParams_.ubSize:是UB的总大小；UB_RESERVED_BYTE和BINARY_TMP_LOCAL_SHAPE是保留和临时使用的字节数；DOUBLE_BUFFER:双缓冲的倍数；分母表示每块需要的内存大小
    OP_TILING_CHECK(
        (ubFactor <= 0),
        OP_LOGI(context_->GetNodeName(), "AR full load template is not capable. r is %ld, ubFactor %ld", r_, ubFactor),
        return ge::GRAPH_PARAM_INVALID);

    int64_t rLoopCount =
        CeilDiv(rAligned, vlFp32_); // vlFP32:每周期处理的FP32元素数量；rloopcount表示需要多少次循环来处理rAligned的数据

    // 按a1分核
    int64_t aBlockFactor = CeilDiv(
        a1_,
        static_cast<int64_t>(
            aicoreParams_.blockDim)); // aicoreParams_.blockDim是每个核心能处理的块大小；ablockfactor：单核处理的行数。
    usedCoreNums_ = CeilDiv(a1_, aBlockFactor); // usedCoreNums：实际使用的核数

    ubFactor = ubFactor < aBlockFactor ? ubFactor : aBlockFactor;

    // tiling data设置
    tilingData_.set_a(a1_);
    tilingData_.set_r(r_);
    tilingData_.set_rAligned(rAligned);
    tilingData_.set_aBlockFactor(aBlockFactor);
    tilingData_.set_rLoopCount(rLoopCount);
    tilingData_.set_ubFactor(ubFactor);
    tilingData_.set_x2IsScalar(isX2Scalar_ ? 1L : 0L);

    return ge::GRAPH_SUCCESS;
}

uint64_t SoftmaxGradExtTilingAR::GetTilingKey() const
{
    return TILINGKEY_AR;
}

ge::graphStatus SoftmaxGradExtTilingAR::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);                      // 设置实际使用的核数，即a1分成的块数
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1); // 获取工作空间
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_; // 设置顶一个工作空间的大小为workspacesize
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize()); // 将tilingdata的内容保存到缓冲区
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxGradExt, SoftmaxGradExtTilingAR, TEMPLATE_AR_FULL_LOAD_PRIORITY);

} // namespace optiling