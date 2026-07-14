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
 * \file lamb_apply_weight_assign_tiling_arch35.cpp
 * \brief lamb_apply_weight_assign_tiling_arch35 source file
 */

#include "lamb_apply_weight_assign_tiling_arch35.h"
#include "../../../lamb_apply_common/lamb_apply_check_util.h"
#include <graph/utils/type_utils.h>
#include <string>
#include "infershape_broadcast_util.h"
#include "../../op_kernel/arch35/lamb_apply_weight_assign_dag.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_templates_registry.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr static uint64_t LAMB_APPLY_WEIGHT_ASSIGN_TILING_PRIORITY = 0;
constexpr static int32_t INPUT_NUM = 5;
constexpr static int32_t OUTPUT_NUM = 1;
static const char* const kInputNames[] = {"input0", "input1", "input2", "input3", "input_param"};
static const char* const kOutputNames[] = {"input_param"};

static ge::graphStatus TilingPrepareForLambApplyWeightAssign(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<LambApplyWeightAssignCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambApplyWeightAssignTiling::GetShapeAttrsInfo()
{
    static const int32_t kScalarInputIdx[] = {0, 1, 2};
    if (CheckLambApplyDtypeConsistency(context_, INPUT_NUM, kInputNames, OUTPUT_NUM, kOutputNames) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckLambApplyScalarNotEmpty(context_, kScalarInputIdx, sizeof(kScalarInputIdx) / sizeof(kScalarInputIdx[0]),
                                     kInputNames) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckInplaceShapeConstraint();
}

// input_param 是 in-place 更新的参数输出(next_param 原地写回其输入 buffer,见 proto "(in-place)"),
// 内核按 broadcast(input3, input_param) 的完整网格计算并写回,故 input_param 形状必须 == 该网格。
// 等价充要条件:input3 能广播进 input_param。
ge::graphStatus LambApplyWeightAssignTiling::CheckInplaceShapeConstraint()
{
    auto input3Shape = context_->GetInputShape(3);
    auto inputParamShape = context_->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input3Shape);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputParamShape);
    const auto& i3s = input3Shape->GetStorageShape();
    const auto& ips = inputParamShape->GetStorageShape();
    // input3 能广播进 input_param <=> broadcast(input3, input_param) == input_param
    gert::Shape bcShape;
    if (!Ops::Base::BroadcastShape(&i3s, &ips, &bcShape) || !(bcShape == ips)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "input3", Ops::Base::ToString(i3s).c_str(),
            "input3 must be broadcastable into the in-place param shape input_param (input_param is updated "
            "in-place and must equal the broadcast output shape)");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool LambApplyWeightAssignTiling::IsCapable() { return true; }

ge::graphStatus LambApplyWeightAssignTiling::DoOpTiling()
{
    // 空 tensor 应对(空进空出): 输出为空(0元素)时设 1 核(空转), 配合全0 tiling 数据(blockFormer=0)使 kernel 空转退出,
    // 直接成功。
    auto emptyTensorOutShape0 = context_->GetOutputShape(0);
    if (emptyTensorOutShape0 != nullptr && emptyTensorOutShape0->GetStorageShape().GetShapeSize() == 0) {
        auto emptyRawTiling = context_->GetRawTilingData();
        if (emptyRawTiling != nullptr && emptyRawTiling->GetData() != nullptr) {
            size_t emptyCap = emptyRawTiling->GetCapacity();
            uint8_t* emptyPtr = reinterpret_cast<uint8_t*>(emptyRawTiling->GetData());
            for (size_t emptyIdx = 0; emptyIdx < emptyCap; ++emptyIdx) {
                emptyPtr[emptyIdx] = 0;
            }
            emptyRawTiling->SetDataSize(emptyCap);
        }
        size_t* emptyWs = context_->GetWorkspaceSizes(1);
        if (emptyWs != nullptr) {
            emptyWs[0] = 0;
        }
        context_->SetBlockDim(1);
        tilingKey = GET_TPL_TILING_KEY(1); // schMode=1(已编译), 配合全0 tiling(blockFormer=0)空转
        return ge::GRAPH_SUCCESS;
    }
    auto input0Desc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, input0Desc);
    ge::DataType input0DType = input0Desc->GetDataType();
    if (input0DType == ge::DT_FLOAT16) {
        BroadcastBaseTiling<LambApplyWeightAssignOp::LambApplyWeightAssignCompute<half, float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else if (input0DType == ge::DT_FLOAT) {
        BroadcastBaseTiling<LambApplyWeightAssignOp::LambApplyWeightAssignCompute<float, float>::OpDag> brcBaseTiling(
            context_, static_cast<uint32_t>(BROADCAST_KERNEL_TYPE::KERNEL_TYPE_NDDMA));
        OP_CHECK_IF(brcBaseTiling.DoTiling() == ge::GRAPH_FAILED,
                    OP_LOGE(context_->GetNodeName(), "Do tiling failed. Please check the detailed log."),
                    return ge::GRAPH_FAILED);
        tilingKey = GET_TPL_TILING_KEY(brcBaseTiling.GetSchMode());
    } else {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "input0", Ops::Base::ToString(input0DType).c_str(),
                                  "fp16 or fp32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus LambApplyWeightAssignTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t LambApplyWeightAssignTiling::GetTilingKey() const { return tilingKey; }

ge::graphStatus LambApplyWeightAssignTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambApplyWeightAssignTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus LambApplyWeightAssignTiling::GetPlatformInfo() { return ge::GRAPH_SUCCESS; }

static ge::graphStatus TilingForLambApplyWeightAssign(gert::TilingContext* context)
{
    OP_LOGD("LambApplyWeightAssignTiling", "Enter TilingForLambApplyWeightAssign");
    if (context == nullptr) {
        OP_LOGE("LambApplyWeightAssignTiling", "Tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto compileInfo = reinterpret_cast<const LambApplyWeightAssignCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_LOGD(context, "Enter ascendc LambApplyWeightAssignTiling");
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

IMPL_OP_OPTILING(LambApplyWeightAssign)
    .Tiling(TilingForLambApplyWeightAssign)
    .TilingParse<LambApplyWeightAssignCompileInfo>(TilingPrepareForLambApplyWeightAssign);

REGISTER_OPS_TILING_TEMPLATE(LambApplyWeightAssign, LambApplyWeightAssignTiling,
                             LAMB_APPLY_WEIGHT_ASSIGN_TILING_PRIORITY);
} // namespace optiling
