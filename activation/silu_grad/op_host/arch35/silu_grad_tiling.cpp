/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file silu_grad_tiling.cpp
 * \brief
 */

#include "silu_grad_tiling.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "atvoss/broadcast/broadcast_tiling.h"

using namespace ge;

namespace optiling {

static constexpr uint64_t OP_KEY_INVALID = 0;
static constexpr uint64_t OP_KEY_1 = 1;
static constexpr uint64_t OP_KEY_2 = 2;
static constexpr uint64_t OP_KEY_3 = 3;
static constexpr uint64_t OP_KEY_4 = 4;
static constexpr uint64_t OP_KEY_5 = 5;
static constexpr uint64_t OP_KEY_6 = 6;
static constexpr uint64_t OP_KEY_7 = 7;
static constexpr uint64_t OP_KEY_8 = 8;
static constexpr uint64_t OP_KEY_9 = 9;
static constexpr uint64_t INDEX_0 = 0;
static constexpr uint64_t INDEX_1 = 1;
static constexpr uint64_t INDEX_2 = 2;
static constexpr uint64_t WORKSPACE_SIZE = 32;

ge::graphStatus SiluGradTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const SiluGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
                        return ge::GRAPH_FAILED);
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

uint64_t SiluGradTiling::GetOpKey(ge::DataType dyDtype, ge::DataType xDtype, ge::DataType dxDtype)
{
    bool opKey1Flag = dyDtype == DT_FLOAT16 && xDtype == DT_FLOAT16 && dxDtype == DT_FLOAT16;
    if (opKey1Flag) {
        return OP_KEY_1;
    }
    bool opKey2Flag = dyDtype == DT_BF16 && xDtype == DT_BF16 && dxDtype == DT_BF16;
    if (opKey2Flag) {
        return OP_KEY_2;
    }
    bool opKey3Flag = dyDtype == DT_FLOAT && xDtype == DT_FLOAT && dxDtype == DT_FLOAT;
    if (opKey3Flag) {
        return OP_KEY_3;
    }
    bool opKey4Flag = dyDtype == DT_FLOAT16 && xDtype == DT_BF16 && dxDtype == DT_FLOAT;
    if (opKey4Flag) {
        return OP_KEY_4;
    }
    bool opKey5Flag = dyDtype == DT_FLOAT16 && xDtype == DT_FLOAT && dxDtype == DT_FLOAT;
    if (opKey5Flag) {
        return OP_KEY_5;
    }
    bool opKey6Flag = dyDtype == DT_BF16 && xDtype == DT_FLOAT16 && dxDtype == DT_FLOAT;
    if (opKey6Flag) {
        return OP_KEY_6;
    }
    bool opKey7Flag = dyDtype == DT_BF16 && xDtype == DT_FLOAT && dxDtype == DT_FLOAT;
    if (opKey7Flag) {
        return OP_KEY_7;
    }
    bool opKey8Flag = dyDtype == DT_FLOAT && xDtype == DT_FLOAT16 && dxDtype == DT_FLOAT;
    if (opKey8Flag) {
        return OP_KEY_8;
    }
    bool opKey9Flag = dyDtype == DT_FLOAT && xDtype == DT_BF16 && dxDtype == DT_FLOAT;
    if (opKey9Flag) {
        return OP_KEY_9;
    }

    return OP_KEY_INVALID;
}

uint64_t SiluGradTiling::GenerateTilingKey(uint64_t innerKey)
{
    return opKey * Ops::Base::BROADCAST_OP_KEY_OFFSET + innerKey;
}

std::map<uint64_t, Ops::Base::BroadcastComputeParams> SiluGradTiling::GetComputeMap(uint64_t opKeyParam)
{
    Ops::Base::BroadcastComputeParams computeParams0;
        switch (opKeyParam) {
        case OP_KEY_1:
        case OP_KEY_2:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {192, 192};
            return {{1, computeParams0}};
        case OP_KEY_3:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {192, 192};
            return {{1, computeParams0}};
        case OP_KEY_4:
        case OP_KEY_6:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {128, 128};
            return {{1, computeParams0}};
        case OP_KEY_5:
        case OP_KEY_7:
        case OP_KEY_8:
        case OP_KEY_9:
            computeParams0.maxDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS32_SIZE);
            computeParams0.minDtypeBits = static_cast<int64_t>(Ops::Base::BROADCAST_BITS_SIZE::BITS16_SIZE);
            computeParams0.extraSize = {0, 0};
            computeParams0.bufferDivisor = {160, 160};
            return {{1, computeParams0}};
        default:
            return {};
    }
}

ge::graphStatus SiluGradTiling::GetShapeAttrsInfo()
{
    auto dy = context_->GetInputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dy);
    auto dyDtype = dy->GetDataType();
    auto x = context_->GetInputDesc(INDEX_1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, x);
    auto xDtype = x->GetDataType();
    auto dx = context_->GetOutputDesc(INDEX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dx);
    auto dxDtype = dx->GetDataType();

    opKey = GetOpKey(dyDtype, xDtype, dxDtype);
    OP_CHECK_IF((opKey == OP_KEY_INVALID),
                    OP_LOGE(context_->GetNodeName(), "can not get opKey"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool SiluGradTiling::IsCapable()
{
    return true;
}

ge::graphStatus SiluGradTiling::DoOpTiling()
{
    Ops::Base::BroadcastTilingParams broadcastTilingParams;
    for (uint64_t i = 0; i < context_->GetComputeNodeInputNum(); i++) {
        auto shape = context_->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shape);
        broadcastTilingParams.inShape.push_back(Ops::Base::EnsureNotScalar(shape->GetStorageShape()));
    }

    auto outShape = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outShape);
    broadcastTilingParams.outShape = Ops::Base::EnsureNotScalar(outShape->GetStorageShape());
    broadcastTilingParams.computeMap = GetComputeMap(opKey);
    broadcastTilingParams.coreNum = coreNum;
    broadcastTilingParams.ubSize = ubSize;

    Ops::Base::BroadcastTilingData broadcastTilingData;
    ge::graphStatus status = BroadcastTiling(broadcastTilingParams, broadcastTilingData);
    if (status != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "broadcast tiling failed.");
        return ge::GRAPH_FAILED;
    }

    tilingKey_ = GenerateTilingKey(broadcastTilingData.innerKey);
    blockNum = broadcastTilingData.blockNum;
    tilingData.set_blockFormer(broadcastTilingData.blockFormer);
    tilingData.set_ubFormer(broadcastTilingData.ubFormer);
    tilingData.set_ubOuter(broadcastTilingData.ubOuter);
    tilingData.set_ubTail(broadcastTilingData.ubTail);
    tilingData.set_blockTail(broadcastTilingData.blockTail);
    tilingData.set_shapeLen(broadcastTilingData.shapeLen);
    tilingData.set_ubSplitAxis(broadcastTilingData.ubSplitAxis);
    tilingData.set_dimProductBeforeUbInner(broadcastTilingData.dimProductBeforeUbInner);
    tilingData.set_elemNum(broadcastTilingData.elemNum);

    std::copy(broadcastTilingData.dims[INDEX_0].begin(), broadcastTilingData.dims[INDEX_0].end(), input0Dims);
    tilingData.set_input0Dims(input0Dims);
    std::copy(broadcastTilingData.dims[INDEX_1].begin(), broadcastTilingData.dims[INDEX_1].end(), input1Dims);
    tilingData.set_input1Dims(input1Dims);
    std::copy(broadcastTilingData.dims[INDEX_2].begin(), broadcastTilingData.dims[INDEX_2].end(), outputDims);
    tilingData.set_outputDims(outputDims);
    std::copy(broadcastTilingData.strides[INDEX_0].begin(), broadcastTilingData.strides[INDEX_0].end(), input0Strides);
    tilingData.set_input0Strides(input0Strides);
    std::copy(broadcastTilingData.strides[INDEX_1].begin(), broadcastTilingData.strides[INDEX_1].end(), input1Strides);
    tilingData.set_input1Strides(input1Strides);
    std::copy(broadcastTilingData.strides[INDEX_2].begin(), broadcastTilingData.strides[INDEX_2].end(), outputStrides);
    tilingData.set_outputStrides(outputStrides);

    return ge::GRAPH_SUCCESS;
}

std::string SiluGradTiling::ToString(SiluGradTilingData &tilingDataParam) {
    std::string str;
    str += " blockFormer:" + std::to_string(tilingDataParam.get_blockFormer());
    str += " ubFormer:" + std::to_string(tilingDataParam.get_ubFormer());
    str += " ubOuter:" + std::to_string(tilingDataParam.get_ubOuter());
    str += " ubTail:" + std::to_string(tilingDataParam.get_ubTail());
    str += " blockTail:" + std::to_string(tilingDataParam.get_blockTail());
    str += " shapeLen:" + std::to_string(tilingDataParam.get_shapeLen());
    str += " ubSplitAxis:" + std::to_string(tilingDataParam.get_ubSplitAxis());
    str += " dimProductBeforeUbInner:" + std::to_string(tilingDataParam.get_dimProductBeforeUbInner());
    str += " elemNum:" + std::to_string(tilingDataParam.get_elemNum());
    return str;
}

ge::graphStatus SiluGradTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SiluGradTiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus SiluGradTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SiluGradTiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    context_->SetBlockDim(blockNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(context_, "TilingInfo: %s.", ToString(tilingData).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSiluGrad(gert::TilingContext* context)
{
    auto compileInfo = reinterpret_cast<const SiluGradCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    SiluGradTiling tiling(context);
    return tiling.DoTiling();
}

ge::graphStatus TilingPrepareForSiluGrad(gert::TilingParseContext* context)
{
    auto compileInfoPtr = context->GetCompiledInfo<SiluGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SiluGrad)
    .Tiling(TilingForSiluGrad)
    .TilingParse<SiluGradCompileInfo>(TilingPrepareForSiluGrad);

}  // namespace optiling
