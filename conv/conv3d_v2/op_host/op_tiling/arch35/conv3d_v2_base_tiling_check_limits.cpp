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
 * \file conv3d_v2_base_tiling_check_limits.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
 
namespace optiling {
namespace conv_ops_tiling {
uint64_t Conv3dBaseTilingV2::GetLoad3dMaxFilterHW()
{
    uint64_t load3dMaxFilterHW = 0;
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW || descInfo_.fMapFormat == ge::Format::FORMAT_NDHWC) {
        load3dMaxFilterHW = LOAD3D_MAX_FILTER_H_W_DAV;
    }
    return load3dMaxFilterHW;
}

ge::graphStatus Conv3dBaseTilingV2::CheckLoad3DLimits()
{
    if (attrInfo_.strideH > LOAD3D_MAX_STRIDE_H_W || attrInfo_.strideW > LOAD3D_MAX_STRIDE_H_W) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Attrs not satisfy Load3D's limits: strideH=%u, strideW=%u, which must <= %lu.",
                paramInfo_.nodeType.c_str(), attrInfo_.strideH, attrInfo_.strideW, LOAD3D_MAX_STRIDE_H_W);
        return ge::GRAPH_FAILED;
    }
    if ((attrInfo_.padTop > LOAD3D_MAX_PAD || attrInfo_.padBottom > LOAD3D_MAX_PAD ||
                attrInfo_.padLeft > LOAD3D_MAX_PAD || attrInfo_.padRight > LOAD3D_MAX_PAD)) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Attrs not satisfy Load3D's limit: pads=[%u, %u, %u, %u], each dim must <= %lu.",
                paramInfo_.nodeType.c_str(), attrInfo_.padTop, attrInfo_.padBottom, attrInfo_.padLeft,
                attrInfo_.padRight, LOAD3D_MAX_PAD);
    }
    if (attrInfo_.dilationH > LOAD3D_MAX_DILATION_H_W || attrInfo_.dilationW > LOAD3D_MAX_DILATION_H_W) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Attrs not satisfy Load3D's limits: dilationH=%u, dilationW=%u, which must <= %lu.",
                paramInfo_.nodeType.c_str(), attrInfo_.dilationH, attrInfo_.dilationW, LOAD3D_MAX_DILATION_H_W);
        return ge::GRAPH_FAILED;
    }
    uint64_t load3dMaxFilterHW = GetLoad3dMaxFilterHW();
    if ((shapeInfo_.kh > load3dMaxFilterHW || shapeInfo_.kw > load3dMaxFilterHW)) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Weight shape not satisfy Load3D's limits: kh=%lu, kw=%lu, which must <= %lu.",
                paramInfo_.nodeType.c_str(), shapeInfo_.kh, shapeInfo_.kw, load3dMaxFilterHW);
        return ge::GRAPH_FAILED;
    }
    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap[descInfo_.fMapDtype], MKN_K_IDX);
    uint64_t weightShapeValue = shapeInfo_.kh * shapeInfo_.kw * k0;
    uint64_t weightShapeLimit = MAX_16_BIT_NUM;
    if (weightShapeValue > weightShapeLimit) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Weight shape not satisfy Load3D's limits: kH(%lu)*kW(%lu)*k0(%u)=%lu, which must <= %lu.",
            paramInfo_.nodeType.c_str(), shapeInfo_.kh, shapeInfo_.kw, k0, weightShapeValue, weightShapeLimit);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckDataCopyLimits()
{
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW) {
        uint64_t loadAL1loop1SrcStride = shapeInfo_.di * shapeInfo_.hi * shapeInfo_.wi *
                                         dtypeSizeTab[descInfo_.fMapDtype];
        if (loadAL1loop1SrcStride > MAX_40_BIT_NUM) {
            OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Fmap shape not satisfy DataCopy's limits: di(%lu)*hi(%lu)*wi(%lu)*typesize(%u)>%lu.",
                paramInfo_.nodeType.c_str(), shapeInfo_.di, shapeInfo_.hi, shapeInfo_.wi,
                dtypeSizeTab[descInfo_.fMapDtype], MAX_40_BIT_NUM);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckFixPipeLimits()
{
    if (descInfo_.fMapFormat == ge::Format::FORMAT_NCDHW) {
        uint64_t fixpipeDstNdStride = shapeInfo_.dout * shapeInfo_.ho * shapeInfo_.wo;
        if (fixpipeDstNdStride > MAX_32_BIT_NUM) {
            OP_LOGE(context_->GetNodeName(),
                    "%s AscendC: Output shape not satisfy Fixpipe's limits: dout(%lu)*hout(%lu)*wout(%lu)>%lu.",
                    paramInfo_.nodeType.c_str(), shapeInfo_.dout, shapeInfo_.ho, shapeInfo_.wo, MAX_32_BIT_NUM);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv3dBaseTilingV2::CheckInstructionLimits()
{
    if (CheckLoad3DLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDataCopyLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFixPipeLimits() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}
}