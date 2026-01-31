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
 * \file conv2d_v2_base_tiling_check_limits.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {

ge::graphStatus Conv2dBaseTiling::CheckLoad3DLimits()
{
    // LOAD3D limits
    if (attrInfo_.strideH > LOAD3D_MAX_STRIDE_H_W || attrInfo_.strideW > LOAD3D_MAX_STRIDE_H_W) {
        OP_LOGD(context_->GetNodeName(),
                "%s AscendC: Attrs not satisfy Load3D's limits: strideH=%u, strideW=%u, which must <= %u.",
                paramInfo_.nodeType.c_str(), attrInfo_.strideH, attrInfo_.strideW, LOAD3D_MAX_STRIDE_H_W);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.padTop > LOAD3D_MAX_PAD || attrInfo_.padBottom > LOAD3D_MAX_PAD ||
        attrInfo_.padLeft > LOAD3D_MAX_PAD || attrInfo_.padRight > LOAD3D_MAX_PAD) {
        OP_LOGD(context_->GetNodeName(),
                "%s AscendC: Attrs not satisfy Load3D's limit: pads=[%u, %u, %u, %u], each dim must <= %u.",
                paramInfo_.nodeType.c_str(), attrInfo_.padTop, attrInfo_.padBottom, attrInfo_.padLeft,
                attrInfo_.padRight, LOAD3D_MAX_PAD);
        return ge::GRAPH_FAILED;
    }

    if (attrInfo_.dilationH > LOAD3D_MAX_DILATION_H_W || attrInfo_.dilationW > LOAD3D_MAX_DILATION_H_W) {
        OP_LOGD(context_->GetNodeName(),
                "%s AscendC: Attrs not satisfy Load3D's limits: dilationH=%u, dilationW=%u, which must <= %u.",
                paramInfo_.nodeType.c_str(), attrInfo_.dilationH, attrInfo_.dilationW, LOAD3D_MAX_DILATION_H_W);
        return ge::GRAPH_FAILED;
    }

    if (shapeInfo_.kh > LOAD3D_MAX_FILTER_H_W || shapeInfo_.kw > LOAD3D_MAX_FILTER_H_W) {
        OP_LOGD(context_->GetNodeName(),
                "%s AscendC: Weight shape not satisfy Load3D's limits: kh=%lu, kw=%lu, which must <= %lu.",
                paramInfo_.nodeType.c_str(), shapeInfo_.kh, shapeInfo_.kw, LOAD3D_MAX_FILTER_H_W);
        return ge::GRAPH_FAILED;
    }

    auto k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.weightDtype), MKN_K_IDX);
    uint64_t load3dPoskLimit = MAX_16_BIT_NUM;
    uint64_t load3dPosk = shapeInfo_.kh * shapeInfo_.kw * k0;
    if (load3dPosk >= load3dPoskLimit) {
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: Weight shape not satisfy Load3D's limits: kH(%lu)*kW(%lu)*k0(%u)=%lu, which must <= %lu.",
            paramInfo_.nodeType.c_str(), shapeInfo_.kh, shapeInfo_.kw, k0, load3dPosk, load3dPoskLimit);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}


ge::graphStatus Conv2dBaseTiling::CheckL1SizeLimitsKernelFullLoad()
{
    uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t biasDtypeSize = dtypeSizeTab.at(descInfo_.biasDtype);
    uint64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);
    uint64_t nBL1min = convOpsConstParams_.n0;
    uint64_t biasUsedL1Size = flagInfo_.hasBias ? ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = ConvAlignB(
        static_cast<uint64_t>(nBL1min * fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE), C0_SIZE);
    uint64_t kBL1min = convOpsConstParams_.k0 * shapeInfo_.kh * shapeInfo_.kw;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);

    uint64_t fmapUsedL1Size = 0;
    uint64_t hoAL1min = std::min(shapeInfo_.wo < convOpsConstParams_.m0 ?
                                 ConvCeilDiv(convOpsConstParams_.m0, shapeInfo_.wo) : 1, shapeInfo_.ho);
    uint64_t hiAL1min = ConvInferHiL1(hoAL1min, shapeInfo_.hi, shapeInfo_.kh, attrInfo_.dilationH, attrInfo_.strideH);
    uint64_t kAL1min = convOpsConstParams_.k0;
    uint64_t woAL1min = convOpsConstParams_.m0;
    uint64_t wiAL1min = ConvInferWiL1(woAL1min, shapeInfo_.wi, shapeInfo_.kw, attrInfo_.dilationW, attrInfo_.strideW);
    fmapUsedL1Size = ConvAlignB(hiAL1min * wiAL1min * kAL1min * fMapDtypeSize, C0_SIZE);
 
    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;
    if (minL1LoadSize > opInfo_->l1Size) {
        OP_LOGD(context_->GetNodeName(),
            "%s AscendC: KernelSplitMinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                context_->GetNodeType(), minL1LoadSize, opInfo_->l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckL1SizeLimitsKernelSplit()
{
    uint64_t fMapDtypeSize = dtypeSizeTab.at(descInfo_.fMapDtype);
    uint64_t biasDtypeSize = dtypeSizeTab.at(descInfo_.biasDtype);
    uint64_t weightDtypeSize = dtypeSizeTab.at(descInfo_.weightDtype);
    uint64_t nBL1min = convOpsConstParams_.n0;
    uint64_t biasUsedL1Size = flagInfo_.hasBias ? ConvAlignB(nBL1min * biasDtypeSize, C0_SIZE) : 0;
    uint64_t scaleUsedL1Size = ConvAlignB(
        static_cast<uint64_t>(nBL1min * fixpipeInfo_.channelWiseCoeff * FP16_DTYPE_SIZE), C0_SIZE);
    uint64_t kBL1min = convOpsConstParams_.k0;
    uint64_t weightUsedL1Size = ConvAlignB(kBL1min * nBL1min * weightDtypeSize, C0_SIZE);

    uint64_t fmapUsedL1Size = 0;
    uint64_t hoAL1min = 1;
    uint64_t kAL1min = convOpsConstParams_.k0;
    uint64_t woAL1min = convOpsConstParams_.m0;
    fmapUsedL1Size = ConvAlignB(hoAL1min * woAL1min * kAL1min * fMapDtypeSize, C0_SIZE);

    uint64_t minL1LoadSize = biasUsedL1Size + scaleUsedL1Size + fmapUsedL1Size + weightUsedL1Size;

    if (minL1LoadSize > opInfo_->l1Size) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: KernelFullLoadMinL1LoadSize > L1size, current L1size: %lu, maxL1Size: %lu",
                context_->GetNodeType(), minL1LoadSize, opInfo_->l1Size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckDmaLimits()
{
    uint64_t woAL1Min = convOpsConstParams_.m0;
    uint64_t hoAL1Min = 1;

    uint64_t fmapUbSizeMin =
        ConvAlignB(hoAL1Min * woAL1Min * convOpsConstParams_.k0 * dtypeSizeTab.at(descInfo_.fMapDtype), C0_BYTE_SIZE);
    if (fmapUbSizeMin > opInfo_->ubSize) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: DMA min ub size not enough: platformInfo.ubSize=%lu, fmapUbSizeMin=%lu.",
            paramInfo_.nodeType.c_str(), opInfo_->ubSize, fmapUbSizeMin);
        return ge::GRAPH_FAILED;
    }

    return CheckL1SizeLimitsKernelSplit();
}

ge::graphStatus Conv2dBaseTiling::CheckInstructionLimits()
{
    // DataCopy limits
    uint64_t loadAL1loop1SrcStrideLimits = MAX_40_BIT_NUM;
    uint64_t loadAL1loop1SrcStride = shapeInfo_.hi * shapeInfo_.wi * dtypeSizeTab.at(descInfo_.fMapDtype);
    if (descInfo_.fMapFormat == ge::FORMAT_NCHW && loadAL1loop1SrcStride > loadAL1loop1SrcStrideLimits) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Fmap shape exceeds DataCopy's limits: hi(%lu)*wi(%lu)*typesize(%u)=%lu, which must <= %lu",
            paramInfo_.nodeType.c_str(), shapeInfo_.hi, shapeInfo_.wi, dtypeSizeTab.at(descInfo_.fMapDtype),
            loadAL1loop1SrcStride, loadAL1loop1SrcStrideLimits);
        return ge::GRAPH_FAILED;
    }

    // FixPipe limits
    uint64_t fixpipeLoop2DstStrideLimit = MAX_32_BIT_NUM;
    uint64_t fixpipeLoop2DstStride = shapeInfo_.ho * shapeInfo_.wo;
    if (fixpipeLoop2DstStride > fixpipeLoop2DstStrideLimit) {
        OP_LOGE(context_->GetNodeName(),
                "%s AscendC: Output shape not satisfy Fixpipe's limits: hout(%lu)*wout(%lu)=%lu, which must <= %lu",
                paramInfo_.nodeType.c_str(), shapeInfo_.ho, shapeInfo_.wo, fixpipeLoop2DstStride, fixpipeLoop2DstStrideLimit);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckDisContinuousInstrLimits()
{
    // disContinuous ND2NZ limits
    uint64_t srcDValue = shapeInfo_.ci * shapeInfo_.batch;
    if (flagInfo_.disContinuousFlag && srcDValue > SRC_D_VALUE_MAX) {
        OP_LOGE(context_->GetNodeName(), 
                "%s AscendC: disContinuous input exceeds ND2NZ's limits: ci(%lu)*batch(%lu)=%lu, which must <= %lu",
                paramInfo_.nodeType.c_str(), shapeInfo_.ci, shapeInfo_.batch, srcDValue, SRC_D_VALUE_MAX);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckL0c2GmNZ2NDInstrLimits() {
    // loop3_dst_stride limits check
    uint64_t fixpipeLoop3DstStrideLimit = MAX_32_BIT_NUM;
    uint64_t fixpipeLoop3DstStride = shapeInfo_.wo * shapeInfo_.co;
    if (fixpipeLoop3DstStride > fixpipeLoop3DstStrideLimit) {
        OP_LOGE(context_->GetNodeName(),
            "%s AscendC: Output shape not satisfy Fixpipe's limits: wout(%lu)*cout(%lu)=%lu, which must <= %lu",
            paramInfo_.nodeType.c_str(), shapeInfo_.wo, shapeInfo_.co, fixpipeLoop3DstStride, fixpipeLoop3DstStrideLimit);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Conv2dBaseTiling::CheckNHWCDataCopyLimits() {
    if (paramInfo_.paramsFormat[paramInfo_.FMAP_PARAM_IDX] != ge::FORMAT_NHWC) {
        return ge::GRAPH_SUCCESS;
    }
    return CheckL0c2GmNZ2NDInstrLimits();
}

}
}