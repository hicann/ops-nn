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
 * \file deformable_offsets_tiling_arch35.h
 * \brief deformable_offsets_tiling_arch35 info
 */

#ifndef DEFORMABLE_OFFSETS_TILING_ARCH35_H
#define DEFORMABLE_OFFSETS_TILING_ARCH35_H

#include <cstdint>

#include "util/shape_util.h"
#include "register/tilingdata_base.h"
#include "op_common/op_host/util/opbase_export.h"
#include "register/op_impl_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeformableOffsetsTilingDataSimt);
TILING_DATA_FIELD_DEF(uint32_t, blockNum);
TILING_DATA_FIELD_DEF(uint32_t, strideHeight);
TILING_DATA_FIELD_DEF(uint32_t, strideWidth);
TILING_DATA_FIELD_DEF(uint32_t, dilationHeight);
TILING_DATA_FIELD_DEF(uint32_t, dilationWidth);
TILING_DATA_FIELD_DEF(uint32_t, padsHeight);
TILING_DATA_FIELD_DEF(uint32_t, padsWidth);
TILING_DATA_FIELD_DEF(uint32_t, dimKHeight);
TILING_DATA_FIELD_DEF(uint32_t, dimKWidth);
TILING_DATA_FIELD_DEF(uint32_t, imgChannel);
TILING_DATA_FIELD_DEF(uint32_t, imgWidth);
TILING_DATA_FIELD_DEF(uint32_t, imgHeight);
TILING_DATA_FIELD_DEF(uint32_t, imgWidthStride);
TILING_DATA_FIELD_DEF(uint32_t, imgOutHeight);
TILING_DATA_FIELD_DEF(uint32_t, imgOutWidth);
TILING_DATA_FIELD_DEF(uint32_t, offsetKernelElementStride);
TILING_DATA_FIELD_DEF(uint32_t, offsetPointStride);
TILING_DATA_FIELD_DEF(uint32_t, offsetWidthStride);
TILING_DATA_FIELD_DEF(uint32_t, offsetValueDim);
TILING_DATA_FIELD_DEF(uint32_t, deformableGroups);
TILING_DATA_FIELD_DEF(uint32_t, outputPointWidthStride);
TILING_DATA_FIELD_DEF(uint32_t, outputWidthStride);
TILING_DATA_FIELD_DEF(uint32_t, outputKernelWidthStride);
TILING_DATA_FIELD_DEF(uint32_t, numKernels);
TILING_DATA_FIELD_DEF(uint32_t, imgBatchStride);
TILING_DATA_FIELD_DEF(uint32_t, offsetBatchStride);
TILING_DATA_FIELD_DEF(uint32_t, outputBatchStride);
TILING_DATA_FIELD_DEF(uint32_t, imgBatchNum);
END_TILING_DATA_DEF;

struct TilingPrepareForDeformableOffsetsCompileInfo {
    int64_t coreNum;
    int64_t ubSize;
};

struct DeformableOffsetAttr {
    int64_t strideH;
    int64_t strideW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t padsHeightUp;
    int64_t padsHeightDown;
    int64_t padsWidthLeft;
    int64_t padsWidthRight;
    int64_t dimKh;
    int64_t dimKw;
    uint32_t deformableGroupsAttr;
    uint32_t offsetValueDim;
};

struct DeformableOffsetsOffset {
    uint32_t imgBatchNum;
    uint32_t imgChannel;
    uint32_t imgWidth;
    uint32_t imgHeight;
    uint32_t imgWidthStride;
    uint32_t imgBatchStride;
    uint32_t imgOutHeight;
    uint32_t imgOutWidth;
    uint32_t offsetBatchStride;
    uint32_t deformableGroups;
    uint32_t offsetKernelElementStride;
    uint32_t offsetPointStride;
    uint32_t offsetWidthStride;

    uint32_t outputBatchStride;
    uint32_t outputPointWidthStride;
    uint32_t outputWidthStride;
    uint32_t outputKernelWidthStride;

    uint32_t numKernels;
    uint32_t blockDimValue;
};

REGISTER_TILING_DATA_CLASS(DeformableOffsets, DeformableOffsetsTilingDataSimt)

ge::graphStatus DeformableOffsetTilingSimt(gert::TilingContext* context, int32_t maxCoreNum);
} // namespace optiling
#endif