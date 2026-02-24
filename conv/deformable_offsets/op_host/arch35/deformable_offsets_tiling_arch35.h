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
TILING_DATA_FIELD_DEF(int64_t, blockNum);
TILING_DATA_FIELD_DEF(int64_t, strideHeight);
TILING_DATA_FIELD_DEF(int64_t, strideWidth);
TILING_DATA_FIELD_DEF(int64_t, dilationHeight);
TILING_DATA_FIELD_DEF(int64_t, dilationWidth);
TILING_DATA_FIELD_DEF(int64_t, padsHeight);
TILING_DATA_FIELD_DEF(int64_t, padsWidth);
TILING_DATA_FIELD_DEF(int64_t, dimKHeight);
TILING_DATA_FIELD_DEF(int64_t, dimKWidth);
TILING_DATA_FIELD_DEF(int64_t, imgChannel);
TILING_DATA_FIELD_DEF(int64_t, imgWidth);
TILING_DATA_FIELD_DEF(int64_t, imgHeight);
TILING_DATA_FIELD_DEF(int64_t, imgWidthStride);
TILING_DATA_FIELD_DEF(int64_t, imgOutHeight);
TILING_DATA_FIELD_DEF(int64_t, imgOutWidth);
TILING_DATA_FIELD_DEF(int64_t, offsetKernelElementStride);
TILING_DATA_FIELD_DEF(int64_t, offsetPointStride);
TILING_DATA_FIELD_DEF(int64_t, offsetWidthStride);
TILING_DATA_FIELD_DEF(int64_t, offsetValueDim);
TILING_DATA_FIELD_DEF(int64_t, deformableGroups);
TILING_DATA_FIELD_DEF(int64_t, outputPointWidthStride);
TILING_DATA_FIELD_DEF(int64_t, outputWidthStride);
TILING_DATA_FIELD_DEF(int64_t, outputKernelWidthStride);
TILING_DATA_FIELD_DEF(int64_t, numKernels);
TILING_DATA_FIELD_DEF(int64_t, imgBatchStride);
TILING_DATA_FIELD_DEF(int64_t, offsetBatchStride);
TILING_DATA_FIELD_DEF(int64_t, outputBatchStride);
TILING_DATA_FIELD_DEF(int64_t, imgBatchNum);
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
    int64_t deformableGroupsAttr;
    int64_t offsetValueDim;
};

struct DeformableOffsetsOffset {
    int64_t imgBatchNum;
    int64_t imgChannel;
    int64_t imgWidth;
    int64_t imgHeight;
    int64_t imgWidthStride;
    int64_t imgBatchStride;
    int64_t imgOutHeight;
    int64_t imgOutWidth;
    int64_t offsetBatchStride;
    int64_t deformableGroups;
    int64_t offsetKernelElementStride;
    int64_t offsetPointStride;
    int64_t offsetWidthStride;

    int64_t outputBatchStride;
    int64_t outputPointWidthStride;
    int64_t outputWidthStride;
    int64_t outputKernelWidthStride;

    int64_t numKernels;
    int64_t blockDimValue;
};

REGISTER_TILING_DATA_CLASS(DeformableOffsets, DeformableOffsetsTilingDataSimt)

ge::graphStatus DeformableOffsetTilingSimt(gert::TilingContext* context, int32_t maxCoreNum);
} // namespace optiling
#endif