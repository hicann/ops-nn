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
 * \file max_pool3d_with_argmax_v2_nc_transpose_tiling.h
 * \brief
 */

#ifndef MAX_POOL3D_WITH_AGRMAX_V2_NC_TRANSPOSE_TILING_H_
#define MAX_POOL3D_WITH_AGRMAX_V2_NC_TRANSPOSE_TILING_H_

#include "max_pool3d_with_argmax_v2_tiling_base.h"
#include "../op_kernel/arch35/max_pool3d_with_argmax_v2_tiling_struct.h"

namespace optiling {

struct MaxPool3DWithArgmaxV2NcTransposeBaseInfo {
    int64_t inputBytes = 0;
    int64_t indexBytes = 0;
    int64_t availableUb = 0;
    int64_t totalCoreNum = 0;
    int64_t oneBlockNumT1 = 0;
    int64_t oneBlockNumT2 = 0;
    int64_t vlNumT1 = 0;

    int64_t dKernel = 0;
    int64_t hKernel = 0;
    int64_t wKernel = 0;
    int64_t padFront = 0;
    int64_t padTop = 0;
    int64_t padLeft = 0;
    int64_t dInput = 0;
    int64_t hInput = 0;
    int64_t wInput = 0;
    int64_t dStride = 0;
    int64_t hStride = 0;
    int64_t wStride = 0;
    int64_t dOutput = 0;
    int64_t hOutput = 0;
    int64_t wOutput = 0;

    int64_t isPad = 0;
    int64_t highAxisTotal = 0;
};

struct MaxPool3DWithArgmaxV2NcTransposeSplitInfo {
    int64_t ncInner = 0;
    int64_t ncTail = 0;
    int64_t ncOuter = 0;

    int64_t hOutputOuter = 0;
    int64_t wOutputInner = 0;
    int64_t wOutputTail = 0;
    int64_t wOutputOuter = 0;
    int64_t dOutputInner = 0;
    int64_t dOutputTail = 0;
    int64_t dOutputOuter = 0;
    int64_t hOutputInner = 0;
    int64_t hOutputTail = 0;
    int64_t dInputInner = 0;
    int64_t hInputInner = 0;
    int64_t wInputInner = 0;

    int64_t totalBaseBlockNum = 0;
    int64_t normalCoreProcessNum = 0;
    int64_t tailCoreProcessNum = 0;
    int64_t usedCoreNum = 0;

    int64_t inputBufferSize = 0;
    int64_t transBufferSize = 0;
    int64_t maxValueBufferSize = 0;
    int64_t argmaxBufferSize = 0;
    int64_t totalBufferSize = 0;
    int64_t deltaIndexBufferSize = 0;
    int64_t castBufferSize = 0;
};

class MaxPool3DWithArgmaxV2NcTransposeTiling : public MaxPool3DWithArgmaxV2BaseTiling {
public:
    explicit MaxPool3DWithArgmaxV2NcTransposeTiling(gert::TilingContext* context)
        : MaxPool3DWithArgmaxV2BaseTiling(context)
    {}

    ~MaxPool3DWithArgmaxV2NcTransposeTiling() override {}

private:
    void InitializationVars();
    void DoUBTiling();
    void DoBlockTiling();
    void DoBufferCalculate();
    bool IsMeetUBSize();
    bool IsMeetTargetCoreNum() const;
    void SearchBestTiling();
    void BinarySearch(int64_t start, int64_t end, int64_t* value);
    bool TrySplitNC();
    bool TrySplitD();
    bool TrySplitH();
    bool TrySplitW();
    void SetTilingData();
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus ValidatePlatformAndInput();
    ge::graphStatus ParseShapeAndFormat();
    ge::graphStatus ParseKernelStrideAttrs(const gert::RuntimeAttrs* runtimeAttrs);
    ge::graphStatus ParsePadDilationAttrs(const gert::RuntimeAttrs* runtimeAttrs);
    ge::graphStatus ParseCeilModeAndDtype(const gert::RuntimeAttrs* runtimeAttrs);

    MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData* tilingData_ = context_->GetTilingData<
        MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2NcTransposeTilingData>();
    MaxPool3DWithArgmaxV2NcTransposeBaseInfo baseData_;
    MaxPool3DWithArgmaxV2NcTransposeSplitInfo splitData_;
};

} // namespace optiling

#endif