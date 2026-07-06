/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_DIM_TILING_ARCH35_H
#define UNIQUE_DIM_TILING_ARCH35_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(UniqueDimTilingData)
TILING_DATA_FIELD_DEF(int64_t, numInp);
TILING_DATA_FIELD_DEF(int64_t, rowLen);
TILING_DATA_FIELD_DEF(int64_t, returnInverse);
TILING_DATA_FIELD_DEF(int64_t, returnCounts);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, elementsPerCore);
TILING_DATA_FIELD_DEF(int64_t, tailCoreElements);
TILING_DATA_FIELD_DEF(int64_t, blockDimX);
TILING_DATA_FIELD_DEF(int64_t, inputDtypeSize);
TILING_DATA_FIELD_DEF(int64_t, tileSize);
TILING_DATA_FIELD_DEF(int64_t, numTiles);
// workspace offsets (relative to user workspace start)
TILING_DATA_FIELD_DEF(int64_t, indicesOffset);
TILING_DATA_FIELD_DEF(int64_t, sortBufOffset);
TILING_DATA_FIELD_DEF(int64_t, flagsOffset);
TILING_DATA_FIELD_DEF(int64_t, positionsOffset);
TILING_DATA_FIELD_DEF(int64_t, partialSumOffset);
TILING_DATA_FIELD_DEF(int64_t, globalPrefixOffset);
TILING_DATA_FIELD_DEF(int64_t, shapeOutOffset);
TILING_DATA_FIELD_DEF(int64_t, workspaceSize);
TILING_DATA_FIELD_DEF(int64_t, dim);
TILING_DATA_FIELD_DEF(int64_t, outerSize);
TILING_DATA_FIELD_DEF(int64_t, innerSize);
TILING_DATA_FIELD_DEF(int64_t, transposeDstOffset);
TILING_DATA_FIELD_DEF(int64_t, inputDimNum);
TILING_DATA_FIELD_DEF(int64_t, inputDim0);
TILING_DATA_FIELD_DEF(int64_t, inputDim1);
TILING_DATA_FIELD_DEF(int64_t, inputDim2);
TILING_DATA_FIELD_DEF(int64_t, inputDim3);
TILING_DATA_FIELD_DEF(int64_t, inputDim4);
TILING_DATA_FIELD_DEF(int64_t, inputDim5);
TILING_DATA_FIELD_DEF(int64_t, inputDim6);
TILING_DATA_FIELD_DEF(int64_t, inputDim7);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UniqueDim, UniqueDimTilingData)

struct UniqueDimCompileInfo {
    int32_t aivCoreNum = 0;
    int32_t sysWorkspaceSize = 0;
    int64_t ubSize = 0;
    int32_t blockSize = 0;
};

class UniqueDimTilingHelper {
public:
    explicit UniqueDimTilingHelper(gert::TilingContext* context) : context_(context) {}
    ~UniqueDimTilingHelper() = default;

    bool DoTiling();
    void SetTilingData(UniqueDimTilingData* tiling);

private:
    bool GetPlatformInfo();
    bool GetShapeInfo();
    bool ReadAttrs();
    bool DoBlockTiling();
    bool ComputeWorkspaces();

    gert::TilingContext* context_;

    // platform info
    uint64_t ubSize_ = 0;
    uint32_t blockSize_ = 1;
    uint32_t aivCoreNum_ = 1;
    uint32_t sysWorkspaceSize_ = 0;

    // tensor info
    ge::DataType dataTypeX_;
    int64_t dtSizeX_ = 0;
    int64_t numInp_ = 0;
    int64_t rowLen_ = 0;
    int64_t totalElements_ = 0;

    // attrs
    bool returnInverse_ = true;
    bool returnCounts_ = true;

    // tiling params
    int64_t usedCoreNum_ = 1;
    int64_t elementsPerCore_ = 1;
    int64_t tailCoreElements_ = 1;
    int64_t tileSize_ = 1024;
    int64_t numTiles_ = 1;
    int64_t blockDimX_ = 256;

    // workspace layout
    int64_t indicesOffset_ = 0;
    int64_t sortBufOffset_ = 0;
    int64_t flagsOffset_ = 0;
    int64_t positionsOffset_ = 0;
    int64_t partialSumOffset_ = 0;
    int64_t globalPrefixOffset_ = 0;
    int64_t shapeOutOffset_ = 0;
    int64_t workspaceSize_ = 0;

    // transpose support for dim!=0
    int64_t dim_ = 0;
    int64_t outerSize_ = 1;
    int64_t innerSize_ = 1;
    int64_t transposeDstOffset_ = 0;

    // original input shape for output shape construction
    int64_t inputDimNum_ = 0;
    int64_t inputDims_[8] = {0};
};

} // namespace optiling

#endif // UNIQUE_DIM_TILING_ARCH35_H
