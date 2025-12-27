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
 * \file unique_consecutive_tiling_arch35.h
 * \brief
 */
#ifndef UNIQUE_CONSECUTIVE_TILING_H
#define UNIQUE_CONSECUTIVE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(UniqueConsecutiveTilingData)
TILING_DATA_FIELD_DEF(int64_t, totalSize);
TILING_DATA_FIELD_DEF(int64_t, useCoreNums);
TILING_DATA_FIELD_DEF(int64_t, tileLengthPerCore);
TILING_DATA_FIELD_DEF(int64_t, tileLengthTailCore);
TILING_DATA_FIELD_DEF(int64_t, adjUbTileLength);
TILING_DATA_FIELD_DEF(int64_t, valueQueueSize);
TILING_DATA_FIELD_DEF(int64_t, countQueueSize);
TILING_DATA_FIELD_DEF(int64_t, idxCopyInQueueSize);
TILING_DATA_FIELD_DEF(int64_t, collectingCntBufSize);
TILING_DATA_FIELD_DEF(int64_t, offsetCntBufSize);
TILING_DATA_FIELD_DEF(int64_t, prevIdxBufSize);
TILING_DATA_FIELD_DEF(int64_t, shapeBufSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UniqueConsecutive, UniqueConsecutiveTilingData)

struct UniqueConsecutiveCompileInfo {
    int32_t aivCoreNum_ = 0;
    int32_t sysWorkspaceSize_ = 0;
    int64_t ubSize_ = 0;
    int32_t vecRegSize_ = 0;
    int32_t blockSize_ = 0;
};

class UniqueConsecutiveTilingHelper
{
public:
    explicit UniqueConsecutiveTilingHelper(gert::TilingContext* context) : context_(context)
    {
    }
    ~UniqueConsecutiveTilingHelper() = default;
    bool DoTiling();
    void SetTilingDataAndTilingKeyAndWorkSpace(UniqueConsecutiveTilingData* tiling);

    bool GetBaseInfo();
    bool GetShapeInfo();
    bool DoBlockTiling();
    bool ComputeWorkspaces();
    bool DoUbTiling();

private:
    bool CheckTensorAndAttr();
    bool GetPlatformInfo();
    bool GetAttrs();

private:
    gert::TilingContext* context_{nullptr};

    // soc info
    uint64_t ubSize_{1};
    uint32_t blockSize_{1};
    uint32_t vecRegSize_{1};
    uint32_t aivCoreNum_{1};
    uint64_t sysWorkspaceSize_{1};

    // attrs
    bool retCounts_{true};
    ge::DataType outIdxDtype_{ge::DataType::DT_INT64};

    // tensor info
    ge::DataType dataTypeX_{ge::DataType::DT_BF16};
    int64_t dtSizeX_{2};
    int64_t totalSize_{1};
    bool isInt64_{false};
    
    //UbTiling nums
    int64_t adjUbTileLength_{1};
    int64_t tileLength_{1};
    int64_t valueQueueSize_{1};
    int64_t countQueueSize_{1};
    int64_t idxCopyInQueueSize_{1};
    int64_t collectingCntBufSize_{1};
    int64_t offsetCntBufSize_{1};
    int64_t prevIdxBufSize_{1};
    int64_t shapeBufSize_{1};
    int64_t byteSize_{1};
    int64_t tempUbSize_{1};

    // BlockTiling nums
    int64_t useCoreNums_{1};
    int64_t tileLengthPerCore_{1};
    int64_t tileLengthTailCore_{1};

    // worksapce vars
    int64_t idxWorkSpace_{1};
    int64_t valueWorkSpace_{1};
    int64_t coreWorkSpace_{1};

    // debug vars
    int64_t debugOnlyMaxSingleCoreBytes_{1024};
};

}  // namespace optiling

#endif  // UNIQUE_CONSECUTIVE_TILING_H