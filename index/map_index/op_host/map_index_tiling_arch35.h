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
 * \file map_index_tiling_arch35.h
 * \brief
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_MAP_INDEX_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_MAP_INDEX_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MapIndexTilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalCoreNum);
    TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);           // 实际使用的核数
    TILING_DATA_FIELD_DEF(int64_t, normalCoreProcessNum);           // 单核循环次数
    TILING_DATA_FIELD_DEF(int64_t, tailCoreProcessNum);       // 尾核循环次数
    TILING_DATA_FIELD_DEF(int64_t, Dim1Size);                    
    TILING_DATA_FIELD_DEF(int64_t, Dim1SizeAlign);              
    TILING_DATA_FIELD_DEF(int64_t, CopyInDim0);              // 一次最多搬几行
    TILING_DATA_FIELD_DEF(int64_t, CopyInDim0Times);          // 需要搬运几次
    TILING_DATA_FIELD_DEF(int64_t, tailCopyInDim0Times);      // 尾块需要搬运几次
    TILING_DATA_FIELD_DEF(int64_t, doubleBuffNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MapIndex, MapIndexTilingData)

struct MapIndexCompileInfo
{
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct MapIndexTilingParam
{
    int64_t totalCoreNum{ 0 };
    int64_t ubSize { 0 };
    uint32_t vfLen { 0 };
    uint32_t workspaceSize { 0 };
    int64_t usedCoreNum { 0 };
    int64_t normalCoreProcessNum {0};
    int64_t tailCoreProcessNum {0};
    int64_t Dim0Size {0};
    int64_t Dim1Size {0};
    int64_t Dim1SizeAlign {0};
    int64_t CopyInDim0 {1};
    int64_t CopyInDim0Times {1};
    int64_t tailCopyInDim0Times {1};
    int64_t doubleBuffNum {0};
    bool hasLevelIndex = true;
};

}
#endif