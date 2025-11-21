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
 * \file conv3d_backprop_filter_v2_stream_k_tiling.h
 * \brief
 */
#ifndef CONV3D_DW_STREAM_K_TILING_H
#define CONV3D_DW_STREAM_K_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "conv3d_backprop_filter_v2_basic_block_tiling.h"
#include "conv3d_backprop_filter_v2_common.h"

namespace Ops {
namespace NN {
namespace Conv {
class Conv3DBackpropFilterV2StreamKTiling : public Conv3DDWV2BasicBlockTilingArch35 {
public:
    explicit Conv3DBackpropFilterV2StreamKTiling(gert::TilingContext *context) : Conv3DDWV2BasicBlockTilingArch35(context) {
        Reset();
    }
    ~Conv3DBackpropFilterV2StreamKTiling() override = default;

protected:
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

private:
    void InitSplitWOI();
    void AdjustSmallCaseBaseBlock();
    uint64_t GetSingleShapeKByStreamK();
    bool IsSplitBatchDoutBetter();
    void DoStreamKTiling();
    void DoStreamkByBatchDout();
    void DoStreamkByHWout();
};
}
}
}
#endif  // CONV3D_DW_BASIC_BLOCK_TILING_H