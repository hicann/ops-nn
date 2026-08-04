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
 * \file normalize_bbox.cpp
 * \brief normalize_bbox kernel entry (template<bool reversedBox> + DTYPE_BOXES dispatch)
 *        Ascend950 / arch35 only. Kept at op_kernel top level so the binary compile
 *        kernel-source resolver (get_kernel_source) locates it at ascendc/<op>/<op>.cpp.
 */

#include "arch35/normalize_bbox.h"
#include "arch35/normalize_bbox_tiling_data.h"
#include "arch35/normalize_bbox_tiling_key.h"

template <bool reversedBox>
__global__ __aicore__ void normalize_bbox(GM_ADDR boxes, GM_ADDR shape_hw, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe tpipe;
    REGISTER_TILING_DEFAULT(NormalizeBBoxTilingData);
    GET_TILING_DATA_WITH_STRUCT(NormalizeBBoxTilingData, tilingData, tiling);
#if defined(DTYPE_BOXES)
    NormalizeBBox::NormalizeBBoxKernel<DTYPE_BOXES, reversedBox> op(tpipe, tilingData);
    op.Init(boxes, shape_hw, y);
    op.Process();
#endif
}
