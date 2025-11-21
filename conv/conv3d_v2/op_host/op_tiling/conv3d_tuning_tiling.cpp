/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "conv3d_tuning_tiling.h"

namespace tuningtiling {
DECLARE_STRUCT_RELATE_WITH_OP_V2(Conv3D, Conv3DInputArgs, aDtype, bDtype, cDtype, biasDtype, aShapeN, aShapeD,
                              aShapeH, aShapeW, bShapeN, bShapeC, bShapeD, bShapeH, bShapeW, cShapeD,
                              cShapeH, cShapeW, aFormat, bFormat, cFormat, groups, strideD,
                              strideH, strideW, dilationD, dilationH, dilationW, padHead, padTail, padTop,
                              padBottom, padLeft, padRight, biasFlag);

REGISTER_TUNING_TILING_CLASS(Conv3D, Conv3DTunnerTiling);
}  // namespace tuningtiling