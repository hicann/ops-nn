/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file edge_softmax.cpp
 * \brief edge_softmax kernal
 */

#include "edge_softmax.h

using namespace AscendC;

using namespace EdgeSoftmax;


extern "C" __global__ __aicore__ void edge_softmax(GM_ADDR x, GM_ADDR idx, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    KernelEdgeSoftmax op;
    op.Init(x, idx, y, tiling);
    op.process();
}