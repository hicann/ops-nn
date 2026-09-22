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
 * \file cla_gate_quant_tilingdata.h
 * \brief Kernel-side TilingData plain struct for ClaGateQuant
 */

#ifndef OPS_NN_CLA_GATE_QUANT_TILINGDATA_H
#define OPS_NN_CLA_GATE_QUANT_TILINGDATA_H

struct ClaGateQuantTilingData {
    int64_t usedCoreNum;          // Number of active vector cores.
    int64_t rowCount;             // Input row count T.
    int64_t rowLength;            // Flattened row length K = N * D.
    int64_t colTileNum;           // Number of 256-element column tiles in dual-axis mode.
    int64_t colTailSize;          // Valid elements in the final dual-axis column tile.
    int64_t headCount;            // Head count N.
    int64_t headDim;              // Elements per head, either 128 or 256.
    int64_t streamTailSize;       // Valid elements in the final single-axis stream segment.
    int64_t batchSegmentCapacity; // Maximum single-axis segments resident in one UB batch.
    int64_t baseTaskCount;        // Base number of tasks assigned to each core.
    int64_t extraTaskCoreCount;   // Number of cores assigned one additional task.
};

#endif // OPS_NN_CLA_GATE_QUANT_TILINGDATA_H
