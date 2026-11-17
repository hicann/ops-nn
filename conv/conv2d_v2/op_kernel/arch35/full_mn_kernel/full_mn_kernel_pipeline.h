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
 * \file full_mn_kernel_pipeline.h
 * \brief Conv2dFullMNKernel + Conv2dFullMNKernelPipeline: entry-point templates.
 */

#ifndef FULL_MN_KERNEL_PIPELINE_H
#define FULL_MN_KERNEL_PIPELINE_H

#include "kernel_tiling/kernel_tiling.h"
#include "full_mn_kernel_common.h"
#include "full_mn_kernel_block_ctx.h"
#include "full_mn_kernel_scheduler.h"
#include "full_mn_kernel_processor.h"

namespace full_mn {

template <typename Scheduler, typename Processor, typename BlockCtx, typename TilingData>
struct Conv2dFullMNKernel {
    __aicore__ inline void operator()(const TilingData& tiling, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      GM_ADDR workspace) const
    {
        if (!isActive(tiling))
            return;
        Scheduler scheduler;
        BlockCtx bc = scheduler(tiling);
        if (bc.actualCo == 0 || bc.actualM == 0)
            return;
        Processor processor;
        processor(bc, tiling, x, filter, bias, y, workspace);
    }

    __aicore__ inline bool isActive(const TilingData& tiling) const
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t totalBlocks = tiling.batchDim * tiling.nDim * tiling.hoDim;
        return blockIdx < totalBlocks;
    }
};

template <typename FmapT, typename WeightT, typename OutputT, typename BiasT, typename TilingData>
struct Conv2dFullMNKernelPipeline {
    using BlockCtxT = BlockCtx<FmapT, WeightT, OutputT, BiasT>;

    __aicore__ inline void operator()(const TilingData& tiling, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                      GM_ADDR workspace) const
    {
        if ASCEND_IS_AIC_CONV {
            Conv2dFullMNKernel<BlockScheduler<BlockCtxT, TilingData>,
                               BatchProcessor<BlockCtxT, TilingData, FullMNKernelProcessor<BlockCtxT, TilingData>>,
                               BlockCtxT, TilingData>
                kernel;
            kernel(tiling, x, filter, bias, y, workspace);
        }
        if ASCEND_IS_AIV_CONV {
            Conv2dFunc::RunWeight5HDVec<DTYPE_FILTER>(tiling, filter, workspace);
        }
    }
};

} // namespace full_mn

#endif // FULL_MN_KERNEL_PIPELINE_H
