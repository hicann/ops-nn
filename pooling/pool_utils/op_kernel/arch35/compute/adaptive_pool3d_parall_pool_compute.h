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
 * \file adaptive_pool3d_parall_pool_compute.h
 * \brief AdaptiveAvgPool3D/AdaptiveMaxPool3D parallel pool 模板共用的核内切分参数计算接口。
 */

#ifndef POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_POOL3D_PARALL_POOL_COMPUTE_H_
#define POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_POOL3D_PARALL_POOL_COMPUTE_H_

#include <cstdint>

#include "op_kernel/math_util.h"
#include "kernel_operator.h"

namespace PoolUtils {
namespace Compute {

/*
 * 功能：按当前 block 序号推导 nc/do/ho/wo 的切分索引与数量，并换算出输入窗口起点、长度与 GM 偏移。
 * 说明：TilingT 为 AdaptivePool3dParaKernelTilingData，BlockSplitParamT 为算子侧的 BlockSplitParam。
 */
template <typename TilingT, typename BlockSplitParamT>
__aicore__ inline void CalInputBlockPara(int64_t curBlockIdx, BlockSplitParamT& blockPara, const TilingT& tilingData,
                                         int64_t inDHW, int64_t inHW)
{
    int64_t dhwOuter = tilingData.doOuter * tilingData.hoOuter * tilingData.woOuter;
    int64_t hwOuter = tilingData.hoOuter * tilingData.woOuter;

    blockPara.ncIdx = curBlockIdx / dhwOuter;
    blockPara.ncNum = (blockPara.ncIdx == (tilingData.ncOuter - 1)) ? tilingData.ncTail : tilingData.ncFactor;
    /* ncdhw */
    int64_t blockIdxOnNc = curBlockIdx % dhwOuter;
    blockPara.doIdx = blockIdxOnNc / hwOuter;
    blockPara.doNum = (blockPara.doIdx == (tilingData.doOuter - 1)) ? tilingData.doTail : tilingData.doFactor;

    int64_t blockIdxOnD = blockIdxOnNc % hwOuter;
    blockPara.hoIdx = blockIdxOnD / tilingData.woOuter;
    blockPara.hoNum = (blockPara.hoIdx == (tilingData.hoOuter - 1)) ? tilingData.hoTail : tilingData.hoFactor;

    int64_t blockIdxOnDH = blockIdxOnD % tilingData.woOuter;
    blockPara.woIdx = blockIdxOnDH % tilingData.woOuter;
    blockPara.woNum = (blockPara.woIdx == (tilingData.woOuter - 1)) ? tilingData.woTail : tilingData.woFactor;

    blockPara.kerDStartIdx = ((blockPara.doIdx * tilingData.doFactor) * tilingData.dIn) / tilingData.dOut;
    blockPara.kerHStartIdx = ((blockPara.hoIdx * tilingData.hoFactor) * tilingData.hIn) / tilingData.hOut;
    blockPara.kerWStartIdx = ((blockPara.woIdx * tilingData.woFactor) * tilingData.wIn) / tilingData.wOut;
    int32_t kerDEndIdx = Ops::Base::CeilDiv((blockPara.doIdx * tilingData.doFactor + blockPara.doNum) * tilingData.dIn,
                                            tilingData.dOut);
    int32_t kerHEndIdx = Ops::Base::CeilDiv((blockPara.hoIdx * tilingData.hoFactor + blockPara.hoNum) * tilingData.hIn,
                                            tilingData.hOut);
    int32_t kerWEndIdx = Ops::Base::CeilDiv((blockPara.woIdx * tilingData.woFactor + blockPara.woNum) * tilingData.wIn,
                                            tilingData.wOut);

    blockPara.diDataLen = kerDEndIdx - blockPara.kerDStartIdx;
    blockPara.hiDataLen = kerHEndIdx - blockPara.kerHStartIdx;
    blockPara.wiDataLen = kerWEndIdx - blockPara.kerWStartIdx;
    blockPara.xOffset = blockPara.ncIdx * tilingData.ncFactor * inDHW + blockPara.kerDStartIdx * inHW +
                        blockPara.kerHStartIdx * tilingData.wIn + blockPara.kerWStartIdx;
}

} // namespace Compute
} // namespace PoolUtils

#endif // POOL_UTILS_ARCH35_COMPUTE_ADAPTIVE_POOL3D_PARALL_POOL_COMPUTE_H_
