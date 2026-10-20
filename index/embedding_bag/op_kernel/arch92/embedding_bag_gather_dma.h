/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_bag_gather_dma.h
 * \brief EmbeddingBagRegBaseIndx2d 的 GatherDMA 搬运实现（仅 arch92/Ascend960 系列）
 */

#ifndef EMBEDDING_BAG_GATHER_DMA_H_
#define EMBEDDING_BAG_GATHER_DMA_H_

#include "../arch35/embedding_bag_regbase_2d.h"

namespace EmbeddingBag {

// GatherParams/GatherPadParams/GatherDataCopy 为 arch92 专属 API，仅在 arch92 入口的编译单元内
// 提供本定义；arch35 入口不实例化 GATTER_DMA == 1 路径，无需此定义。
template <typename W, typename I, typename P, int32_t GATTER_DMA>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P, GATTER_DMA>::CopyInWeightByGatherDma(
    LocalTensor<I>& inidcesLocal, LocalTensor<W>& weightLocal, int64_t indiceOfset, int64_t loopIdx,
    int64_t weightRowNum, int64_t weightDimLen, int64_t weightDimOfset)
{
    LocalTensor<int64_t> gatherIndexLocal_ = gatherIndexBuf_.template Get<int64_t>();
    int64_t indiceLocalIdx = indiceOfset + loopIdx * weightRowMainLen_;
    CopyUnAlignWithCast<I, int64_t>(gatherIndexLocal_, inidcesLocal, indiceLocalIdx, weightRowNum);
    Muls(gatherIndexLocal_, gatherIndexLocal_, tilingData_.embeddingDim, weightRowNum);
    Adds(gatherIndexLocal_, gatherIndexLocal_, weightDimOfset, weightRowNum);
    Muls(gatherIndexLocal_, gatherIndexLocal_, sizeof(W), weightRowNum);

    // gatherIndexLocal_ 为buffer需要手动加同步
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
    GatherParams gatherParams{
        static_cast<uint16_t>(weightRowNum),
        static_cast<uint32_t>(weightDimLen * sizeof(W)),
        static_cast<uint32_t>(tilingData_.weightDimFactor * sizeof(W)),
    };
    GatherPadParams<W> gatherPadParams{
        GatherPaddingMode::CONSTANT_PADDING_MODE,
        static_cast<W>(0),
    };
    GatherDataCopy(weightLocal, weightGm_, gatherIndexLocal_, gatherParams, gatherPadParams);
}

} // namespace EmbeddingBag

#endif // EMBEDDING_BAG_GATHER_DMA_H_
