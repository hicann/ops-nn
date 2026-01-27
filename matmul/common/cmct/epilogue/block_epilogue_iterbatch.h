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
 * \file block_epilogue_iterbatch.h
 * \brief
 */

#ifndef CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_ITERBATCH_H
#define CMCT_INCLUDE_EPILOGUE_BLOCK_EPILOGUE_ITERBATCH_H
#include "kernel_basic_intf.h"
#include "fusion/default_fusion_op.h"
#include "../utils/common_utils.h"
#include "../utils/device_utils.h"
#include "../utils/status_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {

template <typename DataTypeOut_>
class BlockEpilogueIterbatch {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;
    using DataTypeOut = DataTypeOut_;

    AscendC::LocalTensor<DataTypeOut> ubLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<DataTypeOut> outputGlobal_;
    uint64_t m_{0};
    uint64_t n_{0};
    uint64_t l0CEventID_{0};

    struct Arguments {
        Arguments() = default;
    };

    struct Params {
        GM_ADDR cGmAddr{nullptr};
    };

    __aicore__ inline BlockEpilogueIterbatch()
    {}

    __aicore__ inline void Run(
        int64_t offsetC, uint64_t baseM, uint64_t baseN, uint64_t curIterBatchL1, uint64_t mainIterBatchL0,
        bool isLastTurn)
    {
        AscendC::GlobalTensor<DataTypeOut> tmpGlobal = outputGlobal_[offsetC];

        uint64_t mL0Cnt = CeilDiv(m_, baseM);
        uint64_t nL0Cnt = CeilDiv(n_, baseN);

        // calculate how much loop needed between l1 and l0.
        uint64_t stepIterBatchL1L0 = CeilDiv(curIterBatchL1, mainIterBatchL0);
        uint64_t loopMax = stepIterBatchL1L0 * nL0Cnt * mL0Cnt;
        uint64_t loopTimes = 0;
        for (uint64_t iter1 = 0; iter1 < stepIterBatchL1L0; ++iter1) {
            uint64_t curIterBatchL0 = (iter1 + 1 == stepIterBatchL1L0) ? // if tailloop of l1 and l0, cal tail iter num.
                                          (curIterBatchL1 - mainIterBatchL0 * iter1) :
                                          mainIterBatchL0;
            for (uint64_t iterNL0 = 0; iterNL0 < nL0Cnt; ++iterNL0) {
                uint64_t curNL0 = (iterNL0 == nL0Cnt - 1) ? (n_ - (nL0Cnt - 1) * baseN) : baseN;
                for (uint64_t iterML0 = 0; iterML0 < mL0Cnt; ++iterML0) {
                    if ((l0CEventID_ & 0x1) == AscendC::GetSubBlockIdx()) {
                        uint64_t curML0 = (iterML0 == mL0Cnt - 1) ? (m_ - (mL0Cnt - 1) * baseM) : baseM;
                        uint64_t offsetCGMOfCopyOut =
                            iter1 * mainIterBatchL0 * m_ * n_ + iterML0 * baseM * n_ + iterNL0 * baseN;
                        uint16_t blockCount = static_cast<uint16_t>(curIterBatchL0 * curML0);
                        uint32_t blockLen = static_cast<uint32_t>(curNL0 * sizeof(DataTypeOut));
                        uint32_t srcStride = static_cast<uint32_t>(
                            (CeilAlign(curNL0, AscendC::BLOCK_CUBE) - curNL0) * sizeof(DataTypeOut) / UB_ALIGN_SIZE);
                        uint32_t dstStride = static_cast<uint32_t>((n_ - curNL0) * sizeof(DataTypeOut));
                        AscendC::DataCopyExtParams copyParams{blockCount, blockLen, srcStride, dstStride, 0};
                        AscendC::CrossCoreWaitFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(0x0);
                        AscendC::DataCopyPad(tmpGlobal[offsetCGMOfCopyOut], ubLocal_, copyParams);
                        if (!isLastTurn || (loopTimes + SYNC_OFFSET < loopMax)) {
                            AscendC::CrossCoreSetFlag<AIC_SYNC_AIV_MODE_4, PIPE_MTE3>(SYNC_OFFSET);
                        }
                    }
                    l0CEventID_++;
                    loopTimes++;
                }
            }
        }
    }

    __aicore__ inline void Init(Params const& params, const BlockShape& shape)
    {
        outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeOut*>(params.cGmAddr));
        m_ = Get<DIMENSION_M>(shape);
        n_ = Get<DIMENSION_N>(shape);
        l0CEventID_ = 0;
    }

    __aicore__ inline auto GetTensor()
    {
        return ubLocal_;
    }

    __aicore__ inline void operator()(
        int64_t offsetC, uint64_t baseM, uint64_t baseN, uint64_t curIterBatchL1, uint64_t mainIterBatchL0,
        bool isLastTurn)
    {
        Run(offsetC, baseM, baseN, curIterBatchL1, mainIterBatchL0, isLastTurn);
    }

private:
    constexpr static uint16_t DIMENSION_M = 0;
    constexpr static uint16_t DIMENSION_N = 1;
    constexpr static uint16_t SYNC_OFFSET = 2;
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif
