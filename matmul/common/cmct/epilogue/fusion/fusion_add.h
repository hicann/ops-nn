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
 * \file fusion_add.h
 * \brief
 */

#ifndef EPILOGUE_FUSION_EPILOGUE_FUSION_ADD_H
#define EPILOGUE_FUSION_EPILOGUE_FUSION_ADD_H
#include "kernel_operator.h"
#include "../../utils/common_utils.h"
#include "../../utils/device_utils.h"

namespace Cmct {
namespace Gemm {
namespace Block {
template <typename DataTypeOut_, typename DataTypeIn_>
class FusionAdd {
public:
    using DataTypeOut = DataTypeOut_;
    using DataTypeIn = DataTypeIn_;
    __aicore__ inline FusionAdd(){};

    struct Arguments {
        GM_ADDR inputGmAddr{nullptr};
    };

    struct Params {
        GM_ADDR inputGmAddr{nullptr};
    };

    static constexpr uint16_t ZERO_FLAG = 0;
    AscendC::LocalTensor<DataTypeIn> inputLocal_{AscendC::TPosition::VECIN, 0, AscendC::TOTAL_UB_SIZE};
    AscendC::GlobalTensor<DataTypeIn> inputGlobal_; // add的输入x3Gm
    int64_t stageSize_ = 0;

    int64_t ubCalcM_{0};
    int64_t ubCalcN_{0};
    int64_t strideN_{0};

    template <class LocalTensor>
    __aicore__ inline void Init(Params const& params, LocalTensor ubTensor, int64_t ubCalcM, int64_t ubCalcN,
                                int64_t& ubOffset, int64_t& stageSize)
    {
        static constexpr int64_t stageNum = 2;
        inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ DataTypeIn*>(params.inputGmAddr));// get x3 from gm
        int64_t lastUBSize = AscendC::TOTAL_UB_SIZE - ubOffset * sizeof(DataTypeIn);
        stageSize_ = AscendC::Std::min(
            static_cast<int64_t>(lastUBSize / stageNum / sizeof(DataTypeIn_) / ubCalcN * ubCalcN), ubCalcM * ubCalcN);
        inputLocal_ = ubTensor[ubOffset]; // get y from ub
        ubOffset += stageSize_;
        stageSize = stageSize_;
    }

    __aicore__ inline void operator()(const AscendC::LocalTensor<DataTypeIn>& srcLocal,
                                      AscendC::LocalTensor<DataTypeOut>& outputLocal, int64_t offset, int64_t curAivM,
                                      int64_t curAivN, int strideN, int64_t stageSize)
    {
        int64_t curAivNAlign = AlignBlock<DataTypeIn>(curAivN);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ZERO_FLAG);

        AscendC::DataCopyExtParams copyParams{static_cast<uint16_t>(stageSize / curAivNAlign),
                                     static_cast<uint32_t>(curAivN * sizeof(DataTypeOut)),
                                     static_cast<uint32_t>((strideN - curAivN) * sizeof(DataTypeIn)), 0, 0};
        AscendC::DataCopyPadExtParams<DataTypeIn> padParams{true, 0, 0, 0};
        // x3Gm -> x3local
        AscendC::DataCopyPad(inputLocal_, inputGlobal_[offset], copyParams, padParams);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ZERO_FLAG);

        // yLocal + x3Local -> outputLocal
        AscendC::Add(outputLocal, inputLocal_, srcLocal, stageSize);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR /* workspaceGm */)
    {
        return {args.inputGmAddr};
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Cmct
#endif // EPILOGUE_FUSION_EPILOGUE_FUSION_ADD_H
