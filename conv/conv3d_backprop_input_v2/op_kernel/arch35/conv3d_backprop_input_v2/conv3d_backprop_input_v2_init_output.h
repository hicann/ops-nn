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
 * \file conv3d_backprop_input_v2_init_output.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ADVANCE_H
#define CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ADVANCE_H

#include "conv3d_backprop_input_v2_tiling_data.h"

namespace AscendC {
#if __CCE_AICORE__ == 310
    #if defined(__DAV_C310__) || defined(__DAV_310R6__)
        constexpr int32_t TOTALL0CSIZE = 262144;
    #endif
#endif

enum class InitOutputFlag {
    NO_INIT = 0,
    L0_INIT = 1,
    L1_INIT = 2,
};

template <typename yType>
class Conv3dDxInitOutput {
public:
    __aicore__ inline Conv3dDxInitOutput() {}
    __aicore__ inline void Init(GM_ADDR y, const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        InitTilingData(tilingData);
        // init global buffer
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        pipe_.InitBuffer(localBuffer_, TOTALL0CSIZE);
    }

    /** main logical function
     */

    __aicore__ inline uint64_t Ceil(uint64_t a, uint32_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void ProcessWithL0()
    {
        // david上数据量为128B对齐搬运效率最高，因此需对齐到128B，由于搬运的是fp32，因此搬运数量为32的倍数
        uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_, GetBlockNum()), ONE_BLK_SIZE);
        if (GetBlockIdx() >= Ceil(outputSize_, clearSizePerCore)) {
            SyncAllCores();
            return;
        }

        uint64_t offset = clearSizePerCore * GetBlockIdx();
        uint64_t realClearSize = outputSize_ - offset;
        realClearSize = realClearSize > clearSizePerCore ? clearSizePerCore : realClearSize;

        // popbuffer的dtype应为src的dtype float类型
        LocalTensor<float> popBuffer = localBuffer_.template Get<float>();
        // fixpipe的mSize的范围为【1，65535】，loc的size为262144，fp32的数据个数为65536，大于mSize，因此需减去32去对齐
        uint32_t localSize = (static_cast<uint64_t>(popBuffer.GetSize()) - ONE_BLK_SIZE) < clearSizePerCore
                                 ? (popBuffer.GetSize() - ONE_BLK_SIZE)
                                 : static_cast<uint32_t>(clearSizePerCore);

        uint64_t round = realClearSize / localSize;
        uint32_t tail = realClearSize % localSize;
        QuantMode_t quantPre = QuantMode_t::QF322BF16_PRE;
        if constexpr (IsSameType<yType, float>::value) {
            quantPre = QuantMode_t::QF322F32_PRE;
        } else if constexpr (IsSameType<yType, half>::value) {
            quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (IsSameType<yType, hifloat8_t>::value) {
            quantPre = QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        } else if constexpr (IsSameType<yType, fp8_e4m3fn_t>::value) {
            quantPre = QuantMode_t::QF322FP8_PRE;
        }
        // fixPipeParams前四个参数为nsize msize srcstride dststride，由于nsize为16，srcstride dststride可以不配置
        AscendC::FixpipeParamsC310<CO2Layout::NZ> fixPipeParamsDma(BLOCK_CUBE, localSize / BLOCK_CUBE, 1, 1);
        fixPipeParamsDma.deqScalar = 0;
        fixPipeParamsDma.quantPre = quantPre;
        for (uint64_t idx = 0; idx < round; ++idx) {
            // 由于我们是通过量化参数为0去达到清零效果，因此，我们的src起始地址可以始终为0
            // 若要配置src的起始地址不能使用popBuffer[offset]，因为offset可能会大于65536，导致越界
            AscendC::Fixpipe<yType, float, CFG_NZ>(yGm_[offset], popBuffer, fixPipeParamsDma);
            offset += localSize;
        }

        if (tail != 0) {
            if (tail / BLOCK_CUBE > 0) {
                // tail大于16，可以使用nz2nz去搬运
                AscendC::FixpipeParamsC310<CO2Layout::NZ> fixPipeParamsDmaTail(BLOCK_CUBE, tail / BLOCK_CUBE, 1, 1);
                fixPipeParamsDmaTail.deqScalar = 0;
                fixPipeParamsDmaTail.quantPre = quantPre;
                AscendC::Fixpipe<yType, float, CFG_NZ>(yGm_[offset], popBuffer, fixPipeParamsDmaTail);
            }
            // tail大于16和tail对16取余不等于0 可以同时存在，也可单独存在
            if (tail % BLOCK_CUBE != 0) {
                // tail对16取余不等于0，表明清零的size并不是16对齐的，需使用nz2nd去清零
                offset += tail / BLOCK_CUBE * BLOCK_CUBE;
                AscendC::FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixPipeParamsNz2nd;
                fixPipeParamsNz2nd.params.ndNum = 1;
                fixPipeParamsNz2nd.mSize = 1;
                fixPipeParamsNz2nd.nSize = tail % BLOCK_CUBE;
                // 由于msize为1，srcstride dststride可以不配置
                fixPipeParamsNz2nd.srcStride = BLOCK_CUBE;
                fixPipeParamsNz2nd.dstStride = tail % BLOCK_CUBE;
                fixPipeParamsNz2nd.deqScalar = 0;
                fixPipeParamsNz2nd.quantPre = quantPre;

                AscendC::Fixpipe<yType, float, CFG_ROW_MAJOR>(yGm_[offset], popBuffer, fixPipeParamsNz2nd);
            }
        }
        SyncAllCores();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            return;
        }

        ProcessWithL0();
    }

    __aicore__ inline void SyncAllCores()
    {
        CrossCoreSetFlag<0, PIPE_FIX>(SYNC_AIC_FLAG);
        CrossCoreWaitFlag(SYNC_AIC_FLAG);
    }

    __aicore__ inline void Destroy()
    {
        pipe_.Destroy();
    }

protected:
    uint64_t outputSize_;
    TPipe pipe_;
    TBuf<TPosition::CO1> localBuffer_;
    GlobalTensor<yType> yGm_;

    __aicore__ inline void InitTilingData(const conv_bp_v2_kernel::Conv3DBackpropInputV2TilingData *tilingData)
    {
        uint64_t mSize = static_cast<uint64_t>(tilingData->conv3DDxTiling.hi) * tilingData->conv3DDxTiling.wi;
        uint64_t nSize = static_cast<uint64_t>(tilingData->conv3DDxTiling.cin);
        outputSize_ = mSize * nSize * tilingData->conv3DDxTiling.di * tilingData->conv3DDxTiling.batch;
    }
};
}  // namespace AscendC

#endif  // CONV3D_BACKPROP_INPUT_V2_INIT_OUTPUT_ADVANCE_H