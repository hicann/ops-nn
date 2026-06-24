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
 * \file adam_apply_one_with_decay_assign.h
 * \brief
 * */
#ifndef ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_H
#define ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "adam_apply_one_with_decay_assign_tiling_key.h"
#include "adam_apply_one_with_decay_assign_tiling_data.h"
#include "kernel_operator.h"
namespace NsAdamApplyOneWithDecayAssign {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;
template <typename T>
class AdamApplyOneWithDecayAssign {
public:
    __aicore__ inline AdamApplyOneWithDecayAssign(){};

    __aicore__ inline void Init(
        GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3, GM_ADDR input4, GM_ADDR mul0_x, GM_ADDR mul1_x,
        GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR mul4_x, GM_ADDR add2_y, GM_ADDR output0, GM_ADDR output1,
        GM_ADDR output2, GM_ADDR workspace, const AdamApplyOneWithDecayAssignTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueInput0;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueInput1;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueInput2;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueInput3;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueInput4;

    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueMul0_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueMul1_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueMul2_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueMul3_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueMul4_x;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inputQueueAdd2_y;

    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueueOutput0;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueueOutput1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outputQueueOutput2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp0, tmp1, tmp2, tmp3;

    AscendC::GlobalTensor<T> inputGMInput0;
    AscendC::GlobalTensor<T> inputGMInput1;
    AscendC::GlobalTensor<T> inputGMInput2;
    AscendC::GlobalTensor<T> inputGMInput3;
    AscendC::GlobalTensor<T> inputGMInput4;

    AscendC::GlobalTensor<T> inputGMMul0_x;
    AscendC::GlobalTensor<T> inputGMMul1_x;
    AscendC::GlobalTensor<T> inputGMMul2_x;
    AscendC::GlobalTensor<T> inputGMMul3_x;
    AscendC::GlobalTensor<T> inputGMMul4_x;

    AscendC::GlobalTensor<T> inputGMAdd2_y;

    AscendC::GlobalTensor<T> outputGMOutput0;
    AscendC::GlobalTensor<T> outputGMOutput1;
    AscendC::GlobalTensor<T> outputGMOutput2;

    uint32_t coreDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t processDataNum = 0;
};

template <typename T>
__aicore__ inline void AdamApplyOneWithDecayAssign<T>::Init(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3, GM_ADDR input4, GM_ADDR mul0_x, GM_ADDR mul1_x,
    GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR mul4_x, GM_ADDR add2_y, GM_ADDR output0, GM_ADDR output1, GM_ADDR output2,
    GM_ADDR workspace, const AdamApplyOneWithDecayAssignTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    uint32_t coreNum = AscendC::GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }

    inputGMInput0.SetGlobalBuffer((__gm__ T*)input0 + globalBufferIndex, this->coreDataNum);
    inputGMInput1.SetGlobalBuffer((__gm__ T*)input1 + globalBufferIndex, this->coreDataNum);
    inputGMInput2.SetGlobalBuffer((__gm__ T*)input2 + globalBufferIndex, this->coreDataNum);
    inputGMInput3.SetGlobalBuffer((__gm__ T*)input3 + globalBufferIndex, this->coreDataNum);
    inputGMInput4.SetGlobalBuffer((__gm__ T*)input4 + globalBufferIndex, this->coreDataNum);
    inputGMMul0_x.SetGlobalBuffer((__gm__ T*)mul0_x + globalBufferIndex, this->coreDataNum);
    inputGMMul1_x.SetGlobalBuffer((__gm__ T*)mul1_x + globalBufferIndex, this->coreDataNum);
    inputGMMul2_x.SetGlobalBuffer((__gm__ T*)mul2_x + globalBufferIndex, this->coreDataNum);
    inputGMMul3_x.SetGlobalBuffer((__gm__ T*)mul3_x + globalBufferIndex, this->coreDataNum);
    inputGMMul4_x.SetGlobalBuffer((__gm__ T*)mul4_x + globalBufferIndex, this->coreDataNum);
    inputGMAdd2_y.SetGlobalBuffer((__gm__ T*)add2_y + globalBufferIndex, this->coreDataNum);
    outputGMOutput0.SetGlobalBuffer((__gm__ T*)output0 + globalBufferIndex, this->coreDataNum);
    outputGMOutput1.SetGlobalBuffer((__gm__ T*)output1 + globalBufferIndex, this->coreDataNum);
    outputGMOutput2.SetGlobalBuffer((__gm__ T*)output2 + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inputQueueInput0, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueInput1, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueInput2, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueInput3, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueInput4, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(inputQueueMul0_x, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueMul1_x, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueMul2_x, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueMul3_x, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inputQueueMul4_x, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(inputQueueAdd2_y, BUFFER_NUM, this->tileDataNum * sizeof(T));

    pipe.InitBuffer(outputQueueOutput0, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueOutput1, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outputQueueOutput2, BUFFER_NUM, this->tileDataNum * sizeof(T));

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
        pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
    } else if constexpr (std::is_same_v<T, half>) {
        pipe.InitBuffer(tmp0, this->tileDataNum * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void AdamApplyOneWithDecayAssign<T>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> input0Local = inputQueueInput0.AllocTensor<T>();
    AscendC::LocalTensor<T> input1Local = inputQueueInput1.AllocTensor<T>();
    AscendC::LocalTensor<T> input2Local = inputQueueInput2.AllocTensor<T>();
    AscendC::LocalTensor<T> input3Local = inputQueueInput3.AllocTensor<T>();
    AscendC::LocalTensor<T> input4Local = inputQueueInput4.AllocTensor<T>();

    AscendC::LocalTensor<T> mul0_xLocal = inputQueueMul0_x.AllocTensor<T>();
    AscendC::LocalTensor<T> mul1_xLocal = inputQueueMul1_x.AllocTensor<T>();
    AscendC::LocalTensor<T> mul2_xLocal = inputQueueMul2_x.AllocTensor<T>();
    AscendC::LocalTensor<T> mul3_xLocal = inputQueueMul3_x.AllocTensor<T>();
    AscendC::LocalTensor<T> mul4_xLocal = inputQueueMul4_x.AllocTensor<T>();
    AscendC::LocalTensor<T> add2_yLocal = inputQueueAdd2_y.AllocTensor<T>();

    AscendC::DataCopy(input0Local, inputGMInput0[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(input1Local, inputGMInput1[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(input2Local, inputGMInput2[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(input3Local, inputGMInput3[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(input4Local, inputGMInput4[progress * this->tileDataNum], this->processDataNum);

    AscendC::DataCopy(mul0_xLocal, inputGMMul0_x[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(mul1_xLocal, inputGMMul1_x[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(mul2_xLocal, inputGMMul2_x[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(mul3_xLocal, inputGMMul3_x[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(mul4_xLocal, inputGMMul4_x[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(add2_yLocal, inputGMAdd2_y[progress * this->tileDataNum], this->processDataNum);

    inputQueueInput0.EnQue(input0Local);
    inputQueueInput1.EnQue(input1Local);
    inputQueueInput2.EnQue(input2Local);
    inputQueueInput3.EnQue(input3Local);
    inputQueueInput4.EnQue(input4Local);
    inputQueueMul0_x.EnQue(mul0_xLocal);
    inputQueueMul1_x.EnQue(mul1_xLocal);
    inputQueueMul2_x.EnQue(mul2_xLocal);
    inputQueueMul3_x.EnQue(mul3_xLocal);
    inputQueueMul4_x.EnQue(mul4_xLocal);
    inputQueueAdd2_y.EnQue(add2_yLocal);
}

template <typename T>
__aicore__ inline void AdamApplyOneWithDecayAssign<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> output0Local = outputQueueOutput0.DeQue<T>();
    AscendC::LocalTensor<T> output1Local = outputQueueOutput1.DeQue<T>();
    AscendC::LocalTensor<T> output2Local = outputQueueOutput2.DeQue<T>();

    AscendC::DataCopy(outputGMOutput0[progress * this->tileDataNum], output0Local, this->processDataNum);
    AscendC::DataCopy(outputGMOutput1[progress * this->tileDataNum], output1Local, this->processDataNum);
    AscendC::DataCopy(outputGMOutput2[progress * this->tileDataNum], output2Local, this->processDataNum);

    outputQueueOutput0.FreeTensor(output0Local);
    outputQueueOutput1.FreeTensor(output1Local);
    outputQueueOutput2.FreeTensor(output2Local);
}

template <typename T>
__aicore__ inline void AdamApplyOneWithDecayAssign<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> input0Local = inputQueueInput0.DeQue<T>();
    AscendC::LocalTensor<T> input1Local = inputQueueInput1.DeQue<T>();
    AscendC::LocalTensor<T> input2Local = inputQueueInput2.DeQue<T>();
    AscendC::LocalTensor<T> input3Local = inputQueueInput3.DeQue<T>();
    AscendC::LocalTensor<T> input4Local = inputQueueInput4.DeQue<T>();

    AscendC::LocalTensor<T> mul0_xLocal = inputQueueMul0_x.DeQue<T>();
    AscendC::LocalTensor<T> mul1_xLocal = inputQueueMul1_x.DeQue<T>();
    AscendC::LocalTensor<T> mul2_xLocal = inputQueueMul2_x.DeQue<T>();
    AscendC::LocalTensor<T> mul3_xLocal = inputQueueMul3_x.DeQue<T>();
    AscendC::LocalTensor<T> mul4_xLocal = inputQueueMul4_x.DeQue<T>();
    AscendC::LocalTensor<T> add2_yLocal = inputQueueAdd2_y.DeQue<T>();

    AscendC::LocalTensor<T> output0Local = outputQueueOutput0.AllocTensor<T>();
    AscendC::LocalTensor<T> output1Local = outputQueueOutput1.AllocTensor<T>();
    AscendC::LocalTensor<T> output2Local = outputQueueOutput2.AllocTensor<T>();

    int len = this->processDataNum;

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        AscendC::LocalTensor<float> tLocal0 = tmp0.Get<float>();
        AscendC::LocalTensor<float> tLocal1 = tmp1.Get<float>();
        AscendC::LocalTensor<float> tLocal2 = tmp2.Get<float>();
        AscendC::LocalTensor<float> tLocal3 = tmp3.Get<float>();

        // output1 = input2 * mul0_x + input0 * mul1_x
        AscendC::Cast(tLocal0, input2Local, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Cast(tLocal1, mul0_xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tLocal1, tLocal0, tLocal1, len);
        AscendC::Cast(tLocal2, input0Local, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Cast(tLocal3, mul1_xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tLocal3, tLocal2, tLocal3, len);
        AscendC::Add(tLocal3, tLocal1, tLocal3, len);
        AscendC::Cast(output1Local, tLocal3, AscendC::RoundMode::CAST_RINT, len);

        // output0 = input0 * input0 * mul3_x + input1 * mul2_x
        AscendC::Mul(tLocal0, tLocal2, tLocal2, len);
        AscendC::Cast(tLocal1, mul3_xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tLocal1, tLocal0, tLocal1, len);
        AscendC::Cast(tLocal0, input1Local, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Cast(tLocal2, mul2_xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tLocal0, tLocal2, tLocal0, len);
        AscendC::Add(tLocal0, tLocal0, tLocal1, len);
        AscendC::Cast(output0Local, tLocal0, AscendC::RoundMode::CAST_RINT, len);

        // output2 = input3 - (output1 / (sqrt(output0) + add2_y) + input3 * mul4_x) * input4
        AscendC::Sqrt(tLocal0, tLocal0, len);
        AscendC::Cast(tLocal1, add2_yLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Add(tLocal1, tLocal0, tLocal1, len);
        AscendC::Div(tLocal1, tLocal3, tLocal1, len);
        AscendC::Cast(tLocal0, input3Local, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Cast(tLocal2, mul4_xLocal, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tLocal2, tLocal0, tLocal2, len);
        AscendC::Add(tLocal1, tLocal1, tLocal2, len);
        AscendC::Cast(tLocal2, input4Local, AscendC::RoundMode::CAST_NONE, len);
        AscendC::Mul(tLocal1, tLocal1, tLocal2, len);
        AscendC::Sub(tLocal2, tLocal0, tLocal1, len);
        AscendC::Cast(output2Local, tLocal2, AscendC::RoundMode::CAST_RINT, len);
    } else {
        AscendC::Mul(output0Local, input0Local, input0Local, len);
        AscendC::Mul(output0Local, mul3_xLocal, output0Local, len);
        AscendC::Mul(input1Local, input1Local, mul2_xLocal, len); // tmp2
        AscendC::Add(output0Local, output0Local, input1Local, len);

        AscendC::Mul(input1Local, input0Local, mul1_xLocal, len); // tmp4
        AscendC::Mul(input0Local, input2Local, mul0_xLocal, len); // tmp3
        AscendC::Add(output1Local, input0Local, input1Local, len);

        if constexpr (std::is_same_v<T, half>) {
            AscendC::LocalTensor<float> tLocal0 = tmp0.Get<float>();
            AscendC::Cast(tLocal0, output0Local, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Sqrt(tLocal0, tLocal0, len); // tmp5
            AscendC::Cast(input0Local, tLocal0, AscendC::RoundMode::CAST_RINT, len);
        } else {
            AscendC::Sqrt(input0Local, output0Local, len); // tmp5
        }

        AscendC::Add(input0Local, input0Local, add2_yLocal, len);  // tmp6
        AscendC::Div(input1Local, output1Local, input0Local, len); // tmp7
        AscendC::Mul(input0Local, input3Local, mul4_xLocal, len);  // tmp8
        AscendC::Add(input1Local, input1Local, input0Local, len);  // tmp9
        AscendC::Mul(input1Local, input1Local, input4Local, len);  // tmp10
        AscendC::Sub(output2Local, input3Local, input1Local, len);
    }

    inputQueueInput0.FreeTensor(input0Local);
    inputQueueInput1.FreeTensor(input1Local);
    inputQueueInput2.FreeTensor(input2Local);
    inputQueueInput3.FreeTensor(input3Local);
    inputQueueInput4.FreeTensor(input4Local);

    inputQueueMul0_x.FreeTensor(mul0_xLocal);
    inputQueueMul1_x.FreeTensor(mul1_xLocal);
    inputQueueMul2_x.FreeTensor(mul2_xLocal);
    inputQueueMul3_x.FreeTensor(mul3_xLocal);
    inputQueueMul4_x.FreeTensor(mul4_xLocal);
    inputQueueAdd2_y.FreeTensor(add2_yLocal);

    outputQueueOutput0.EnQue(output0Local);
    outputQueueOutput1.EnQue(output1Local);
    outputQueueOutput2.EnQue(output2Local);
}

template <typename T>
__aicore__ inline void AdamApplyOneWithDecayAssign<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}
} // namespace NsAdamApplyOneWithDecayAssign
#endif // ADAM_APPLY_ONE_WITH_DECAY_ASSIGN_H
