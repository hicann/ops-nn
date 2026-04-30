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
 * \file log_softmax_v2.h
 * \brief
*/
#ifndef LOGSOFTMAX_H
#define LOGSOFTMAX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "log_softmax_v2_tiling_data.h"
#include "log_softmax_v2_tiling_key.h"

using namespace AscendC;

namespace LogSoftmaxConst {
    static const int ZERO = 0;
    static const int ONE = 1;
    static const int TWO = 2;
    static const int FOUR = 4;
    static const int SEVEN = 7;
    static const int EIGHT = 8;
    static const int SIXTEEN = 16;
    static const int THIRTY_TWO = 32;
    static const int SIXTY_FOUR = 64;
    static const int ONE_TWENTY_EIGHT = 128;
    static const int ONE_NINETY_TWO = 192;
    static const int TWO_FIFTY_SIX = 256;
    static const int TWO_THOUSAND_FORTY_EIGHT = 2048;
    static const int FOUR_THOUSAND_NINETY_SIX = 4096;
    static const int EIGHT_THOUSAND_ONE_HUNDRED_NINETY_TWO = 8192;
    static const int SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR = 16384;
    static const int THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT = 32768;

    static const int FLOAT_SIZE = 4;
    static const int HALF_SIZE = 2;
    static const int BF16_SIZE = 2;

    static const int FLOAT_ALIGN_BLOCK = 8;
    static const int HALF_ALIGN_BLOCK = 16;

    static const int LENGTH_ARRAY_SIZE = 4;

    static const float FLOAT_NEG_INF = -3.4028235e38f;
    static const half HALF_NEG_INF = half(-65504);
}

__aicore__ inline int align_to(int a, int b)
{
    if (b == 0) {
        return 1;
    }
    return (a + b - 1) / b * b;
}

__aicore__ inline int mmin(int a, int b)
{
    if (a <= b) {
        return a;
    }
    return b;
}

class LogSoftmax {
public:
    __aicore__ inline LogSoftmax() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int m, int n)
    {
        this->inputGm.SetGlobalBuffer((__gm__ float *)input);
        this->outGm.SetGlobalBuffer((__gm__ float *)out);
        this->m = m;
        this->n = n;

        int len = n;
        int j = LogSoftmaxConst::ZERO;
        while (len > LogSoftmaxConst::SIXTY_FOUR) {
            this->length[j] = len;
            len = (len + LogSoftmaxConst::SEVEN) / LogSoftmaxConst::EIGHT;
            j++;
        }
        this->iters = j;
        this->length[j] = len;

        pipe.InitBuffer(inQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
        pipe.InitBuffer(outQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
        pipe.InitBuffer(workQueue1, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
        pipe.InitBuffer(workQueue2, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
    }


public:
#if defined(HIGH_PERFORMANCE) && HIGH_PERFORMANCE == 1
    __aicore__ inline void Process1(int start, int end)
    {
        LocalTensor<half> workLocal1 = workQueue1.Get<half>();
        LocalTensor<float> workLocal2 = workQueue2.Get<float>();
        LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        int align_n;
        if (n <= LogSoftmaxConst::ONE_TWENTY_EIGHT) {
            align_n = LogSoftmaxConst::ONE_TWENTY_EIGHT;
        } else if (n <= LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT) {
            align_n = LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT;
        } else {
            align_n = LogSoftmaxConst::EIGHT_THOUSAND_ONE_HUNDRED_NINETY_TWO;
        }
        int step = LogSoftmaxConst::EIGHT_THOUSAND_ONE_HUNDRED_NINETY_TWO / align_n;
        int loop = (end - start + step - LogSoftmaxConst::ONE) / step;
        int tail = (end - start) % step;
        tail = tail == LogSoftmaxConst::ZERO ? step : tail;
        DataCopyExtParams copyParams{uint16_t(tail), uint32_t(n * LogSoftmaxConst::FLOAT_SIZE), 0,
            uint32_t((align_n - n) / LogSoftmaxConst::FLOAT_ALIGN_BLOCK), 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[start * n], copyParams, padParams);
        inQueue.EnQue(inputLocal);
        DataCopyExtParams copyParamsOut{uint16_t(tail), static_cast<uint32_t>(n * LogSoftmaxConst::FLOAT_SIZE),
            uint32_t((align_n - n) / LogSoftmaxConst::FLOAT_ALIGN_BLOCK), 0, 0};
        int offset = start * n;
        for (int i = 0; i != loop; i++) {
            inputLocal = inQueue.DeQue<float>();
            if (i != loop - LogSoftmaxConst::ONE) {
                LocalTensor<float> aLocal = inQueue.AllocTensor<float>();
                copyParams.blockCount = step;
                DataCopyPad(aLocal, inputGm[(i * step + start + tail) * n], copyParams, padParams);
                inQueue.EnQue(aLocal);
            }
            int lines = i == LogSoftmaxConst::ZERO ? tail : step;
            Cast(workLocal1, inputLocal, RoundMode::CAST_RINT, lines * align_n);
            LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

            if (align_n == LogSoftmaxConst::ONE_TWENTY_EIGHT) {
                for (int j = 0; j != lines; j++) {
                    ReduceMax<half>(workLocal1[j * LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        workLocal1[j * LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        workLocal1[j * LogSoftmaxConst::ONE_TWENTY_EIGHT], n, true);
                    half maxIndex = workLocal1.GetValue(LogSoftmaxConst::ONE + j * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    uint16_t realIndex = *reinterpret_cast<uint16_t *>(&maxIndex);
                    Adds(outLocal[j * LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        inputLocal[j * LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        -float(inputLocal(j * LogSoftmaxConst::ONE_TWENTY_EIGHT + realIndex)), n);
                }
            } else if (align_n == LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT) {
                for (int j = 0; j != lines; j++) {
                    ReduceMax<half>(workLocal1[j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT],
                        workLocal1[j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT],
                        workLocal1[j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT], n, true);
                    half maxIndex = workLocal1.GetValue(LogSoftmaxConst::ONE + j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT);
                    uint16_t realIndex = *reinterpret_cast<uint16_t *>(&maxIndex);
                    Adds(outLocal[j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT],
                        inputLocal[j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT],
                        -float(inputLocal(j * LogSoftmaxConst::TWO_THOUSAND_FORTY_EIGHT + realIndex)), n);
                }
            } else {
                ReduceMax<half>(workLocal1, workLocal1, workLocal1, n, true);
                half maxIndex = workLocal1.GetValue(LogSoftmaxConst::ONE);
                uint16_t realIndex = *reinterpret_cast<uint16_t *>(&maxIndex);
                Adds(outLocal, inputLocal, -float(inputLocal(realIndex)), n);
            }

            if (align_n == LogSoftmaxConst::EIGHT_THOUSAND_ONE_HUNDRED_NINETY_TWO) {
                Exp(inputLocal, outLocal, n);
            } else {
                Exp(inputLocal, outLocal, align_n * lines);
            }

            SetMaskCount();
            for (int k = 0; k != lines; k++) {
                for (int j = 0; j != iters; j++) {
                    SetVectorMask<float>(0, length[j]);
                    BlockReduceSum<float, false>(inputLocal[align_n * k], inputLocal[align_n * k], 1, MASK_PLACEHOLDER,
                        1, 1, LogSoftmaxConst::EIGHT);
                    PipeBarrier<PIPE_V>();
                }
                SetVectorMask<float>(0, length[iters]);
                WholeReduceSum<float, false>(inputLocal[align_n * k], inputLocal[align_n * k], 1, MASK_PLACEHOLDER,
                    1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();

            for (int j = 0; j != lines; j++) {
                Ln(inputLocal[align_n * j], inputLocal[align_n * j], LogSoftmaxConst::EIGHT);
                Adds(outLocal[align_n * j], outLocal[align_n * j], -inputLocal(align_n * j), n);
            }
            inQueue.FreeTensor(inputLocal);
            outQueue.EnQue(outLocal);
            outQueue.DeQue<float>();
            copyParamsOut.blockCount = lines;
            DataCopyPad(outGm[offset], outLocal, copyParamsOut);
            offset += lines * n;
            outQueue.FreeTensor(outLocal);
        }
    }
#else
    __aicore__ inline void Process1(int start, int end)
    {
        LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        DataCopy(inputLocal, inputGm[start * n], align_to(n, LogSoftmaxConst::THIRTY_TWO));
        inQueue.EnQue(inputLocal);

        for (int i = start; i != end; i++) {
            inputLocal = inQueue.DeQue<float>();
            if (i != end - LogSoftmaxConst::ONE) {
                LocalTensor<float> aLocal = inQueue.AllocTensor<float>();
                DataCopy(aLocal, inputGm[(i + LogSoftmaxConst::ONE) * n], align_to(n, LogSoftmaxConst::THIRTY_TWO));
                inQueue.EnQue(aLocal);
            }

            LocalTensor<float> srcLocal = inputLocal;
            LocalTensor<float> outLocal = outQueue.AllocTensor<float>();

            SetMaskCount();
            for (int j = 0; j != iters; j++) {
                SetVectorMask<float>(0, length[j]);
                BlockReduceMax<float, false>(outLocal, srcLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
                srcLocal = outLocal;
            }
            SetVectorMask<float>(0, length[iters]);
            WholeReduceMax<float, false>(outLocal, srcLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();

            Adds(outLocal, inputLocal, -outLocal(0), n);
            Exp(inputLocal, outLocal, n);

            SetMaskCount();
            for (int j = 0; j != iters; j++) {
                SetVectorMask<float>(0, length[j]);
                BlockReduceSum<float, false>(inputLocal, inputLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
            }
            SetVectorMask<float>(0, length[iters]);
            WholeReduceSum<float, false>(inputLocal, inputLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();

            Ln(inputLocal, inputLocal, LogSoftmaxConst::EIGHT);

            Adds(outLocal, outLocal, -inputLocal(0), n);

            outQueue.EnQue(outLocal);
            outQueue.DeQue<float>();

            if (n % LogSoftmaxConst::EIGHT) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(n * LogSoftmaxConst::FLOAT_SIZE), 0, 0, 0};
                DataCopyPad(outGm[i * n], outLocal, copyParams);
            } else {
                DataCopy(outGm[i * n], outLocal, n);
            }
            outQueue.FreeTensor(outLocal);

            inQueue.FreeTensor(inputLocal);
        }
    }
#endif

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 4096;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> workQueue1;
    TBuf<TPosition::VECCALC> workQueue2;

    GlobalTensor<float> inputGm;
    GlobalTensor<float> outGm;

    int m, n, iters;
    int length[LogSoftmaxConst::LENGTH_ARRAY_SIZE];
};

class LogSoftmaxCol {
public:
    __aicore__ inline LogSoftmaxCol() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int m, int n, int flag)
    {
        this->inputGm.SetGlobalBuffer((__gm__ float *)input);
        this->outGm.SetGlobalBuffer((__gm__ float *)out);
        this->m = m;
        this->n = n;
        if (flag) {
            coliters = (m + LogSoftmaxConst::ONE_TWENTY_EIGHT - 1) / LogSoftmaxConst::ONE_TWENTY_EIGHT;
            coltail = m % LogSoftmaxConst::ONE_TWENTY_EIGHT;
            if (!coltail) {
                coltail = LogSoftmaxConst::ONE_TWENTY_EIGHT;
            }
            rowiters = (n + LogSoftmaxConst::SIXTEEN - 1) / LogSoftmaxConst::SIXTEEN;
            rowtail = n % LogSoftmaxConst::SIXTEEN;
            if (!rowtail) {
                rowtail = LogSoftmaxConst::SIXTEEN;
            }
        } else {
            coliters = (m + LogSoftmaxConst::SIXTY_FOUR - 1) / LogSoftmaxConst::SIXTY_FOUR;
            coltail = m % LogSoftmaxConst::SIXTY_FOUR;
            if (!coltail) {
                coltail = LogSoftmaxConst::SIXTY_FOUR;
            }
            rowiters = (n + LogSoftmaxConst::ONE_TWENTY_EIGHT - 1) / LogSoftmaxConst::ONE_TWENTY_EIGHT;
            rowtail = n % LogSoftmaxConst::ONE_TWENTY_EIGHT;
            if (!rowtail) {
                rowtail = LogSoftmaxConst::ONE_TWENTY_EIGHT;
            }
        }

        pipe.InitBuffer(inQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
        pipe.InitBuffer(workQueue, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT * LogSoftmaxConst::TWO);
        pipe.InitBuffer(outQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
    }

    __aicore__ inline void Process3(int batch, int start, int end)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal[LogSoftmaxConst::SIXTY_FOUR + chunksize];
        LocalTensor<float> expsum = workLocal[LogSoftmaxConst::SIXTY_FOUR + LogSoftmaxConst::TWO * chunksize];
        for (int i = start; i != end; i++) {
            int len = mmin(n - i * chunksize, chunksize);
            Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, chunksize);
            Duplicate(expsum, 0.0f, chunksize);
            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>((n - len) * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::TWO_FIFTY_SIX - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<float>();
                for (int k = 0; k != blocklen; k++) {
                    Max(oldmax, oldmax, inputLocal[k * chunksize], len);
                }
                inQueue.FreeTensor(inputLocal);
            }
            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>((n - len) * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::TWO_FIFTY_SIX - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<float>();
                for (int k = 0; k != blocklen; k++) {
                    Sub(inputLocal[k * chunksize], inputLocal[k * chunksize], oldmax, len);
                }
                Exp(inputLocal, inputLocal, blocklen * chunksize);
                for (int k = 0; k != blocklen; k++) {
                    Add(expsum, inputLocal[k * chunksize], expsum, len);
                }
                inQueue.FreeTensor(inputLocal);
            }

            Ln(expsum, expsum, len);

            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>((n - len) * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::TWO_FIFTY_SIX - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<float>();
                LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                for (int k = 0; k != blocklen; k++) {
                    Sub(outLocal[k * chunksize], inputLocal[k * chunksize], oldmax, len);
                    Sub(outLocal[k * chunksize], outLocal[k * chunksize], expsum, len);
                }
                inQueue.FreeTensor(inputLocal);
                outQueue.EnQue(outLocal);
                outQueue.DeQue<float>();
                DataCopyExtParams copyoutParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::TWO_FIFTY_SIX - (len + 7) / LogSoftmaxConst::EIGHT),
                    static_cast<uint32_t>((n - len) * sizeof(float)), 0};
                DataCopyPad(outGm[batch * m * n + j * n + i * chunksize], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);
            }
        }
    }

    __aicore__ inline void Process4(int batch, int start, int end, int chunk)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> expsum = workLocal[LogSoftmaxConst::SIXTY_FOUR + chunk];
        int blocklen = chunk / LogSoftmaxConst::EIGHT;
        for (int i = start; i != end; i++) {
            int len = mmin(n - i * chunk, chunk);
            Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, chunk);
            Duplicate(expsum, 0.0f, chunk);

            LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(m),
                static_cast<uint32_t>(len * sizeof(float)),
                static_cast<uint32_t>((n - len) * sizeof(float)),
                static_cast<uint32_t>(blocklen - (len + 7) / LogSoftmaxConst::EIGHT), 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm[batch * m * n + i * chunk], copyParams, padParams);
            inQueue.EnQue(inputLocal);
            inputLocal = inQueue.DeQue<float>();
            for (int k = 0; k != m; k++) {
                Max(oldmax, oldmax, inputLocal[k * chunk], len);
            }
            LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            for (int k = 0; k != m; k++) {
                Sub(outLocal[k * chunk], inputLocal[k * chunk], oldmax, len);
            }
            Exp(inputLocal, outLocal, chunk * m);
            for (int k = 0; k != m; k++) {
                Add(expsum, inputLocal[k * chunk], expsum, len);
            }
            inQueue.FreeTensor(inputLocal);
            Ln(expsum, expsum, chunk);
            for (int k = 0; k != m; k++) {
                Sub(outLocal[k * chunk], outLocal[k * chunk], expsum, len);
            }
            outQueue.EnQue(outLocal);
            outQueue.DeQue<float>();
            DataCopyExtParams copyoutParams{static_cast<uint16_t>(m),
                static_cast<uint32_t>(len * sizeof(float)),
                static_cast<uint32_t>(blocklen - (len + 7) / LogSoftmaxConst::EIGHT),
                static_cast<uint32_t>((n - len) * sizeof(float)), 0};
            DataCopyPad(outGm[batch * m * n + i * chunk], outLocal, copyoutParams);
            outQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void Process1(int start, int end)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT];
        LocalTensor<float> expsum = workLocal[LogSoftmaxConst::ONE_NINETY_TWO];
        LocalTensor<float> expmax = workLocal[LogSoftmaxConst::TWO_FIFTY_SIX];
        if (coliters == 1) {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::SIXTEEN;
                LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();

                DataCopyExtParams copyParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>((n - len) * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::TWO - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTEEN], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<float>();

                LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                Duplicate(newmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::SIXTEEN);
                Max(newmax, inputLocal, newmax, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 2, 0});

                Sub(outLocal, inputLocal, newmax, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(coltail), {1, 1, 1, 2, 2, 0});

                Exp(inputLocal, outLocal, coltail * LogSoftmaxConst::SIXTEEN);

                Duplicate(expsum, 0.0f, LogSoftmaxConst::SIXTEEN);
                Add(expsum, inputLocal, expsum, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 2, 0});

                Ln(expsum, expsum, LogSoftmaxConst::SIXTEEN);

                Sub(outLocal, outLocal, expsum, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(coltail), {1, 1, 1, 2, 2, 0});

                outQueue.EnQue(outLocal);
                outQueue.DeQue<float>();

                DataCopyExtParams copyoutParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::TWO - (len + 7) / LogSoftmaxConst::EIGHT),
                    static_cast<uint32_t>((n - len) * sizeof(float)), 0};

                DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::SIXTEEN], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);

                inQueue.FreeTensor(inputLocal);
            }
        } else {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::SIXTEEN;
                Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::SIXTEEN);
                Duplicate(expsum, 0.0f, LogSoftmaxConst::SIXTEEN);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>((n - len) * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::TWO - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTEEN + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<float>();

                    Max(oldmax, inputLocal, oldmax, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(collen), {1, 1, 1, 0, 2, 0});
                    inQueue.FreeTensor(inputLocal);
                }

                for (int j = 0; j != coliters; j++) {
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>((n - len) * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::TWO - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTEEN + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<float>();

                    Sub(inputLocal, inputLocal, oldmax, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(collen), {1, 1, 1, 2, 2, 0});
                    Exp(inputLocal, inputLocal, collen * LogSoftmaxConst::SIXTEEN);
                    Add(expsum, inputLocal, expsum, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(collen), {1, 1, 1, 0, 2, 0});
                    inQueue.FreeTensor(inputLocal);
                }
                Ln(expsum, expsum, LogSoftmaxConst::SIXTEEN);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>((n - len) * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::TWO - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTEEN + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<float>();
                    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                    Sub(outLocal, inputLocal, oldmax, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(collen), {1, 1, 1, 2, 2, 0});
                    inQueue.FreeTensor(inputLocal);
                    Sub(outLocal, outLocal, expsum, LogSoftmaxConst::SIXTEEN, static_cast<uint8_t>(collen), {1, 1, 1, 2, 2, 0});
                    outQueue.EnQue(outLocal);
                    outQueue.DeQue<float>();
                    DataCopyExtParams copyoutParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::TWO - (len + 7) / LogSoftmaxConst::EIGHT),
                        static_cast<uint32_t>((n - len) * sizeof(float)), 0};

                    DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::SIXTEEN + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], outLocal, copyoutParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }

    __aicore__ inline void Process2(int start, int end)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal[LogSoftmaxConst::ONE_NINETY_TWO];
        LocalTensor<float> expsum = workLocal[320];
        LocalTensor<float> expmax = workLocal[448];
        if (coliters == 1) {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();

                DataCopyExtParams copyParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>((n - len) * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<float>();

                LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                Duplicate(newmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Max(newmax, inputLocal, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});
                Max(newmax[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], newmax[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});

                Sub(outLocal, inputLocal, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});
                Sub(outLocal[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], newmax[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});

                Exp(inputLocal, outLocal, coltail * LogSoftmaxConst::ONE_TWENTY_EIGHT);

                Duplicate(expsum, 0.0f, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Add(expsum, inputLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});
                Add(expsum[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});

                Ln(expsum, expsum, LogSoftmaxConst::ONE_TWENTY_EIGHT);

                Sub(outLocal, outLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});
                Sub(outLocal[LogSoftmaxConst::SIXTY_FOUR], outLocal[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});

                outQueue.EnQue(outLocal);
                outQueue.DeQue<float>();

                DataCopyExtParams copyoutParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(float)),
                    static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 7) / LogSoftmaxConst::EIGHT),
                    static_cast<uint32_t>((n - len) * sizeof(float)), 0};

                DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);

                inQueue.FreeTensor(inputLocal);
            }
        } else {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Duplicate(expsum, 0.0f, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::SIXTY_FOUR;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>((n - len) * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<float>();

                    Max(oldmax, inputLocal, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    Max(oldmax[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    inQueue.FreeTensor(inputLocal);
                }

                for (int j = 0; j != coliters; j++) {
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::SIXTY_FOUR;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>((n - len) * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<float>();

                    Sub(inputLocal, inputLocal, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 16, 16, 0});
                    Sub(inputLocal[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 16, 16, 0});
                    Exp(inputLocal, inputLocal, collen * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    Add(expsum, inputLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    Add(expsum[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    inQueue.FreeTensor(inputLocal);
                }
                Ln(expsum, expsum, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::SIXTY_FOUR;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>((n - len) * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 7) / LogSoftmaxConst::EIGHT), 0};
                    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<float>();
                    LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
                    Sub(outLocal, inputLocal, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 16, 16, 0});
                    Sub(outLocal[LogSoftmaxConst::SIXTY_FOUR], inputLocal[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 16, 16, 0});
                    inQueue.FreeTensor(inputLocal);
                    Sub(outLocal, outLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 16, 16, 0});
                    Sub(outLocal[LogSoftmaxConst::SIXTY_FOUR], outLocal[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 16, 16, 0});

                    outQueue.EnQue(outLocal);
                    outLocal = outQueue.DeQue<float>();
                    DataCopyExtParams copyoutParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(float)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 7) / LogSoftmaxConst::EIGHT),
                        static_cast<uint32_t>((n - len) * sizeof(float)), 0};

                    DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], outLocal, copyoutParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 2048;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<TPosition::VECCALC> workQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<float> inputGm;
    GlobalTensor<float> outGm;

    int m, n, coliters, coltail, rowiters, rowtail;
};

class LogSoftmaxHalf {
public:
    __aicore__ inline LogSoftmaxHalf() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int m, int n)
    {
        this->inputGm.SetGlobalBuffer((__gm__ half *)input);
        this->outGm.SetGlobalBuffer((__gm__ half *)out);
        this->m = m;
        this->n = n;

        int len = n;
        int i = 0;
        while (len > LogSoftmaxConst::ONE_TWENTY_EIGHT) {
            this->lengthHalf[i] = len;
            len = (len + 15) / LogSoftmaxConst::SIXTEEN;
            i++;
        }
        this->itersHalf = i;
        this->lengthHalf[i] = len;

        len = n;
        i = 0;
        while (len > LogSoftmaxConst::SIXTY_FOUR) {
            this->length[i] = len;
            len = (len + 7) / LogSoftmaxConst::EIGHT;
            i++;
        }
        this->iters = i;
        this->length[i] = len;

        pipe.InitBuffer(inQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
        pipe.InitBuffer(workQueue1, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT + LogSoftmaxConst::TWO_FIFTY_SIX);
        pipe.InitBuffer(workQueue2, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
        pipe.InitBuffer(outQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
    }

    __aicore__ inline void Process1(int start, int end)
    {
        int align_n = align_to(n, LogSoftmaxConst::SIXTY_FOUR);
        if (align_n == 0) {
            return;
        }
        int step = LogSoftmaxConst::EIGHT_THOUSAND_ONE_HUNDRED_NINETY_TWO / align_n;
        int loop = (end - start + step - 1) / step;
        int tail = (end - start) % step;
        tail = tail == 0 ? step : tail;
        DataCopyExtParams copyParams{uint16_t(tail), uint32_t(n * LogSoftmaxConst::HALF_SIZE), 0,
            uint32_t((align_n - n) / LogSoftmaxConst::HALF_ALIGN_BLOCK), 0};
        DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
        LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
        DataCopyPad(inputLocal, inputGm[start * n], copyParams, padParams);
        inQueue.EnQue(inputLocal);
        int offset = start * n;
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> workLocal2 = workQueue2.Get<float>();
        DataCopyExtParams copyParamsOut{uint16_t(tail), static_cast<uint32_t>(n * LogSoftmaxConst::HALF_SIZE),
            uint32_t((align_n - n) / LogSoftmaxConst::HALF_ALIGN_BLOCK), 0, 0};
        copyParams.blockCount = step;
        for (int i = 0; i != loop; i++) {
            inputLocal = inQueue.DeQue<half>();
            if (i != loop - 1) {
                LocalTensor<half> aLocal = inQueue.AllocTensor<half>();
                DataCopyPad(aLocal, inputGm[(i * step + start + tail) * n], copyParams, padParams);
                inQueue.EnQue(aLocal);
            }

            LocalTensor<half> srcLocal, dstLocal;
            LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
            int lines = i == 0 ? tail : step;
            SetMaskCount();
            for (int k = 0; k != lines; k++) {
                srcLocal = inputLocal[k * align_n];
                dstLocal = outLocal[k * align_n];
                for (int j = 0; j != itersHalf; j++) {
                    SetVectorMask<half>(0, lengthHalf[j]);
                    BlockReduceMax<half, false>(dstLocal, srcLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                    PipeBarrier<PIPE_V>();
                    srcLocal = dstLocal;
                }
                SetVectorMask<half>(0, lengthHalf[itersHalf]);
                WholeReduceMax<half, false>(dstLocal, srcLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();

            Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, align_n * lines);
            inQueue.FreeTensor(inputLocal);
            for (int k = 0; k != lines; k++) {
                Adds(workLocal2[k * align_n], workLocal1[k * align_n], -((float)outLocal(k * align_n)), n);
            }
            Exp(workLocal1, workLocal2, align_n * lines);

            SetMaskCount();
            for (int k = 0; k != lines; k++) {
                LocalTensor<float> workLocal = workLocal1[k * align_n];
                for (int j = 0; j != iters; j++) {
                    SetVectorMask<float>(0, length[j]);
                    BlockReduceSum<float, false>(workLocal, workLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                    PipeBarrier<PIPE_V>();
                }
                SetVectorMask<float>(0, length[iters]);
                WholeReduceSum<float, false>(workLocal, workLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();

            for (int k = 0; k != lines; k++) {
                Ln(workLocal1[k * align_n], workLocal1[k * align_n], LogSoftmaxConst::EIGHT);
                Adds(workLocal2[k * align_n], workLocal2[k * align_n], -workLocal1(k * align_n), n);
            }

            Cast(outLocal, workLocal2, RoundMode::CAST_RINT, lines * align_n);

            outQueue.EnQue(outLocal);
            outQueue.DeQue<half>();
            copyParamsOut.blockCount = lines;
            DataCopyPad(outGm[offset], outLocal, copyParamsOut);
            offset += n * lines;
            outQueue.FreeTensor(outLocal);
        }
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 4096;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> workQueue1;
    TBuf<TPosition::VECCALC> workQueue2;

    GlobalTensor<half> inputGm;
    GlobalTensor<half> outGm;

    int m, n, iters, itersHalf;
    int length[LogSoftmaxConst::LENGTH_ARRAY_SIZE], lengthHalf[LogSoftmaxConst::LENGTH_ARRAY_SIZE];
};

class LogSoftmaxHalfCol {
public:
    __aicore__ inline LogSoftmaxHalfCol() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int m, int n, int flag)
    {
        this->inputGm.SetGlobalBuffer((__gm__ half *)input);
        this->outGm.SetGlobalBuffer((__gm__ half *)out);
        this->m = m;
        this->n = n;
        if (flag) {
            coliters = (m + LogSoftmaxConst::ONE_TWENTY_EIGHT - 1) / LogSoftmaxConst::ONE_TWENTY_EIGHT;
            coltail = m % LogSoftmaxConst::ONE_TWENTY_EIGHT;
            if (!coltail) {
                coltail = LogSoftmaxConst::ONE_TWENTY_EIGHT;
            }
            rowiters = (n + LogSoftmaxConst::SIXTY_FOUR - 1) / LogSoftmaxConst::SIXTY_FOUR;
            rowtail = n % LogSoftmaxConst::SIXTY_FOUR;
            if (!rowtail) {
                rowtail = LogSoftmaxConst::SIXTY_FOUR;
            }
        } else {
            coliters = (m + LogSoftmaxConst::THIRTY_TWO - 1) / LogSoftmaxConst::THIRTY_TWO;
            coltail = m % LogSoftmaxConst::THIRTY_TWO;
            if (!coltail) {
                coltail = LogSoftmaxConst::THIRTY_TWO;
            }
            rowiters = (n + LogSoftmaxConst::TWO_FIFTY_SIX - 1) / LogSoftmaxConst::TWO_FIFTY_SIX;
            rowtail = n % LogSoftmaxConst::TWO_FIFTY_SIX;
            if (!rowtail) {
                rowtail = LogSoftmaxConst::TWO_FIFTY_SIX;
            }
        }

        pipe.InitBuffer(inQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
        pipe.InitBuffer(workQueue, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
        pipe.InitBuffer(workQueue1, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT * 3);
        pipe.InitBuffer(outQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
    }

    __aicore__ inline void Process4(int batch, int start, int end, int chunk)
    {
        LocalTensor<half> workLocal = workQueue.Get<half>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<half> oldmax = workLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT];
        LocalTensor<float> expsum = workLocal1[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal1[LogSoftmaxConst::SIXTY_FOUR + chunk];
        LocalTensor<float> workLocal2 = workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT + LogSoftmaxConst::TWO * chunk];
        LocalTensor<float> workLocal3 = workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT + (LogSoftmaxConst::TWO + m) * chunk];
        int blocklen = chunk / LogSoftmaxConst::SIXTEEN;
        for (int i = start; i != end; i++) {
            int len = mmin(n - i * chunk, chunk);
            Duplicate(oldmax, LogSoftmaxConst::HALF_NEG_INF, chunk);
            Duplicate(expsum, 0.0f, chunk);

            LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(m), static_cast<uint32_t>(len * sizeof(half)), static_cast<uint32_t>((n - len) * sizeof(half)), static_cast<uint32_t>(blocklen - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
            DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm[batch * m * n + i * chunk], copyParams, padParams);
            inQueue.EnQue(inputLocal);
            inputLocal = inQueue.DeQue<half>();

            for (int k = 0; k != m; k++) {
                Max(oldmax, oldmax, inputLocal[k * chunk], len);
            }
            Cast(workLocal2, inputLocal, RoundMode::CAST_NONE, m * chunk);
            inQueue.FreeTensor(inputLocal);
            Cast(newmax, oldmax, RoundMode::CAST_NONE, len);
            for (int k = 0; k != m; k++) {
                Sub(workLocal2[k * chunk], workLocal2[k * chunk], newmax, len);
            }
            Exp(workLocal3, workLocal2, chunk * m);
            for (int k = 0; k != m; k++) {
                Add(expsum, workLocal3[k * chunk], expsum, len);
            }
            Ln(expsum, expsum, chunk);
            for (int k = 0; k != m; k++) {
                Sub(workLocal2[k * chunk], workLocal2[k * chunk], expsum, len);
            }
            LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
            Cast(outLocal, workLocal2, RoundMode::CAST_RINT, chunk * m);
            outQueue.EnQue(outLocal);
            outQueue.DeQue<half>();
            DataCopyExtParams copyoutParams{static_cast<uint16_t>(m), static_cast<uint32_t>(len * sizeof(half)), static_cast<uint32_t>(blocklen - (len + 15) / LogSoftmaxConst::SIXTEEN), static_cast<uint32_t>((n - len) * sizeof(half)), 0};
            DataCopyPad(outGm[batch * m * n + i * chunk], outLocal, copyoutParams);
            outQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void Process3(int batch, int start, int end)
    {
        LocalTensor<half> workLocal = workQueue.Get<half>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<half> oldmax = workLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT];
        LocalTensor<half> newmax = workLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT + chunksize];
        LocalTensor<float> expsum = workLocal1[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> workLocal2 = workLocal1[LogSoftmaxConst::SIXTY_FOUR + chunksize];
        LocalTensor<float> workLocal3 = workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT + LogSoftmaxConst::TWO * chunksize];
        LocalTensor<float> workLocal4 = workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT + 3 * chunksize];

        for (int i = start; i != end; i++) {
            int len = mmin(n - i * chunksize, chunksize);
            Duplicate(oldmax, LogSoftmaxConst::HALF_NEG_INF, chunksize);
            Duplicate(expsum, 0.0f, chunksize);
            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>((n - len) * sizeof(half)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<half>();
                for (int k = 0; k != blocklen; k++) {
                    Max(oldmax, oldmax, inputLocal[k * chunksize], len);
                }
                inQueue.FreeTensor(inputLocal);
            }
            Cast(workLocal2, oldmax, RoundMode::CAST_NONE, len);
            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>((n - len) * sizeof(half)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<half>();
                for (int k = 0; k != blocklen; k++) {
                    Cast(workLocal3, inputLocal[k * chunksize], RoundMode::CAST_NONE, len);
                    Sub(workLocal3, workLocal3, workLocal2, len);
                    Exp(workLocal3, workLocal3, len);
                    Add(expsum, expsum, workLocal3, len);
                }
                inQueue.FreeTensor(inputLocal);
            }
            Ln(expsum, expsum, len);
            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m -j, LogSoftmaxConst::FOUR);
                LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>((n - len) * sizeof(half)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<half>();
                LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
                for (int k = 0; k != blocklen; k++) {
                    Cast(workLocal3, inputLocal[k * chunksize], RoundMode::CAST_NONE, len);
                    Sub(workLocal3, workLocal3, workLocal2, len);
                    Sub(workLocal3, workLocal3, expsum, len);
                    Cast(outLocal[k * chunksize], workLocal3, RoundMode::CAST_RINT, len);
                }
                inQueue.FreeTensor(inputLocal);
                outQueue.EnQue(outLocal);
                outQueue.DeQue<half>();
                DataCopyExtParams copyoutParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(half)),
         static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN),
                    static_cast<uint32_t>((n - len) * sizeof(half)), 0};
                DataCopyPad(outGm[batch * m * n + j * n + i * chunksize], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);
            }
        }
    }
    __aicore__ inline void Process1(int start, int end)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT];
        LocalTensor<float> expsum = workLocal[LogSoftmaxConst::ONE_NINETY_TWO];
        LocalTensor<float> expmax = workLocal[LogSoftmaxConst::TWO_FIFTY_SIX];
        if (coliters == 1) {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::SIXTY_FOUR;
                LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();

                DataCopyExtParams copyParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>((n - len) * sizeof(half)),
                    static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<half>();

                LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
                Duplicate(outLocal, LogSoftmaxConst::HALF_NEG_INF, LogSoftmaxConst::SIXTY_FOUR);
                Max(outLocal, inputLocal, outLocal, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 8, 0});

                Cast(workLocal, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::SIXTY_FOUR);
                Cast(workLocal1, outLocal, RoundMode::CAST_NONE, LogSoftmaxConst::SIXTY_FOUR);
                Sub(workLocal, workLocal, workLocal1, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 8, 8, 0});
                Exp(workLocal, workLocal, coltail * LogSoftmaxConst::SIXTY_FOUR);

                Duplicate(expsum, 0.0f, LogSoftmaxConst::SIXTY_FOUR);
                Add(expsum, workLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 8, 0});

                Ln(expsum, expsum, LogSoftmaxConst::SIXTY_FOUR);

                Cast(workLocal, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::SIXTY_FOUR);
                Sub(workLocal, workLocal, workLocal1, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 8, 8, 0});
                Sub(workLocal, workLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 8, 8, 0});
                Cast(outLocal, workLocal, RoundMode::CAST_RINT, coltail * LogSoftmaxConst::SIXTY_FOUR);

                outQueue.EnQue(outLocal);
                outQueue.DeQue<half>();

                DataCopyExtParams copyoutParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN),
                    static_cast<uint32_t>((n - len) * sizeof(half)), 0};

                DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);

                inQueue.FreeTensor(inputLocal);
            }
        } else {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::SIXTY_FOUR;
                Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::SIXTY_FOUR);
                Duplicate(expsum, 0.0f, LogSoftmaxConst::SIXTY_FOUR);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>((n - len) * sizeof(half)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<half>();

                    Cast(workLocal, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::SIXTY_FOUR);
                    Max(oldmax, workLocal, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 8, 0});
                    inQueue.FreeTensor(inputLocal);
                }
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>((n - len) * sizeof(half)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<half>();

                    Cast(workLocal, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::SIXTY_FOUR);
                    Sub(workLocal, workLocal, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});
                    Exp(workLocal, workLocal, collen * LogSoftmaxConst::SIXTY_FOUR);
                    Add(expsum, workLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 8, 0});
                    inQueue.FreeTensor(inputLocal);
                }
                Ln(expsum, expsum, LogSoftmaxConst::SIXTY_FOUR);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>((n - len) * sizeof(half)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<half>();

                    LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
                    Cast(workLocal, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::SIXTY_FOUR);
                    inQueue.FreeTensor(inputLocal);
                    Sub(workLocal, workLocal, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});
                    Sub(workLocal, workLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});
                    Cast(outLocal, workLocal, RoundMode::CAST_RINT, collen * LogSoftmaxConst::SIXTY_FOUR);
                    outQueue.EnQue(outLocal);
                    outQueue.DeQue<half>();
                    DataCopyExtParams copyoutParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN),
                        static_cast<uint32_t>((n - len) * sizeof(half)), 0};

                    DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], outLocal, copyoutParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }
    __aicore__ inline void Process2(int start, int end)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal[320];
        LocalTensor<float> expsum = workLocal[576];
        LocalTensor<float> expmax = workLocal[832];
        if (coliters == 1) {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::TWO_FIFTY_SIX;
                LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();

                DataCopyExtParams copyParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>((n - len) * sizeof(half)),
                    static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::TWO_FIFTY_SIX], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<half>();

                LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
                Duplicate(outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], LogSoftmaxConst::HALF_NEG_INF, LogSoftmaxConst::TWO_FIFTY_SIX);
                Max(outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], inputLocal, outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                    LogSoftmaxConst::ONE_TWENTY_EIGHT, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});
                Max(outLocal[LogSoftmaxConst::TWO_FIFTY_SIX], inputLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], outLocal[LogSoftmaxConst::TWO_FIFTY_SIX],
                    LogSoftmaxConst::ONE_TWENTY_EIGHT, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});

                Cast(newmax, outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], RoundMode::CAST_NONE, LogSoftmaxConst::TWO_FIFTY_SIX);
                Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::TWO_FIFTY_SIX);

                Sub(workLocal1, workLocal1, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});
                Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], newmax[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});
                Sub(workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], newmax[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});
                Sub(workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], newmax[LogSoftmaxConst::ONE_NINETY_TWO],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});

                Exp(workLocal1, workLocal1, coltail * LogSoftmaxConst::TWO_FIFTY_SIX);

                Duplicate(expsum, 0.0f, LogSoftmaxConst::TWO_FIFTY_SIX);
                Add(expsum, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 32, 0});
                Add(expsum[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 32, 0});
                Add(expsum[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], expsum[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 32, 0});
                Add(expsum[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], expsum[LogSoftmaxConst::ONE_NINETY_TWO],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 32, 0});

                Ln(expsum, expsum, LogSoftmaxConst::TWO_FIFTY_SIX);
                Add(expsum, expsum, newmax, LogSoftmaxConst::TWO_FIFTY_SIX);
                Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::TWO_FIFTY_SIX);

                Sub(workLocal1, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});
                Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});
                Sub(workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], expsum[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});
                Sub(workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], expsum[LogSoftmaxConst::ONE_NINETY_TWO],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 32, 32, 0});

                Cast(outLocal, workLocal1, RoundMode::CAST_RINT, coltail * LogSoftmaxConst::TWO_FIFTY_SIX);

                outQueue.EnQue(outLocal);
                outQueue.DeQue<half>();

                DataCopyExtParams copyoutParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(half)),
                    static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 15) / LogSoftmaxConst::SIXTEEN),
                    static_cast<uint32_t>((n - len) * sizeof(half)), 0};

                DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::TWO_FIFTY_SIX], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);

                inQueue.FreeTensor(inputLocal);
            }
        } else {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::TWO_FIFTY_SIX;
                Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::TWO_FIFTY_SIX);
                Duplicate(expsum, 0.0f, LogSoftmaxConst::TWO_FIFTY_SIX);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::THIRTY_TWO;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>((n - len) * sizeof(half)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::TWO_FIFTY_SIX + j * LogSoftmaxConst::THIRTY_TWO * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<half>();

                    LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
                    Duplicate(outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], LogSoftmaxConst::HALF_NEG_INF, LogSoftmaxConst::TWO_FIFTY_SIX);
                    Max(outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], inputLocal, outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        LogSoftmaxConst::ONE_TWENTY_EIGHT, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    Max(outLocal[LogSoftmaxConst::TWO_FIFTY_SIX], inputLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], outLocal[LogSoftmaxConst::TWO_FIFTY_SIX],
                        LogSoftmaxConst::ONE_TWENTY_EIGHT, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    Cast(oldmax, outLocal[LogSoftmaxConst::ONE_TWENTY_EIGHT], RoundMode::CAST_NONE, LogSoftmaxConst::TWO_FIFTY_SIX);

                    inQueue.FreeTensor(inputLocal);
                    outQueue.FreeTensor(outLocal);
                }
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::THIRTY_TWO;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>((n - len) * sizeof(half)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::TWO_FIFTY_SIX + j * LogSoftmaxConst::THIRTY_TWO * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<half>();
                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::TWO_FIFTY_SIX);
                    inQueue.FreeTensor(inputLocal);

                    Sub(workLocal1, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], oldmax[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], oldmax[LogSoftmaxConst::ONE_NINETY_TWO],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});

                    Exp(workLocal1, workLocal1, collen * LogSoftmaxConst::TWO_FIFTY_SIX);
                    Add(expsum, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 32, 0});
                    Add(expsum[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 32, 0});
                    Add(expsum[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], expsum[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 32, 0});
                    Add(expsum[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], expsum[LogSoftmaxConst::ONE_NINETY_TWO],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 32, 0});
                }
                Ln(expsum, expsum, LogSoftmaxConst::TWO_FIFTY_SIX);
                for (int k = 0; k != coliters; k++) {
                    LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
                    int collen = k == (coliters - 1) ? coltail : LogSoftmaxConst::THIRTY_TWO;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>((n - len) * sizeof(half)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<half> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::TWO_FIFTY_SIX + k * LogSoftmaxConst::THIRTY_TWO * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<half>();
                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::TWO_FIFTY_SIX);
                    inQueue.FreeTensor(inputLocal);

                    Sub(workLocal1, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], oldmax[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], oldmax[LogSoftmaxConst::ONE_NINETY_TWO],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], workLocal1[LogSoftmaxConst::ONE_TWENTY_EIGHT], expsum[LogSoftmaxConst::ONE_TWENTY_EIGHT],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});
                    Sub(workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], workLocal1[LogSoftmaxConst::ONE_NINETY_TWO], expsum[LogSoftmaxConst::ONE_NINETY_TWO],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 32, 32, 0});

                    LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
                    Cast(outLocal, workLocal1, RoundMode::CAST_RINT, collen * LogSoftmaxConst::TWO_FIFTY_SIX);
                    outQueue.EnQue(outLocal);
                    outQueue.DeQue<half>();
                    DataCopyExtParams copyoutParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(half)),
                        static_cast<uint32_t>(LogSoftmaxConst::SIXTEEN - (len + 15) / LogSoftmaxConst::SIXTEEN),
                        static_cast<uint32_t>((n - len) * sizeof(half)), 0};

                    DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::TWO_FIFTY_SIX + k * LogSoftmaxConst::THIRTY_TWO * n], outLocal, copyoutParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 2048;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<TPosition::VECCALC> workQueue;
    TBuf<TPosition::VECCALC> workQueue1;
    TQue<QuePosition::VECOUT, 1> outQueue;

    GlobalTensor<half> inputGm;
    GlobalTensor<half> outGm;

    int m, n, coliters, coltail, rowiters, rowtail;
};

class LogSoftmaxBf16 {
public:
    __aicore__ inline LogSoftmaxBf16() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int m, int n)
    {
        this->inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input);
        this->outGm.SetGlobalBuffer((__gm__ bfloat16_t *)out);
        this->m = m;
        this->n = n;

        int len = n;
        int k = 0;
        while (len > LogSoftmaxConst::SIXTY_FOUR) {
            this->length[k] = len;
            len = (len + 7) / LogSoftmaxConst::EIGHT;
            k++;
        }
        this->iters = k;
        this->length[k] = len;

        pipe.InitBuffer(inQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
        pipe.InitBuffer(workQueue1, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT + LogSoftmaxConst::TWO_FIFTY_SIX);
        pipe.InitBuffer(workQueue2, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT);
        pipe.InitBuffer(outQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
    }

    __aicore__ inline void Process1(int start, int end)
    {
        int align_n = align_to(n, LogSoftmaxConst::SIXTY_FOUR);
        if (align_n == 0) {
            return;
        }
        int step = LogSoftmaxConst::EIGHT_THOUSAND_ONE_HUNDRED_NINETY_TWO / align_n;
        int loop = (end - start + step - 1) / step;
        int tail = (end - start) % step;
        tail = tail == 0 ? step : tail;
        DataCopyExtParams copyParams{uint16_t(tail), uint32_t(n * LogSoftmaxConst::BF16_SIZE), 0,
            uint32_t((align_n - n) / LogSoftmaxConst::HALF_ALIGN_BLOCK), 0};
        DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

        LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
        DataCopyPad(inputLocal, inputGm[start * n], copyParams, padParams);
        inQueue.EnQue(inputLocal);
        int offset = start * n;
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> workLocal2 = workQueue2.Get<float>();
        DataCopyExtParams copyParamsOut{uint16_t(tail), static_cast<uint32_t>(n * LogSoftmaxConst::BF16_SIZE),
            uint32_t((align_n - n) / LogSoftmaxConst::HALF_ALIGN_BLOCK), 0, 0};
        copyParams.blockCount = step;
        for (int i = 0; i != loop; i++) {
            inputLocal = inQueue.DeQue<bfloat16_t>();
            if (i != loop - 1) {
                LocalTensor<bfloat16_t> aLocal = inQueue.AllocTensor<bfloat16_t>();
                DataCopyPad(aLocal, inputGm[(i * step + start + tail) * n], copyParams, padParams);
                inQueue.EnQue(aLocal);
            }
            int lines = i == 0 ? tail : step;
            Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, align_n * lines);
            inQueue.FreeTensor(inputLocal);
            LocalTensor<float> srcLocal, dstLocal;
            LocalTensor<bfloat16_t> outLocal_bf16 = outQueue.AllocTensor<bfloat16_t>();

            SetMaskCount();
            for (int k = 0; k != lines; k++) {
                srcLocal = workLocal1[k * align_n];
                dstLocal = workLocal2[k * align_n];
                for (int j = 0; j != iters; j++) {
                    SetVectorMask<float>(0, length[j]);
                    BlockReduceMax<float, false>(dstLocal, srcLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                    PipeBarrier<PIPE_V>();
                    srcLocal = dstLocal;
                }
                SetVectorMask<float>(0, length[iters]);
                WholeReduceMax<float, false>(dstLocal, srcLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();

            for (int k = 0; k != lines; k++) {
                float scale = -workLocal2(k * align_n);
                Adds(workLocal2[k * align_n], workLocal1[k * align_n], scale, n);
            }
            Exp(workLocal1, workLocal2, align_n * lines);

            SetMaskCount();
            for (int j = 0; j != lines; j++) {
                LocalTensor<float> workLocal = workLocal1[j * align_n];
                for (int k = 0; k != iters; k++) {
                    SetVectorMask<float>(0, length[k]);
                    BlockReduceSum<float, false>(workLocal, workLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                    PipeBarrier<PIPE_V>();
                }
                SetVectorMask<float>(0, length[iters]);
                WholeReduceSum<float, false>(workLocal, workLocal, 1, MASK_PLACEHOLDER, 1, 1, LogSoftmaxConst::EIGHT);
                PipeBarrier<PIPE_V>();
            }
            SetMaskNorm();

            for (int k = 0; k != lines; k++) {
                Ln(workLocal1[k * align_n], workLocal1[k * align_n], LogSoftmaxConst::EIGHT);
                Adds(workLocal2[k * align_n], workLocal2[k * align_n], -workLocal1(k * align_n), n);
            }

            Cast(outLocal_bf16, workLocal2, RoundMode::CAST_RINT, lines * align_n);

            outQueue.EnQue(outLocal_bf16);
            outQueue.DeQue<bfloat16_t>();

            copyParamsOut.blockCount = lines;
            DataCopyPad(outGm[offset], outLocal_bf16, copyParamsOut);
            offset += n * lines;
            outQueue.FreeTensor(outLocal_bf16);
        }
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 4096;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> workQueue1;
    TBuf<TPosition::VECCALC> workQueue2;

    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<bfloat16_t> outGm;

    int m, n, iters;
    int length[LogSoftmaxConst::LENGTH_ARRAY_SIZE];
};

class LogSoftmaxBF16Col {
public:
    __aicore__ inline LogSoftmaxBF16Col() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR out, int m, int n, int flag)
    {
        this->inputGm.SetGlobalBuffer((__gm__ bfloat16_t *)input);
        this->outGm.SetGlobalBuffer((__gm__ bfloat16_t *)out);
        this->m = m;
        this->n = n;
        if (flag) {
            coliters = (m + LogSoftmaxConst::SIXTY_FOUR - 1) / LogSoftmaxConst::SIXTY_FOUR;
            coltail = m % LogSoftmaxConst::SIXTY_FOUR;
            if (!coltail) {
                coltail = LogSoftmaxConst::SIXTY_FOUR;
            }
            rowiters = (n + LogSoftmaxConst::ONE_TWENTY_EIGHT - 1) / LogSoftmaxConst::ONE_TWENTY_EIGHT;
            rowtail = n % LogSoftmaxConst::ONE_TWENTY_EIGHT;
            if (!rowtail) {
                rowtail = LogSoftmaxConst::ONE_TWENTY_EIGHT;
            }
        } else {
            coliters = (m + LogSoftmaxConst::ONE_TWENTY_EIGHT - 1) / LogSoftmaxConst::ONE_TWENTY_EIGHT;
            coltail = m % LogSoftmaxConst::ONE_TWENTY_EIGHT;
            if (!coltail) {
                coltail = LogSoftmaxConst::ONE_TWENTY_EIGHT;
            }
            rowiters = (n + LogSoftmaxConst::SIXTY_FOUR - 1) / LogSoftmaxConst::SIXTY_FOUR;
            rowtail = n % LogSoftmaxConst::SIXTY_FOUR;
            if (!rowtail) {
                rowtail = LogSoftmaxConst::SIXTY_FOUR;
            }
        }

        pipe.InitBuffer(inQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
        pipe.InitBuffer(workQueue, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT + LogSoftmaxConst::TWO_FIFTY_SIX);
        pipe.InitBuffer(workQueue1, LogSoftmaxConst::THIRTY_TWO_THOUSAND_SEVEN_HUNDRED_SIXTY_EIGHT * 2 + LogSoftmaxConst::TWO_FIFTY_SIX);
        pipe.InitBuffer(outQueue, LogSoftmaxConst::TWO, LogSoftmaxConst::SIXTEEN_THOUSAND_THREE_HUNDRED_EIGHTY_FOUR);
    }

    __aicore__ inline void Process3(int start, int end)
    {
        LocalTensor<float> workLocal1 = workQueue.Get<float>();
        LocalTensor<float> workLocal = workQueue1.Get<float>();
        LocalTensor<float> oldmax = workLocal1[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal1[LogSoftmaxConst::ONE_NINETY_TWO];
        LocalTensor<float> expsum = workLocal1[320];
        LocalTensor<float> expmax = workLocal1[448];
        if (coliters == 1) {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::SIXTY_FOUR;
                LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();

                DataCopyExtParams copyParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<bfloat16_t>();

                LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
                Duplicate(newmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::SIXTY_FOUR);
                Cast(workLocal, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::SIXTY_FOUR);
                Max(newmax, workLocal, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 8, 0});

                Sub(workLocal, workLocal, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 8, 8, 0});

                Exp(workLocal, workLocal, coltail * LogSoftmaxConst::SIXTY_FOUR);

                Duplicate(expsum, 0.0f, LogSoftmaxConst::SIXTY_FOUR);
                Add(expsum, workLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 8, 0});

                Ln(expsum, expsum, LogSoftmaxConst::SIXTY_FOUR);
                Add(expsum, expsum, newmax, LogSoftmaxConst::SIXTY_FOUR);
                Cast(workLocal, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::SIXTY_FOUR);

                Sub(workLocal, workLocal, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 8, 8, 0});
                Cast(outLocal, workLocal, RoundMode::CAST_RINT, coltail * LogSoftmaxConst::SIXTY_FOUR);

                outQueue.EnQue(outLocal);
                outQueue.DeQue<bfloat16_t>();

                DataCopyExtParams copyoutParams1{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)), 0};

                DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR], outLocal, copyoutParams1);
                outQueue.FreeTensor(outLocal);

                inQueue.FreeTensor(inputLocal);
            }
        } else {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::SIXTY_FOUR;
                Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::SIXTY_FOUR);
                Duplicate(expsum, 0.0f, LogSoftmaxConst::SIXTY_FOUR);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<bfloat16_t>();

                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::SIXTY_FOUR);

                    Max(oldmax, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 8, 0});
                    inQueue.FreeTensor(inputLocal);
                    Sub(workLocal1, workLocal1, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});
                }
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<bfloat16_t>();

                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::SIXTY_FOUR);
                    inQueue.FreeTensor(inputLocal);
                    Sub(workLocal1, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});
                    Exp(workLocal1, workLocal1, collen * LogSoftmaxConst::SIXTY_FOUR);
                    Add(expsum, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 0, 8, 0});
                }
                Ln(expsum, expsum, LogSoftmaxConst::SIXTY_FOUR);
                for (int j = 0; j != coliters; j++) {
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<bfloat16_t>();
                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::SIXTY_FOUR);
                    inQueue.FreeTensor(inputLocal);
                    Sub(workLocal1, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});
                    Sub(workLocal1, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 8, 8, 0});

                    LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
                    Cast(outLocal, workLocal1, RoundMode::CAST_RINT, collen * LogSoftmaxConst::SIXTY_FOUR);
                    outQueue.EnQue(outLocal);
                    outQueue.DeQue<bfloat16_t>();

                    DataCopyExtParams copyoutParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(4 - (len + 15) / LogSoftmaxConst::SIXTEEN),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)), 0};

                    DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::SIXTY_FOUR + j * LogSoftmaxConst::ONE_TWENTY_EIGHT * n], outLocal, copyoutParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }
    __aicore__ inline void Process1(int start, int end)
    {
        LocalTensor<float> workLocal0 = workQueue.Get<float>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> oldmax = workLocal0[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal0[LogSoftmaxConst::ONE_NINETY_TWO];
        LocalTensor<float> expsum = workLocal0[320];
        LocalTensor<float> expmax = workLocal0[448];
        if (coliters == 1) {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();

                DataCopyExtParams copyParams{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(8 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<bfloat16_t>();

                LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
                Duplicate(newmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Max(newmax, workLocal1, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});
                Max(newmax[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], newmax[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});

                Sub(workLocal1, workLocal1, newmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});
                Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], newmax[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});

                Exp(workLocal1, workLocal1, coltail * LogSoftmaxConst::ONE_TWENTY_EIGHT);

                Duplicate(expsum, 0.0f, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Add(expsum, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});
                Add(expsum[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 0, 16, 0});

                Ln(expsum, expsum, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Add(expsum, expsum, newmax, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, coltail * LogSoftmaxConst::ONE_TWENTY_EIGHT);

                Sub(workLocal1, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});
                Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                    LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(coltail), {1, 1, 1, 16, 16, 0});
                Cast(outLocal, workLocal1, RoundMode::CAST_RINT, coltail * LogSoftmaxConst::ONE_TWENTY_EIGHT);

                outQueue.EnQue(outLocal);
                outQueue.DeQue<bfloat16_t>();

                DataCopyExtParams copyoutParams2{static_cast<uint16_t>(coltail),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(8 - (len + 15) / LogSoftmaxConst::SIXTEEN),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)), 0};

                DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT], outLocal, copyoutParams2);
                outQueue.FreeTensor(outLocal);

                inQueue.FreeTensor(inputLocal);
            }
        } else {
            for (int i = start; i != end; i++) {
                int batch = i / rowiters;
                int index = i % rowiters;
                int len = index == (rowiters - 1) ? rowtail : LogSoftmaxConst::ONE_TWENTY_EIGHT;
                Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                Duplicate(expsum, 0.0f, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::SIXTY_FOUR;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(8 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<bfloat16_t>();

                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    inQueue.FreeTensor(inputLocal);

                    Max(oldmax, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                    Max(oldmax[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, static_cast<uint8_t>(collen), {1, 1, 1, 0, 16, 0});
                }
                for (int j = 0; j != coliters; j++) {
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::SIXTY_FOUR;
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(8 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<bfloat16_t>();

                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    inQueue.FreeTensor(inputLocal);

                    Sub(workLocal1, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 16, 16, 0});
                    Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 16, 16, 0});
                    Exp(workLocal1, workLocal1, collen * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    Add(expsum, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 0, 16, 0});
                    Add(expsum[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 0, 16, 0});
                }
                Ln(expsum, expsum, LogSoftmaxConst::ONE_TWENTY_EIGHT);
                for (int j = 0; j != coliters; j++) {
                    int collen = j == (coliters - 1) ? coltail : LogSoftmaxConst::SIXTY_FOUR;
                    LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                    DataCopyExtParams copyParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(8 - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                    DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};

                    DataCopyPad(inputLocal, inputGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], copyParams, padParams);
                    inQueue.EnQue(inputLocal);
                    inputLocal = inQueue.DeQue<bfloat16_t>();
                    Cast(workLocal1, inputLocal, RoundMode::CAST_NONE, collen * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    inQueue.FreeTensor(inputLocal);
                    Sub(workLocal1, workLocal1, oldmax, LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 16, 16, 0});
                    Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], oldmax[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 16, 16, 0});
                    Sub(workLocal1, workLocal1, expsum, LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 16, 16, 0});
                    Sub(workLocal1[LogSoftmaxConst::SIXTY_FOUR], workLocal1[LogSoftmaxConst::SIXTY_FOUR], expsum[LogSoftmaxConst::SIXTY_FOUR],
                        LogSoftmaxConst::SIXTY_FOUR, uint8_t(collen), {1, 1, 1, 16, 16, 0});

                    LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
                    Cast(outLocal, workLocal1, RoundMode::CAST_RINT, collen * LogSoftmaxConst::ONE_TWENTY_EIGHT);
                    outQueue.EnQue(outLocal);
                    outQueue.DeQue<bfloat16_t>();

                    DataCopyExtParams copyoutParams{static_cast<uint16_t>(collen),
                        static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                        static_cast<uint32_t>(8 - (len + 15) / LogSoftmaxConst::SIXTEEN),
                        static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)), 0};

                    DataCopyPad(outGm[batch * m * n + index * LogSoftmaxConst::ONE_TWENTY_EIGHT + j * LogSoftmaxConst::SIXTY_FOUR * n], outLocal, copyoutParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }
    __aicore__ inline void Process2(int batch, int start, int end)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> oldmax = workLocal[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> newmax = workLocal[LogSoftmaxConst::SIXTY_FOUR + chunksize];
        LocalTensor<float> expsum = workLocal1[LogSoftmaxConst::SIXTY_FOUR];
        LocalTensor<float> workLocal2 = workLocal1[LogSoftmaxConst::SIXTY_FOUR + chunksize];
        LocalTensor<float> workLocal3 = workLocal1[LogSoftmaxConst::SIXTY_FOUR + LogSoftmaxConst::TWO * chunksize];
        LocalTensor<float> workLocal4 = workLocal1[LogSoftmaxConst::SIXTY_FOUR + 3 * chunksize];

        for (int i = start; i != end; i++) {
            int len = mmin(n - i * chunksize, chunksize);
            Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, chunksize);
            Duplicate(expsum, 0.0f, chunksize);
            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<bfloat16_t> padParams1{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams1);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<bfloat16_t>();
                Cast(workLocal2, inputLocal, RoundMode::CAST_NONE, blocklen * chunksize);
                for (int k = 0; k != blocklen; k++) {
                    Max(oldmax, oldmax, workLocal2[k * chunksize], len);
                }
                inQueue.FreeTensor(inputLocal);
            }

            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<bfloat16_t>();
                Cast(workLocal2, inputLocal, RoundMode::CAST_NONE, blocklen * chunksize);
                for (int k = 0; k != blocklen; k++) {
                    Sub(workLocal2[k * chunksize], workLocal2[k * chunksize], oldmax, len);
                    Exp(workLocal2[k * chunksize], workLocal2[k * chunksize], len);
                    Add(expsum, expsum, workLocal2[k * chunksize], len);
                }
                inQueue.FreeTensor(inputLocal);
            }

            Ln(expsum, expsum, len);

            for (int j = 0; j < m; j += LogSoftmaxConst::FOUR) {
                int blocklen = mmin(m - j, LogSoftmaxConst::FOUR);
                LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
                DataCopyExtParams copyParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
                DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
                DataCopyPad(inputLocal, inputGm[batch * m * n + j * n + i * chunksize], copyParams, padParams);
                inQueue.EnQue(inputLocal);
                inputLocal = inQueue.DeQue<bfloat16_t>();
                Cast(workLocal2, inputLocal, RoundMode::CAST_NONE, chunksize * blocklen);
                inQueue.FreeTensor(inputLocal);
                LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
                for (int k = 0; k != blocklen; k++) {
                    Sub(workLocal2[k * chunksize], workLocal2[k * chunksize], oldmax, len);
                    Sub(workLocal2[k * chunksize], workLocal2[k * chunksize], expsum, len);
                }
                Cast(outLocal, workLocal2, RoundMode::CAST_RINT, chunksize * blocklen);
                outQueue.EnQue(outLocal);
                outQueue.DeQue<bfloat16_t>();
                DataCopyExtParams copyoutParams{static_cast<uint16_t>(blocklen),
                    static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                    static_cast<uint32_t>(LogSoftmaxConst::ONE_TWENTY_EIGHT - (len + 15) / LogSoftmaxConst::SIXTEEN),
                    static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)), 0};
                DataCopyPad(outGm[batch * m * n + j * n + i * chunksize], outLocal, copyoutParams);
                outQueue.FreeTensor(outLocal);
            }
        }
    }
    __aicore__ inline void Process4(int batch, int start, int end, int chunk)
    {
        LocalTensor<float> workLocal = workQueue.Get<float>();
        LocalTensor<float> workLocal1 = workQueue1.Get<float>();
        LocalTensor<float> oldmax = workLocal;
        LocalTensor<float> expsum = workLocal[chunk];
        LocalTensor<float> workLocal2 = workLocal1;
        LocalTensor<float> workLocal3 = workLocal1[m * chunk];
        int blocklen = chunk / LogSoftmaxConst::SIXTEEN;
        for (int i = start; i != end; i++) {
            int len = mmin(n - i * chunk, chunk);
            Duplicate(oldmax, LogSoftmaxConst::FLOAT_NEG_INF, chunk);
            Duplicate(expsum, 0.0f, chunk);

            LocalTensor<bfloat16_t> inputLocal = inQueue.AllocTensor<bfloat16_t>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(m),
                static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)),
                static_cast<uint32_t>(blocklen - (len + 15) / LogSoftmaxConst::SIXTEEN), 0};
            DataCopyPadExtParams<bfloat16_t> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm[batch * m * n + i * chunk], copyParams, padParams);
            inQueue.EnQue(inputLocal);
            inputLocal = inQueue.DeQue<bfloat16_t>();

            Cast(workLocal2, inputLocal, RoundMode::CAST_NONE, m * chunk);
            inQueue.FreeTensor(inputLocal);
            for (int k = 0; k != m; k++) {
                Max(oldmax, oldmax, workLocal2[k * chunk], len);
            }
            for (int k = 0; k != m; k++) {
                Sub(workLocal2[k * chunk], workLocal2[k * chunk], oldmax, len);
            }
            Exp(workLocal3, workLocal2, chunk * m);
            for (int k = 0; k != m; k++) {
                Add(expsum, workLocal3[k * chunk], expsum, len);
            }
            Ln(expsum, expsum, chunk);
            for (int k = 0; k != m; k++) {
                Sub(workLocal2[k * chunk], workLocal2[k * chunk], expsum, len);
            }
            LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();
            Cast(outLocal, workLocal2, RoundMode::CAST_RINT, chunk * m);
            outQueue.EnQue(outLocal);
            outQueue.DeQue<bfloat16_t>();
            DataCopyExtParams copyoutParams{static_cast<uint16_t>(m),
                static_cast<uint32_t>(len * sizeof(bfloat16_t)),
                static_cast<uint32_t>(blocklen - (len + 15) / LogSoftmaxConst::SIXTEEN),
                static_cast<uint32_t>((n - len) * sizeof(bfloat16_t)), 0};
            DataCopyPad(outGm[batch * m * n + i * chunk], outLocal, copyoutParams);
            outQueue.FreeTensor(outLocal);
        }
    }

private:
    static constexpr int BUFFER_NUM = 2;
    static constexpr int chunksize = 2048;
    int m, n, coliters, coltail, rowiters, rowtail;

    TPipe pipe;

    GlobalTensor<bfloat16_t> inputGm;
    GlobalTensor<bfloat16_t> outGm;

    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<TPosition::VECCALC> workQueue;
    TBuf<TPosition::VECCALC> workQueue1;
    TQue<QuePosition::VECOUT, 1> outQueue;
};

#endif // LOGSOFTMAX_H