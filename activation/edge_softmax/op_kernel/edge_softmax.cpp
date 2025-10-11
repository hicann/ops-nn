#include <cfloat>

#include "kernel_operator.h"
using namespace AscendC;

class KernelEdgeSoftmax {
   private:
    // constants
    static constexpr int32_t DATA_BLOCK_SIZE = 32;  // 数据块大小: 32字节
    static constexpr int32_t DATA_BLOCK_LEN_32 =
        DATA_BLOCK_SIZE / sizeof(float);         // 32位数据块长度: 32字节/4字节=8个元素
    static constexpr int32_t REPEAT_SIZE = 256;  // 重复操作大小: 256字节
    static constexpr int32_t REPEAT_LEN_32 =
        REPEAT_SIZE / sizeof(float);  // 32位重复操作长度: 256字节/4字节=64个元素
    static constexpr int32_t TRANS_SIZE = 512;
    static constexpr int32_t TRANS_LEN_32 = TRANS_SIZE / sizeof(float);
    static constexpr int32_t ubSize = 192 * 1024;
    static constexpr int32_t coreNum = 40;
    // variables
    int32_t eventIdMTE3ToMTE2, eventIdVToMTE2;
    int32_t E, F, N, alignF, tileE, blockStartN, blockN;
    uint8_t stride;
    // GM buffers
    GlobalTensor<float> xGm, yGm;
    GlobalTensor<int32_t> idxGm;
    // UB buffers
    TQueBind<TPosition::GM, TPosition::VECIN, 1> inQ, idxQ;
    TQueBind<TPosition::VECOUT, TPosition::GM, 1> outQ;
    TBuf<TPosition::VECCALC> maxBuf, sumBuf;
    // functions
    __aicore__ constexpr inline int32_t CeilDiv(int32_t a, int32_t b) {
        return a == 0 ? 0 : static_cast<int32_t>(1) + (a - 1) / b;
    }
    __aicore__ constexpr inline int32_t AlignUp(int32_t a, int32_t b) { return CeilDiv(a, b) * b; }
    __aicore__ constexpr inline int32_t AlignDown(int32_t a, int32_t b) { return a / b * b; }
    __aicore__ constexpr inline int32_t Min(int32_t a, int32_t b) { return a < b ? a : b; }
    /**
     * @brief Broadcast applied in reduce.
     *
     * @param dst (16, 8), 128 floats, 512B
     * @param src (16, 8), 128 floats, 512B
     *
     * @note dst and src can overlap
     * @note DATA_BLOCK_SIZE = 32, REPEAT_SIZE = 256, TRANS_SIZE = 512
     */
    template <typename T>
    __aicore__ inline void BroadcastInReduce(const LocalTensor<T> &dst, const LocalTensor<T> &src) {
        // broadcast
        UnaryRepeatParams repeatParams{/* dstBlkStride = */ 1,
                                       /* srcBlkStride = */ 0,
                                       /* dstRepStride = */ 8,
                                       /* srcRepStride = */ 0};
        Adds(src, src, static_cast<T>(0), REPEAT_SIZE / sizeof(T), TRANS_SIZE / REPEAT_SIZE,
             repeatParams);

        // transpose
        TransDataTo5HDParams transDataParams{/* dstHighHalf = */ false,
                                             /* srcHighHalf = */ false,
                                             /* repeatTimes = */ 1,
                                             /* dstRepStride = */ 0,
                                             /* srcRepStride = */ 0};
        uint64_t dstList[16];
        uint64_t srcList[16];
        for (int i = 0; i < 16; i++) {
            dstList[i] = (uint64_t)(dst[8 * i].GetPhyAddr());
            srcList[i] = (uint64_t)(src[8 * i].GetPhyAddr());
        }
        TransDataTo5HD<T>(dstList, srcList, transDataParams);
    }

   public:
    __aicore__ inline KernelEdgeSoftmax() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR idx, GM_ADDR y, GM_ADDR tiling) {
        GET_TILING_DATA(tiling_data, tiling);
        F = static_cast<int32_t>(tiling_data.F);
        E = static_cast<int32_t>(tiling_data.E);
        N = static_cast<int32_t>(tiling_data.N);
        int32_t blockNum = static_cast<int32_t>(tiling_data.blockNum);

        alignF = AlignUp(F, REPEAT_LEN_32);
        stride = static_cast<uint8_t>(alignF / DATA_BLOCK_LEN_32);
        eventIdMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        eventIdVToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));

        int32_t bigBlocks = (blockNum != 1) ? N % blockNum : 0;
        int32_t bigBlockN = (bigBlocks == 0) ? N / blockNum : N / blockNum + 1;
        int32_t smallblockN = N / blockNum;
        blockStartN = (GetBlockIdx() < bigBlocks)
                          ? GetBlockIdx() * bigBlockN
                          : bigBlocks * bigBlockN + (GetBlockIdx() - bigBlocks) * smallblockN;
        blockN = (GetBlockIdx() < bigBlocks) ? bigBlockN : smallblockN;

        int32_t maxE = AlignDown(
            Min(ubSize / (2 * alignF + 1) / sizeof(float), MAX_REPEAT_TIMES) - 4, REPEAT_LEN_32);
        // if (GetBlockIdx() == 0) printf("maxE: %d\n", maxE);
        tileE = AlignUp(Min(maxE, E), DATA_BLOCK_LEN_32);
        // if (GetBlockIdx() == 0) printf("tileE: %d\n", tileE);

        int32_t gmE = AlignUp(E, tileE) + maxE;
        // if (GetBlockIdx() == 0) printf("gmE: %d\n", gmE);
        xGm.SetGlobalBuffer((__gm__ float *)x, gmE * alignF);
        idxGm.SetGlobalBuffer((__gm__ int32_t *)idx, gmE);
        yGm.SetGlobalBuffer((__gm__ float *)y, gmE * alignF);

        GetTPipePtr()->InitBuffer(inQ, 1, tileE * alignF * sizeof(float));
        GetTPipePtr()->InitBuffer(idxQ, 1, tileE * sizeof(int32_t));
        GetTPipePtr()->InitBuffer(outQ, 1, tileE * alignF * sizeof(float));
        GetTPipePtr()->InitBuffer(maxBuf, TRANS_LEN_32 * sizeof(float));
        GetTPipePtr()->InitBuffer(sumBuf, alignF * sizeof(float));
    }

    __aicore__ inline void process() {
        int32_t curE, curLen, alignLen, repeatTimes, remains;
        // max_{ij} = max(x_{ij})
        LocalTensor<float> maxLocal = maxBuf.Get<float>();
        Duplicate(maxLocal, -FLT_MAX, TRANS_LEN_32);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        for (int32_t e = 0; e < E; e += tileE) {
            curE = Min(tileE, E - e);
            curLen = curE * F;
            alignLen = AlignUp(curLen, DATA_BLOCK_LEN_32);
            repeatTimes = curLen / REPEAT_LEN_32;
            remains = curLen - repeatTimes * REPEAT_LEN_32;

            LocalTensor<float> xLocal = inQ.AllocTensor<float>();
            DataCopy(xLocal, xGm[e * F], alignLen);
            // if (e == 0) {
            //     uint32_t array[] = {static_cast<uint32_t>(curE), static_cast<uint32_t>(F)};
            //     ShapeInfo shapeInfo(2, array);
            //     DumpTensor(xLocal, 0, alignLen, shapeInfo);
            // }
            inQ.EnQue(xLocal);

            xLocal = inQ.DeQue<float>();
            int32_t i = 0;
            while (i < repeatTimes / MAX_REPEAT_TIMES) {
                Max(maxLocal, xLocal[i * MAX_REPEAT_TIMES * REPEAT_LEN_32], maxLocal, REPEAT_LEN_32,
                    MAX_REPEAT_TIMES, {1, 1, 1, 0, 8, 0});
                ++i;
                repeatTimes -= MAX_REPEAT_TIMES;
            }
            Max(maxLocal, xLocal[i * MAX_REPEAT_TIMES * REPEAT_LEN_32], maxLocal, REPEAT_LEN_32,
                repeatTimes, {1, 1, 1, 0, 8, 0});
            Max(maxLocal, xLocal[REPEAT_LEN_32 * repeatTimes], maxLocal, remains);
            inQ.FreeTensor(xLocal);
        }
        int32_t mask = (E * F < REPEAT_LEN_32) ? E * F : REPEAT_LEN_32;
        WholeReduceMax(maxLocal, maxLocal, mask, 1, 1, 1, 8);
        BroadcastInReduce(maxLocal, maxLocal);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
        // if (GetBlockIdx() == 0) DumpTensor(maxLocal, 0, DATA_BLOCK_LEN_32);

        // y_{ij} = exp(x_{ij} - max_{ij})
        // SyncAll();
        // if (F % DATA_BLOCK_LEN_32 != 0) {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);
        // }
        // if (F % DATA_BLOCK_LEN_32 != 0) {
        //     CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        //     CrossCoreWaitFlag(0x0);
        // }
        for (int32_t e = 0; e < E; e += tileE) {
            curE = Min(tileE, E - e);
            curLen = curE * F;
            alignLen = AlignUp(curLen, DATA_BLOCK_LEN_32);
            // printf("loop: %d, alignLen: %d, alignSize: %d\n", e / tileE, alignLen,
            //        alignLen * sizeof(float));
            // if (GetBlockIdx() == 0)
            //     printf("curE: %d, curLen: %d, alignLen: %d\n", curE, curLen, alignLen);
            repeatTimes = CeilDiv(curLen, REPEAT_LEN_32);

            LocalTensor<float> xLocal = inQ.AllocTensor<float>();
            DataCopy(xLocal, xGm[e * F], alignLen);
            inQ.EnQue(xLocal);

            LocalTensor<float> yLocal = outQ.AllocTensor<float>();
            xLocal = inQ.DeQue<float>();
            int32_t i = 0;
            while (i < repeatTimes / MAX_REPEAT_TIMES) {
                Sub(yLocal[i * MAX_REPEAT_TIMES * REPEAT_LEN_32],
                    xLocal[i * MAX_REPEAT_TIMES * REPEAT_LEN_32], maxLocal, REPEAT_LEN_32,
                    MAX_REPEAT_TIMES, {1, 1, 0, 8, 8, 0});
                ++i;
                repeatTimes -= MAX_REPEAT_TIMES;
            }
            Sub(yLocal[i * MAX_REPEAT_TIMES * REPEAT_LEN_32],
                xLocal[i * MAX_REPEAT_TIMES * REPEAT_LEN_32], maxLocal, REPEAT_LEN_32, repeatTimes,
                {1, 1, 0, 8, 8, 0});
            Exp(yLocal, yLocal, alignLen);
            outQ.EnQue(yLocal);
            inQ.FreeTensor(xLocal);

            yLocal = outQ.DeQue<float>();
            DataCopy(yGm[e * F], yLocal, alignLen);
            outQ.FreeTensor(yLocal);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);

        for (int32_t n = blockStartN; n < blockStartN + blockN; ++n) {
            // count edges
            int32_t edges = 0;
            for (int32_t e = 0; e < E; e += tileE) {
                int32_t curE = Min(tileE, E - e);
                LocalTensor<int32_t> idxLocal = idxQ.AllocTensor<int32_t>();
                DataCopy(idxLocal, idxGm[e], AlignUp(curE, DATA_BLOCK_LEN_32));
                idxQ.EnQue(idxLocal);

                idxLocal = idxQ.DeQue<int32_t>();
                for (int32_t i = 0; i < curE; ++i) {
                    int32_t idxVal = idxLocal.GetValue(i);
                    if (idxVal == n) ++edges;
                }
                idxQ.FreeTensor(idxLocal);
            }
            if (edges == 0) continue;

            // sum_i = sum_i(y_{ij})
            LocalTensor<float> sumLocal = sumBuf.Get<float>();
            for (int32_t e = 0; e < E; e += tileE) {
                int32_t curE = Min(tileE, E - e);
                LocalTensor<int32_t> idxLocal = idxQ.AllocTensor<int32_t>();
                DataCopy(idxLocal, idxGm[e], AlignUp(curE, DATA_BLOCK_LEN_32));
                idxQ.EnQue(idxLocal);

                idxLocal = idxQ.DeQue<int32_t>();
                LocalTensor<float> yLocal = inQ.AllocTensor<float>();
                int32_t edgeNum = 0;
                for (int32_t i = 0; i < curE; ++i) {
                    int32_t idxVal = idxLocal.GetValue(i);
                    if (idxVal == n)
                        DataCopy(yLocal[edgeNum++ * alignF], yGm[(e + i) * F],
                                 AlignUp(F, DATA_BLOCK_LEN_32));
                }
                inQ.EnQue(yLocal);
                idxQ.FreeTensor(idxLocal);

                yLocal = inQ.DeQue<float>();
                if (e == 0) Duplicate(sumLocal, static_cast<float>(0), alignF);
                for (int32_t i = 0; i < F / REPEAT_LEN_32; ++i) {
                    Add(sumLocal[i * REPEAT_LEN_32], yLocal[i * REPEAT_LEN_32],
                        sumLocal[i * REPEAT_LEN_32], REPEAT_LEN_32, edgeNum,
                        {1, 1, 1, 0, stride, 0});
                }
                Add(sumLocal[AlignDown(F, REPEAT_LEN_32)], yLocal[AlignDown(F, REPEAT_LEN_32)],
                    sumLocal[AlignDown(F, REPEAT_LEN_32)], F - AlignDown(F, REPEAT_LEN_32), edgeNum,
                    {1, 1, 1, 0, stride, 0});
                inQ.FreeTensor(yLocal);
            }
            // if (n == 0) {
            //     DumpTensor(sumLocal, n, alignF);
            // }

            // y_{ij} = y_{ij} / sum_i
            for (int32_t e = 0; e < E; e += tileE) {
                int32_t curE = Min(tileE, E - e);
                LocalTensor<int32_t> idxLocal = idxQ.AllocTensor<int32_t>();
                DataCopy(idxLocal, idxGm[e], AlignUp(curE, DATA_BLOCK_LEN_32));
                idxQ.EnQue(idxLocal);

                idxLocal = idxQ.DeQue<int32_t>();
                LocalTensor<float> yInLocal = inQ.AllocTensor<float>();
                int32_t edgeNum = 0;
                for (int32_t i = 0; i < curE; ++i) {
                    int32_t idxVal = idxLocal.GetValue(i);
                    if (idxVal == n)
                        DataCopy(yInLocal[edgeNum++ * alignF], yGm[(e + i) * F],
                                 AlignUp(F, DATA_BLOCK_LEN_32));
                }
                inQ.EnQue(yInLocal);
                // if (n == 0) {
                //     uint32_t array[] = {static_cast<uint32_t>(edgeNum),
                //                         static_cast<uint32_t>(alignF)};
                //     ShapeInfo shapeInfo(2, array);
                //     DumpTensor(yInLocal, n, edgeNum * alignF, shapeInfo);
                // }

                yInLocal = inQ.DeQue<float>();
                LocalTensor<float> yOutLocal = outQ.AllocTensor<float>();
                for (int32_t i = 0; i < F / REPEAT_LEN_32; ++i) {
                    Div(yInLocal[i * REPEAT_LEN_32], yInLocal[i * REPEAT_LEN_32],
                        sumLocal[i * REPEAT_LEN_32], REPEAT_LEN_32, edgeNum,
                        {1, 1, 1, stride, stride, 0});
                }
                Div(yInLocal[AlignDown(F, REPEAT_LEN_32)], yInLocal[AlignDown(F, REPEAT_LEN_32)],
                    sumLocal[AlignDown(F, REPEAT_LEN_32)], F - AlignDown(F, REPEAT_LEN_32), edgeNum,
                    {1, 1, 1, stride, stride, 0});
                Adds(yOutLocal, yInLocal, static_cast<float>(0), edgeNum * alignF);
                // if (n == 0) {
                //     uint32_t array[] = {static_cast<uint32_t>(edgeNum),
                //                         static_cast<uint32_t>(alignF)};
                //     ShapeInfo shapeInfo(2, array);
                //     DumpTensor(yOutLocal, n, edgeNum * alignF, shapeInfo);
                // }
                outQ.EnQue(yOutLocal);
                inQ.FreeTensor(yInLocal);

                yOutLocal = outQ.DeQue<float>();
                int32_t edgeIdx = 0;
                for (int32_t i = 0; i < curE; ++i) {
                    int32_t idxVal = idxLocal.GetValue(i);
                    if (idxVal == n) {
                        DataCopy(yGm[(e + i) * F], yOutLocal[edgeIdx++ * alignF],
                                 AlignUp(F, DATA_BLOCK_LEN_32));
                        PipeBarrier<PIPE_MTE3>();
                    }
                }
                outQ.FreeTensor(yOutLocal);
                idxQ.FreeTensor(idxLocal);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
            }
        }
    }
};

extern "C" __global__ __aicore__ void edge_softmax(GM_ADDR x, GM_ADDR idx, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    KernelEdgeSoftmax op;
    op.Init(x, idx, y, tiling);
    op.process();
}