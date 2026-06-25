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
 * \file add_layer_norm_quant_split_d.h
 * \brief kernel for split d axis case
 * 
 * 计算流程（分三个阶段）：
 * Phase 1: ComputeAddAndWriteWorkspace - 遍历所有 slice，计算 x = x1 + x2 + bias，累加 sum，写入 workspace
 * Phase 2: ComputeVarianceFromWorkspace - 遍历所有 slice，从 workspace 读取 x，计算 variance = sum((x-mean)^2) / N
 * Phase 3: for each slice - 计算 layernorm y = (x-mean)*rstd*gamma+beta，量化输出 y1 = quant(y)
 * 
 * 数据布局：
 * - 输入 x1, x2: shape [numFirstDim, numLastDim]，按 block 切分 first dim
 * - gamma, beta, scales, zeroPoints: shape [numLastDim]，per-last-dim 参数
 * - workspace: float 类型，存储中间结果 x，size = numFirstDim * numLastDim * sizeof(float)
 * 
 * Split-D 设计：当 D 维度 (numLastDim) 较大时，UB 无法容纳整行数据，按 slice 切分 D 维度处理
 * 
 * 同步安全原则：
 * - 每个子函数内部完成自己使用的 TQue 队列的完整生命周期（AllocTensor → EnQue → DeQue → FreeTensor）
 * - 不返回从 TQue 获取的 LocalTensor，不在函数间传递 LocalTensor 引用
 * - 子函数内部通过 TBuf::Get<float>() 自行获取所需缓冲区，TBuf::Get 返回同一块底层缓冲区句柄
 * - 函数间数据流通过 TBuf 隐式传递：前一个函数写入 TBuf 并通过 PipeBarrier 保证完成，
 *   后一个函数通过 TBuf::Get 获取同一缓冲区读取数据
 */

#ifndef ADD_LAYER_NORM_QUANT_SPLIT_D_H_
#define ADD_LAYER_NORM_QUANT_SPLIT_D_H_

#include "add_layer_norm_quant_base.h"

template <typename T, typename S, int TILING_KEY>
class KernelAddLayerNormQuantSplitD : public KernelAddLayerNormQuantBase<T, TILING_KEY> {
public:
    __aicore__ inline KernelAddLayerNormQuantSplitD(TPipe* pipe)
    {
        Ppipe = pipe;
    }


    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta, GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2,
        GM_ADDR zeroPoints1, GM_ADDR zeroPoints2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR layernormRes,
        GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, const AddLayerNormQuantV2TilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y1, y2, x);
        InitSplitDParams(tiling, layernormRes);
        InitSplitDGlobalTensors(scales1, scales2, zeroPoints1, zeroPoints2, workspace);
        InitUBBuffers();
        if (this->isPerTensor && scales1Exist) {
            GetPerTensorScaleOffset();
        }
    }

    /**
     * 主处理函数：按行遍历，每行分三个阶段处理
     * 
     * Phase 1: 计算 x = x1 + x2 + bias，累加 sum 用于计算 mean，写入 workspace
     * Phase 2: 从 workspace 读取 x，计算 variance = sum((x-mean)^2) / N，得到 rstd
     * Phase 3: 遍历每个 slice，计算 layernorm y = (x-mean)*rstd*gamma+beta，量化输出
     */
    __aicore__ inline void Process()
    {
        for (uint32_t rowIdx = 0; rowIdx < this->rowWork; rowIdx++) {
            uint32_t rowOffset = rowIdx * this->numLastDim;

            // Phase 1: 计算加法并写入 workspace，返回 mean
            float mean = ComputeAddAndWriteWorkspace(rowOffset);
            // 等待所有 MTE 写操作完成
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);

            // Phase 2: 从 workspace 读取数据，计算方差和 rstd
            float variance = ComputeVarianceFromWorkspace(rowOffset, mean);
            float rstd = 1.0f / sqrt(variance + this->eps);

            // Phase 3: 遍历每个 slice，计算 layernorm 和量化
            for (uint32_t sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
                uint32_t curSliceSize = (sliceIdx == sliceNum - 1) ? tailSliceSize : sliceSize;
                uint32_t curSliceAligned = (sliceIdx == sliceNum - 1) ? tailSliceSizeAligned : sliceSizeAligned;
                uint32_t curSliceAlignedFp32 = (sliceIdx == sliceNum - 1) ? tailSliceSizeAlignedFp32 : sliceSizeAlignedFp32;
                uint32_t colOffset = rowOffset + sliceIdx * sliceSize;
                uint32_t sliceOffset = sliceIdx * sliceSize;

                ComputeLayerNormAndQuant(colOffset, sliceOffset, curSliceSize, curSliceAligned, curSliceAlignedFp32, mean, rstd);
            }
        }
    }

    __aicore__ inline void InitSplitDParams(const AddLayerNormQuantV2TilingData* tiling, GM_ADDR layernormRes)
    {
        uint32_t scaleOffsetMode = tiling->scaleOffsetMode;
        scales1Exist = scaleOffsetMode >= 100;
        scales2Exist = scaleOffsetMode >= 200;
        isZeroPoint1Exist = scaleOffsetMode % 100 >= 10;
        isZeroPoint2Exist = scaleOffsetMode % 10 >= 1;

        resGm.SetGlobalBuffer((__gm__ T*)layernormRes + block_idx * this->gmOffset_);

        sliceSize = tiling->sliceSize;
        sliceNum = tiling->sliceNum;
        tailSliceSize = tiling->tailSliceSize;

        sliceSizeAligned = tiling->sliceSizeAligned;
        tailSliceSizeAligned = tiling->tailSliceSizeAligned;
        sliceSizeAlignedFp32 = tiling->sliceSizeAlignedFp32;
        tailSliceSizeAlignedFp32 = tiling->tailSliceSizeAlignedFp32;
    }

    __aicore__ inline void InitSplitDGlobalTensors(GM_ADDR scales1, GM_ADDR scales2,
        GM_ADDR zeroPoints1, GM_ADDR zeroPoints2, GM_ADDR workspace)
    {
        if (scales1Exist) {
            scales1Gm.SetGlobalBuffer((__gm__ S*)scales1);
        }
        if (scales2Exist) {
            scales2Gm.SetGlobalBuffer((__gm__ S*)scales2);
        }
        if (isZeroPoint1Exist) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ S*)zeroPoints1);
        }
        if (isZeroPoint2Exist) {
            zeroPoints2Gm.SetGlobalBuffer((__gm__ S*)zeroPoints2);
        }
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + block_idx * this->gmOffset_);
    }

    __aicore__ inline void GetPerTensorScaleOffset()
    {
        if constexpr (is_same<S, float>::value) {
            perTensorScale = scales1Gm.GetValue(0);
        } else if constexpr (is_same<S, half>::value) {
            perTensorScale = static_cast<float>(scales1Gm.GetValue(0));
        } else {
            perTensorScale = Cast(scales1Gm.GetValue(0));
        }

        if (isZeroPoint1Exist) {
            if constexpr (is_same<S, float>::value) {
                perTensorOffset = zeroPoints1Gm.GetValue(0);
            } else if constexpr (is_same<S, half>::value) {
                perTensorOffset = static_cast<float>(zeroPoints1Gm.GetValue(0));
            } else {
                perTensorOffset = Cast(zeroPoints1Gm.GetValue(0));
            }
        }
    }

    __aicore__ inline void InitUBBuffers()
    {
        uint32_t maxSliceAligned = (sliceSizeAligned > tailSliceSizeAligned) ? sliceSizeAligned : tailSliceSizeAligned;
        uint32_t ubFactor = maxSliceAligned;

        Ppipe->InitBuffer(inQueX, BUFFER_NUM_SPLIT_D, 2 * ubFactor * sizeof(T));
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(inQueBias, BUFFER_NUM_SPLIT_D, ubFactor * sizeof(T));
        }
        Ppipe->InitBuffer(inQueGammaBeta, BUFFER_NUM_SPLIT_D, 2 * ubFactor * sizeof(T));
        Ppipe->InitBuffer(outQueY, BUFFER_NUM_SPLIT_D, ubFactor * sizeof(int8_t));
        Ppipe->InitBuffer(outQueRes, BUFFER_NUM_SPLIT_D, ubFactor * sizeof(T));
        Ppipe->InitBuffer(xBufFp32, ubFactor * sizeof(float));
        Ppipe->InitBuffer(yBufFp32, ubFactor * sizeof(float));
        Ppipe->InitBuffer(zBufFp32, ubFactor * sizeof(float));
        Ppipe->InitBuffer(reduceBuf, ELEM_PER_REP_FP32_SPLIT_D * sizeof(float));
        Ppipe->InitBuffer(scalesBuf, ubFactor * sizeof(float));
        if (isZeroPoint1Exist || isZeroPoint2Exist) {
            Ppipe->InitBuffer(offsetBuf, ubFactor * sizeof(float));
        }
        Ppipe->InitBuffer(tensorBuf, 32);
    }

private:
    template<typename U>
    __aicore__ inline void CopyGmToUb(LocalTensor<U> dst, GlobalTensor<U> src, uint32_t size)
    {
#if __NPU_ARCH__ == 2201
        DataCopyPadExtParams<U> padParams;
        padParams.isPad = false;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = size * sizeof(U);
        DataCopyPad(dst, src, copyParams, padParams);
#else
        LocalTensor<U> tensor_local = tensorBuf.Get<U>();
        DataCopyExV2(dst, src, tensor_local, size);
#endif
    }

    template<typename U>
    __aicore__ inline void CopyUbToGm(GlobalTensor<U> dst, LocalTensor<U> src, uint32_t size)
    {
#if __NPU_ARCH__ == 2201
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = size * sizeof(U);
        DataCopyPad(dst, src, copyParams);
#else
        LocalTensor<U> tensor_local = tensorBuf.Get<U>();
        DataCopyExV2(dst, src, tensor_local, size);
#endif
    }

    /**
     * 搬运 x1, x2 到 UB，转换为 float，计算 x = x1 + x2
     * 
     * 同步安全：
     * - inQueX 的完整生命周期（Alloc → EnQue → DeQue → Free）在本函数内完成
     * - 计算结果写入 TBuf xBufFp32/yBufFp32，不返回任何 LocalTensor
     * - 调用方通过 TBuf::Get 获取同一缓冲区读取结果
     * 
     * @param colOffset: 当前 slice 在 GM 中的偏移
     * @param curSliceSize: 当前 slice 的元素个数
     * @param curSliceAligned: 对齐后的元素个数
     */
    __aicore__ inline void CopyInX1X2AndAdd(uint32_t colOffset, uint32_t curSliceSize, uint32_t curSliceAligned)
    {
        LocalTensor<T> x1x2Local = inQueX.AllocTensor<T>();
        CopyGmToUb<T>(x1x2Local, this->x1Gm[colOffset], curSliceSize);
        CopyGmToUb<T>(x1x2Local[curSliceAligned], this->x2Gm[colOffset], curSliceSize);
        inQueX.EnQue(x1x2Local);
        LocalTensor<T> x1x2Deq = inQueX.DeQue<T>();

        // 自行获取 TBuf 缓冲区，不通过参数传递
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yFp32 = yBufFp32.Get<float>();
        CastToFloat<T>(xFp32, x1x2Local, curSliceSize);
        CastToFloat<T>(yFp32, x1x2Local[curSliceAligned], curSliceSize);
        PipeBarrier<PIPE_V>();
        Add(xFp32, xFp32, yFp32, curSliceSize);
        PipeBarrier<PIPE_V>();

        // Cast 完成后 x1x2 数据已写入 TBuf，可安全释放队列 tensor
        inQueX.FreeTensor(x1x2Deq);
    }

    /**
     * 处理 elewise bias: x = x + bias
     * 
     * 同步安全：
     * - inQueBias 的完整生命周期在本函数内完成
     * - 从 TBuf xBufFp32 读取 x 并写回，不通过参数传递 LocalTensor
     */
    __aicore__ inline void AddElewiseBias(uint32_t colOffset, uint32_t curSliceSize)
    {
        LocalTensor<T> biasLocal = inQueBias.AllocTensor<T>();
        CopyGmToUb<T>(biasLocal, this->biasGm[colOffset], curSliceSize);
        inQueBias.EnQue(biasLocal);
        LocalTensor<T> biasDeq = inQueBias.DeQue<T>();

        // 自行获取 TBuf 缓冲区
        LocalTensor<float> biasFp32 = zBufFp32.Get<float>();
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        CastToFloat<T>(biasFp32, biasDeq, curSliceSize);
        PipeBarrier<PIPE_V>();
        Add(xFp32, xFp32, biasFp32, curSliceSize);
        PipeBarrier<PIPE_V>();
        inQueBias.FreeTensor(biasDeq);
    }

    /**
     * 处理 broadcast bias: x = x + bias
     * 
     * 同步安全：
     * - inQueBias 的完整生命周期在本函数内完成
     * - 从 TBuf xBufFp32 读取 x 并写回，不通过参数传递 LocalTensor
     */
    __aicore__ inline void AddBroadcastBias(uint32_t sliceOffset, uint32_t curSliceSize)
    {
        LocalTensor<T> biasLocal = inQueBias.AllocTensor<T>();
        CopyGmToUb<T>(biasLocal, this->biasGm[sliceOffset], curSliceSize);
        inQueBias.EnQue(biasLocal);
        LocalTensor<T> biasDeq = inQueBias.DeQue<T>();

        // 自行获取 TBuf 缓冲区
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<float> biasFp32 = zBufFp32.Get<float>();
        CastToFloat<T>(biasFp32, biasDeq, curSliceSize);
        PipeBarrier<PIPE_V>();
        Add(xFp32, xFp32, biasFp32, curSliceSize);
        PipeBarrier<PIPE_V>();
        inQueBias.FreeTensor(biasDeq);
    }

    /**
     * 输出 xOut 到 GM
     * 
     * 同步安全：
     * - outQueRes 的完整生命周期在本函数内完成
     * - 从 TBuf xBufFp32 读取数据，不通过参数传递 LocalTensor
     */
    __aicore__ inline void CopyOutX(uint32_t colOffset, uint32_t curSliceSize)
    {
        // 自行获取 TBuf 缓冲区
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<T> xOutLocal = outQueRes.AllocTensor<T>();
        if constexpr (is_same<T, float>::value) {
            Adds(xOutLocal, xFp32, 0.0f, curSliceSize);
        } else if constexpr (is_same<T, half>::value) {
            Cast(xOutLocal, xFp32, RoundMode::CAST_NONE, curSliceSize);
        } else {
            Cast(xOutLocal, xFp32, RoundMode::CAST_RINT, curSliceSize);
        }
        PipeBarrier<PIPE_V>();
        outQueRes.EnQue(xOutLocal);
        
        LocalTensor<T> xOutDeq = outQueRes.DeQue<T>();
        CopyUbToGm<T>(this->xGm[colOffset], xOutDeq, curSliceSize);
        outQueRes.FreeTensor(xOutDeq);
    }

    /**
     * Phase 1: 计算 x = x1 + x2 + bias，累加 sum，写入 workspace
     * 
     * 遍历所有 slice，对每个 slice：
     * 1. 从 GM 搬运 x1, x2 到 UB
     * 2. 转换为 float，计算 x = x1 + x2
     * 3. 如果有 bias，x = x + bias
     * 4. 如果 isXOut，输出 x 到 GM
     * 5. 累加 sum += sum(x)
     * 6. 将 x 写入 workspace
     * 
     * @return mean = sum / numLastDim
     */
    __aicore__ inline float ComputeAddAndWriteWorkspace(uint32_t rowOffset)
    {
        LocalTensor<float> sumTensor = reduceBuf.Get<float>();
        Duplicate(sumTensor, 0.0f, 1);

        for (uint32_t sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
            uint32_t curSliceSize = (sliceIdx == sliceNum - 1) ? tailSliceSize : sliceSize;
            uint32_t curSliceAligned = (sliceIdx == sliceNum - 1) ? tailSliceSizeAligned : sliceSizeAligned;
            uint32_t colOffset = rowOffset + sliceIdx * sliceSize;
            uint32_t sliceOffset = sliceIdx * sliceSize;

            // 搬运 x1, x2 并计算加法，结果写入 TBuf xBufFp32
            CopyInX1X2AndAdd(colOffset, curSliceSize, curSliceAligned);
            event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

            // 处理 bias，结果写回 TBuf xBufFp32
            if constexpr (IS_BIAS_ELEWISE) {
                AddElewiseBias(colOffset, curSliceSize);
            }
            if constexpr (IS_BIAS_BROADCAST) {
                AddBroadcastBias(sliceOffset, curSliceSize);
            }
            
            // 输出 xOut
            if (this->isXOut) {
                event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventVMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
                CopyOutX(colOffset, curSliceSize);
            }

            AccumulateSumAndWriteWorkspace(colOffset, curSliceSize, sumTensor);
        }
        return sumTensor.GetValue(0) * this->aveNum;
    }

    __aicore__ inline void AccumulateSumAndWriteWorkspace(uint32_t colOffset, uint32_t curSliceSize,
        LocalTensor<float>& sumTensor)
    {
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yFp32 = yBufFp32.Get<float>();
        LocalTensor<float> tmpReduce = yBufFp32.Get<float>();
        ReduceSum(tmpReduce, xFp32, yFp32, curSliceSize);
        PipeBarrier<PIPE_V>();
        Add(sumTensor, sumTensor, tmpReduce, 1);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        CopyUbToGm<float>(workspaceGm[colOffset], xFp32, curSliceSize);
        event_t eventMTE3V2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V2);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V2);
    }

    /**
     * Phase 2: 从 workspace 读取 x，计算 variance = sum((x-mean)^2) / N
     * 
     * 遍历所有 slice，对每个 slice：
     * 1. 从 workspace 读取 x 到 UB
     * 2. 计算 z = x - mean
     * 3. 计算 y = z * z
     * 4. 累加 ssd += sum(y)
     * 
     * @return variance = ssd / numLastDim
     */
    __aicore__ inline float ComputeVarianceFromWorkspace(uint32_t rowOffset, float mean)
    {
        LocalTensor<float> ssdTensor = reduceBuf.Get<float>();
        Duplicate(ssdTensor, 0.0f, 1);
        LocalTensor<float> tmpReduce = yBufFp32.Get<float>();

        for (uint32_t sliceIdx = 0; sliceIdx < sliceNum; sliceIdx++) {
            uint32_t curSliceSize = (sliceIdx == sliceNum - 1) ? tailSliceSize : sliceSize;
            uint32_t colOffset = rowOffset + sliceIdx * sliceSize;

            LocalTensor<float> xFp32 = xBufFp32.Get<float>();
            LocalTensor<float> yFp32 = yBufFp32.Get<float>();
            LocalTensor<float> zFp32 = zBufFp32.Get<float>();

            CopyGmToUb<float>(xFp32, workspaceGm[colOffset], curSliceSize);
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            Adds(zFp32, xFp32, -mean, curSliceSize);
            PipeBarrier<PIPE_V>();
            Mul(yFp32, zFp32, zFp32, curSliceSize);
            PipeBarrier<PIPE_V>();

            ReduceSum(tmpReduce, yFp32, zFp32, curSliceSize);
            PipeBarrier<PIPE_V>();
            Add(ssdTensor, ssdTensor, tmpReduce, 1);
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventSV);
            WaitFlag<HardEvent::V_S>(eventSV);
        }
        return ssdTensor.GetValue(0) * this->aveNum;
    }

    /**
     * 搬运 gamma/beta 到 UB，从 workspace 读取 x，计算 layernorm
     * 
     * 同步安全：
     * - inQueGammaBeta 的完整生命周期在本函数内完成
     * - 计算结果写入 TBuf xBufFp32，不返回任何 LocalTensor
     * - 调用方通过 TBuf::Get 获取同一缓冲区读取结果
     * 
     * @param colOffset: 当前 slice 在 GM 中的偏移
     * @param sliceOffset: 当前 slice 在 gamma/beta 中的偏移
     * @param size: 当前 slice 的元素个数
     * @param alignedSize: 对齐后的元素个数（T 类型）
     * @param mean: 均值
     * @param rstd: 标准差的倒数
     */
    __aicore__ inline void CopyInGammaBetaAndComputeLN(uint32_t colOffset, uint32_t sliceOffset, uint32_t size,
        uint32_t alignedSize, float mean, float rstd)
    {
        // 搬运 gamma 和 beta 到 UB
        // gamma 存储在 gammaBetaLocal[0:alignedSize]
        // beta 存储在 gammaBetaLocal[alignedSize:2*alignedSize]
        LocalTensor<T> gammaBetaLocal = inQueGammaBeta.AllocTensor<T>();
        CopyGmToUb<T>(gammaBetaLocal, this->gammaGm[sliceOffset], size);
        CopyGmToUb<T>(gammaBetaLocal[alignedSize], this->betaGm[sliceOffset], size);
        inQueGammaBeta.EnQue(gammaBetaLocal);
        LocalTensor<T> gammaBetaDeq = inQueGammaBeta.DeQue<T>();

        // 自行获取 TBuf 缓冲区，不通过参数传递
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yFp32 = yBufFp32.Get<float>();

        CopyGmToUb<float>(xFp32, workspaceGm[colOffset], size);
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

        // 计算 x = (x - mean) * rstd
        Adds(xFp32, xFp32, -mean, size);
        PipeBarrier<PIPE_V>();
        Muls(xFp32, xFp32, rstd, size);
        PipeBarrier<PIPE_V>();

        // 计算 x = x * gamma + beta
        CastToFloat<T>(yFp32, gammaBetaDeq, size);
        PipeBarrier<PIPE_V>();
        Mul(xFp32, xFp32, yFp32, size);
        PipeBarrier<PIPE_V>();
        CastToFloat<T>(yFp32, gammaBetaDeq[alignedSize], size);
        PipeBarrier<PIPE_V>();
        Add(xFp32, xFp32, yFp32, size);
        PipeBarrier<PIPE_V>();

        // layernorm 计算完成，gamma/beta 数据已使用完毕，可安全释放队列 tensor
        inQueGammaBeta.FreeTensor(gammaBetaDeq);
    }

    /**
     * 输出 layernormRes 到 GM
     * 
     * 同步安全：
     * - outQueRes 的完整生命周期在本函数内完成
     * - 从 TBuf xBufFp32 读取数据，不通过参数传递 LocalTensor
     */
    __aicore__ inline void CopyOutLayernormRes(uint32_t colOffset, uint32_t size)
    {
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<T> resLocal = outQueRes.AllocTensor<T>();

        if constexpr (is_same<T, float>::value) {
            Adds(resLocal, xFp32, 0.0f, size);
        } else if constexpr (is_same<T, half>::value) {
            Cast(resLocal, xFp32, RoundMode::CAST_NONE, size);
        } else {
            Cast(resLocal, xFp32, RoundMode::CAST_RINT, size);
        }
        PipeBarrier<PIPE_V>();
        outQueRes.EnQue(resLocal);
        LocalTensor<T> resDeq = outQueRes.DeQue<T>();

        CopyUbToGm<T>(resGm[colOffset], resDeq, size);
        outQueRes.FreeTensor(resDeq);
    }

    __aicore__ inline void CopyInScalesAndZeroPoints(uint32_t sliceOffset, uint32_t size)
    {
        if (scales1Exist) {
            if (this->isPerTensor) {
                return;
            }
            LocalTensor<float> scalesLocal = scalesBuf.Get<float>();
            LocalTensor<S> scalesTmp = inQueX.AllocTensor<S>();
            CopyGmToUb<S>(scalesTmp, scales1Gm[sliceOffset], size);
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

            CastToFloat<S>(scalesLocal, scalesTmp, size);
            PipeBarrier<PIPE_V>();
            inQueX.FreeTensor(scalesTmp);

            if (isZeroPoint1Exist) {
                LocalTensor<float> offsetLocal = offsetBuf.Get<float>();
                LocalTensor<S> offsetTmp = inQueX.AllocTensor<S>();
                CopyGmToUb<S>(offsetTmp, zeroPoints1Gm[sliceOffset], size);
                event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
                CastToFloat<S>(offsetLocal, offsetTmp, size);
                PipeBarrier<PIPE_V>();
                inQueX.FreeTensor(offsetTmp);
            }
        }
    }

    __aicore__ inline void QuantAndOut(uint32_t colOffset, uint32_t size)
    {
        LocalTensor<float> xFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yFp32 = yBufFp32.Get<float>();
        LocalTensor<int8_t> yLocal = outQueY.AllocTensor<int8_t>();

        if (scales1Exist) {
            if (this->isPerTensor) {
                Muls(xFp32, xFp32, perTensorScale, size);
            } else {
                LocalTensor<float> scalesLocal = scalesBuf.Get<float>();
                Mul(xFp32, xFp32, scalesLocal, size);
            }
            PipeBarrier<PIPE_V>();
            if (isZeroPoint1Exist) {
                if (this->isPerTensor) {
                    Adds(xFp32, xFp32, perTensorOffset, size);
                } else {
                    LocalTensor<float> offsetLocal = offsetBuf.Get<float>();
                    Add(xFp32, xFp32, offsetLocal, size);
                }
                PipeBarrier<PIPE_V>();
            }
        }

        Cast(xFp32.ReinterpretCast<int32_t>(), xFp32, RoundMode::CAST_RINT, size);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.0f);
        PipeBarrier<PIPE_V>();
        Cast(yFp32.ReinterpretCast<half>(), xFp32.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();

        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);

        Cast(yLocal, yFp32.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
        PipeBarrier<PIPE_V>();

        outQueY.EnQue(yLocal);
        LocalTensor<int8_t> yDeq = outQueY.DeQue<int8_t>();
        CopyUbToGm<int8_t>(this->y1Gm[colOffset], yDeq, size);
        PipeBarrier<PIPE_MTE3>();
        outQueY.FreeTensor(yDeq);
    }

    /**
     * Phase 3: 计算 layernorm 和量化（每个 slice）
     * 
     * 1. 搬运 gamma/beta，从 workspace 读取 x，计算 layernorm
     * 2. 输出 layernormRes
     * 3. 搬运 scales/zeroPoints，量化，输出量化结果
     * 
     * 各子函数均自包含队列生命周期，通过 TBuf 隐式传递中间结果
     */
    __aicore__ inline void ComputeLayerNormAndQuant(uint32_t colOffset, uint32_t sliceOffset, uint32_t size, uint32_t alignedSize,
        uint32_t alignedSizeFp32, float mean, float rstd)
    {
        // 搬运 gamma/beta 并计算 layernorm，结果写入 TBuf xBufFp32
        CopyInGammaBetaAndComputeLN(colOffset, sliceOffset, size, alignedSize, mean, rstd);

        // 输出 layernormRes，从 TBuf xBufFp32 读取
        CopyOutLayernormRes(colOffset, size);

        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);

        // 搬运 scales/zeroPoints，量化，输出量化结果，从 TBuf xBufFp32/yBufFp32 读取
        CopyInScalesAndZeroPoints(sliceOffset, size);
        QuantAndOut(colOffset, size);
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM_SPLIT_D> inQueX;
    TQue<QuePosition::VECIN, BUFFER_NUM_SPLIT_D> inQueGammaBeta;
    TQue<QuePosition::VECIN, BUFFER_NUM_SPLIT_D> inQueBias;
    TQue<QuePosition::VECOUT, BUFFER_NUM_SPLIT_D> outQueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM_SPLIT_D> outQueRes;
    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> zBufFp32;
    TBuf<TPosition::VECCALC> reduceBuf;
    TBuf<TPosition::VECCALC> scalesBuf;
    TBuf<TPosition::VECCALC> offsetBuf;
    TBuf<TPosition::VECCALC> tensorBuf;

    GlobalTensor<T> resGm;
    GlobalTensor<S> scales1Gm, scales2Gm, zeroPoints1Gm, zeroPoints2Gm;
    GlobalTensor<float> workspaceGm;

    uint32_t sliceSize;
    uint32_t sliceNum;
    uint32_t tailSliceSize;
    uint32_t sliceSizeAligned;
    uint32_t tailSliceSizeAligned;
    uint32_t sliceSizeAlignedFp32;
    uint32_t tailSliceSizeAlignedFp32;

    bool scales1Exist = false;
    bool scales2Exist = false;
    bool isZeroPoint1Exist = false;
    bool isZeroPoint2Exist = false;

    float perTensorScale = 1.0f;
    float perTensorOffset = 0.0f;
};

#endif // ADD_LAYER_NORM_QUANT_SPLIT_D_H_
