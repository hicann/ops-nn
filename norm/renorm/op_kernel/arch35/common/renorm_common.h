/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_COMMON_H_
#define _RENORM_COMMON_H_

#include "kernel_operator.h"

/*
 * ===================== renorm 算子公共工具函数 =====================
 *
 * 【renorm 算子原理】
 *   renorm 沿 dim 维度切分子张量，对每个子张量计算 p-范数(norm)，
 *   若 norm 超过 maxNorm 则将整个子张量按 maxNorm/norm 缩放，否则保持不变。
 *
 * 【GM 布局】
 *   输入/输出 GM 按 [numBlocks, sliceCount, blockSize] 排列：
 *     - sliceCount = shape[dim]，即子张量个数（沿 dim 切分得到）
 *     - blockSize  = prod(shape[dim+1:])，每个连续段的大小
 *     - numBlocks  = prod(shape[:dim])，块数（即归约轴长度）
 *   每个子张量由 numBlocks 个 blockSize 大小的连续段组成，
 *   这些连续段在 GM 中并非物理连续，中间隔着其他子张量的段。
 *
 * 【数据流（核心三阶段）】
 *   阶段1 计算范数：GM --(MTE2搬运)--> UB --(V:Abs/Log/Exp/ReduceSum)--> 标量 norm
 *   阶段2 计算缩放：标量 scale = maxNorm/max(norm,eps) if norm>maxNorm else 1.0
 *   阶段3 应用缩放：GM --(MTE2)--> UB --(V:Cast->Muls->Cast)--> UB --(MTE3)--> GM
 *   不同流水线异步执行，需用 SetFlag/WaitFlag 做跨流水线同步。
 *
 * 【本文件提供的工具函数】
 *   - AlignUpFp32           : 将元素数向上对齐到 FP32_ALIGN(8)，满足 Vector API 对齐要求
 *   - LoadChunkAndCastToFP32: 从 GM 搬运数据到 UB 并 Cast 到 FP32（范数计算/缩放阶段共用）
 *   - ComputeScale          : 根据范数和 maxNorm 计算缩放因子
 *   - CastBackToDtype       : 将 FP32 结果转回原始 dtype，准备写回 GM
 *   - ScaleAndStoreSlice    : 对整个子张量乘以 scale 后写回 GM（阶段3）
 *   - StoreZerosSlice       : maxNorm=0 时向 GM 写全零（无需计算范数）
 *   - ScalarSqrt/Pow/Log/Exp: 标量数学运算（因 Ascend C 无标量版，通过 UB buffer 中转）
 *
 * 【流水线同步概念（小白必读）】
 *   Ascend C 有多条硬件流水线，它们异步并行执行：
 *     MTE2: 外部存储(GM) → 片上缓存(UB) 的数据搬运
 *     V   : 向量计算（Abs/Mul/Add/ReduceSum/Sqrt/Log/Exp 等）
 *     MTE3: 片上缓存(UB) → 外部存储(GM) 的数据搬运
 *     S   : 标量计算（SetValue/GetValue/加减乘除等）
 *   因为异步，所以一条流水线写完 UB 后，另一条流水线要读同一块 UB 时，
 *   必须用 SetFlag/WaitFlag 显式同步，否则可能读到旧数据或未完成的数据。
 *   例如：V 计算写入了 UB 的某 buffer，MTE3 要把这个 buffer 搬到 GM，
 *   就需要 V→MTE3 同步（等 V 写完，MTE3 才能开始搬）。
 *   PipeBarrier<PIPE_V>() 则保证同一流水线(V)内前序指令全部完成后再继续。
 */

namespace NsRenorm {

using namespace AscendC;

// FP32 对齐元素数（32 字节 / 4 字节 = 8）
constexpr int64_t FP32_ALIGN = 8;

// 将 chunkLen 向上对齐到 FP32_ALIGN
__aicore__ inline int64_t AlignUpFp32(int64_t count) { return (count + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN; }

// 从 GM 加载 chunk 到 dataBuf，并 Cast 到 FP32 workBuf
// dataBuf: 输入 dtype 的 LocalTensor
// workBuf: FP32 的 LocalTensor
// 返回对齐后的元素数（用于后续 Vector API 调用）
//
// 【数据流】GM --(MTE2: DataCopyPad)--> dataBuf(原始dtype) --(V: Cast/DataCopy)--> workBuf(FP32)
// 本函数在范数计算阶段和应用缩放阶段都会被循环调用，每次处理一个 chunk（连续段）。
template <typename D_T_X>
__aicore__ inline int64_t LoadChunkAndCastToFP32(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                                 GlobalTensor<D_T_X>& inputGM, int64_t gmOffset, int64_t chunkLen)
{
    // V→MTE2 同步（确保上一次 V 读 dataBuf 完成，防止循环间数据覆盖）
    // 为什么需要：本函数在循环中被反复调用，上一轮 V 流水线可能还在读 dataBuf，
    // 而这一轮 MTE2(DataCopyPad) 又要往 dataBuf 写入新数据。若不加同步，
    // MTE2 会提前覆写 dataBuf，导致上一轮 V 读到错误数据。
    // 机制：SetFlag 打一个事件标记，WaitFlag 阻塞等待该标记对应的硬件操作完成。
    TEventID eventIDV = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
    SetFlag<HardEvent::V_MTE2>(eventIDV);
    WaitFlag<HardEvent::V_MTE2>(eventIDV);

    // DataCopyPad 从 GM 加载到 dataBuf，同时 padding 到 32 字节对齐
    // 为什么需要 padding：Vector API（如 Abs/Cast/ReduceSum）要求操作数地址 32 字节对齐，
    // 但实际的 chunkLen 可能不是 8 的倍数（FP32 下 8 个元素=32 字节）。
    // DataCopyPad 会在搬运时自动在末尾补零(rightPad)，使 dataBuf 中的数据达到对齐要求，
    // 后续 Vector API 就可以安全地操作 alignedLen 个元素（补零部分不影响求和/求最大值等归约结果）。
    int64_t alignedLen = AlignUpFp32(chunkLen);
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(chunkLen * sizeof(D_T_X));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    uint8_t rightPad = static_cast<uint8_t>(alignedLen - chunkLen);
    DataCopyPadExtParams<D_T_X> padParams = {true, 0, rightPad, 0};
    DataCopyPad(dataBuf, inputGM[gmOffset], copyParams, padParams);

    // MTE2 → Vector 同步
    TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(eventID0);
    WaitFlag<HardEvent::MTE2_V>(eventID0);

    // Cast 到 FP32（FP32→FP32 时 Cast 会被编译器优化掉，需用 DataCopy）
    // 为什么 FP32→FP32 用 DataCopy 而非 Cast：当输入本身就是 float 时，Cast 操作是恒等的，
    // 编译器会将其优化掉（不生成任何指令），导致 workBuf 不会被填充。
    // 此时改用 DataCopy 做一次显式的 UB 内拷贝，确保 workBuf 拿到数据。
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        DataCopy(workBuf, dataBuf, static_cast<int32_t>(alignedLen));
    } else {
        Cast(workBuf, dataBuf, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
    }
    PipeBarrier<PIPE_V>();

    return alignedLen;
}

// 计算缩放因子
// scale = maxNorm / max(norm, eps) if norm > maxNorm else 1.0
//
// 【含义解释】
//   - 范数没超标（norm <= maxNorm）：scale = 1.0，不缩放，子张量原样输出。
//   - 范数超标（norm > maxNorm）：scale = maxNorm / norm，将子张量等比缩小，
//     缩小后新范数 = norm * (maxNorm/norm) = maxNorm，恰好达标。
//   - 分母用 max(norm, eps) 而非直接用 norm：防止 norm 极小（接近0）时除法溢出，
//     此时 scale = maxNorm/eps，是一个较大的有限值而非 inf。
__aicore__ inline float ComputeScale(float norm, float maxNorm, float eps)
{
    if (norm > maxNorm) {
        float denom = (norm > eps) ? norm : eps;
        return maxNorm / denom;
    }
    return 1.0f;
}

// 将 FP32 workBuf 转换回 D_T_X dtype 的 dataBuf
// 这是阶段3（应用缩放）中的逆向 Cast：FP32 计算完后需转回原始精度才能写回 GM。
// FP32→FP32 同样用 DataCopy（原因同 LoadChunkAndCastToFP32：Cast 会被编译器优化掉）。
// 非 FP32 dtype 使用 Cast + CAST_RINT 舍入模式：CAST_RINT = 四舍五入到最近偶数(round to nearest even)，
// 这是 FP32→低精度(如FP16/BF16)时最精确的舍入方式，能最小化精度损失。
template <typename D_T_X>
__aicore__ inline void CastBackToDtype(LocalTensor<D_T_X>& dataBuf, LocalTensor<float>& workBuf, int64_t alignedLen)
{
    // FP32→FP32 时 Cast 会被编译器优化掉，需用 DataCopy
    if constexpr (sizeof(D_T_X) == sizeof(float)) {
        DataCopy(dataBuf, workBuf, static_cast<int32_t>(alignedLen));
    } else {
        Cast(dataBuf, workBuf, RoundMode::CAST_RINT, static_cast<int32_t>(alignedLen));
    }
}

// 对一个子张量的所有元素乘以 scale 后写入 outputGM
//
// 【这是阶段3：应用缩放因子】
// 数据流（每个 chunk）：
//   GM(inputGM) --(MTE2: LoadChunkAndCastToFP32)--> UB(workBuf, FP32)
//   --> (V: Muls 乘以 scale) --> (V: CastBackToDtype 转回原dtype, dataBuf)
//   --> (MTE3: DataCopyPad) --> GM(outputGM)
//
// 【循环逻辑】
//   外层循环 b 遍历 numBlocks 个块（归约轴方向），每个块对应 GM 中一段连续区域。
//   内层 while 循环将 blockSize 大小的连续段切分为多个 tileLength 大小的 chunk 逐个处理，
//   因为 UB 容量有限，无法一次装下整个连续段。chunkStart 记录当前 chunk 在段内的偏移。
template <typename D_T_X>
__aicore__ inline void ScaleAndStoreSlice(LocalTensor<float>& workBuf, LocalTensor<D_T_X>& dataBuf,
                                          GlobalTensor<D_T_X>& inputGM, GlobalTensor<D_T_X>& outputGM, int64_t sliceIdx,
                                          int64_t blockSize, int64_t numBlocks, int64_t sliceCount, int64_t tileLength,
                                          float scale)
{
    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;
        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;

            // 加载输入并 Cast 到 FP32
            int64_t alignedLen = LoadChunkAndCastToFP32(workBuf, dataBuf, inputGM, gmOffset + chunkStart, chunkLen);

            // 乘以 scale
            Muls(workBuf, workBuf, scale, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();

            // Cast 回原 dtype
            CastBackToDtype<D_T_X>(dataBuf, workBuf, alignedLen);

            // V→MTE3 同步
            // 为什么需要：上面的 Cast/Muls 等 V 指令刚把结果写入 dataBuf(UB)，
            // MTE3(DataCopyPad) 要从 dataBuf 读数据搬到 GM。必须等 V 写完，MTE3 才能开始读。
            TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);

            // 写入 outputGM
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(chunkLen * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(outputGM[gmOffset + chunkStart], dataBuf, copyParams);

            // MTE3→MTE2 同步
            // 为什么需要：MTE3 刚从 dataBuf 读数据搬到 GM，下一轮循环的 MTE2(LoadChunkAndCastToFP32)
            // 又要往 dataBuf 写新数据。必须等 MTE3 读完，MTE2 才能覆写 dataBuf。
            TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(eventID2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID2);

            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }
}

// 对一个子张量的所有位置写入零到 outputGM（maxNorm=0 场景）
//
// 【为什么 maxNorm=0 时输出全零】
//   maxNorm=0 意味着目标范数为 0，无论原始范数多大，scale = 0/norm = 0，
//   所以子张量每个元素乘以 0 后全部变为 0。此时无需计算范数，直接填零即可。
// 数据流：Duplicate(在UB中填零) --(MTE3: DataCopyPad)--> GM(outputGM)
template <typename D_T_X>
__aicore__ inline void StoreZerosSlice(LocalTensor<D_T_X>& dataBuf, GlobalTensor<D_T_X>& outputGM, int64_t sliceIdx,
                                       int64_t blockSize, int64_t numBlocks, int64_t sliceCount, int64_t tileLength)
{
    for (int64_t b = 0; b < numBlocks; ++b) {
        int64_t gmOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
        int64_t remaining = blockSize;
        int64_t chunkStart = 0;
        while (remaining > 0) {
            int64_t chunkLen = (remaining > tileLength) ? tileLength : remaining;
            int64_t alignedLen = AlignUpFp32(chunkLen);

            Duplicate(dataBuf, static_cast<D_T_X>(0), static_cast<int32_t>(alignedLen));

            TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(chunkLen * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(outputGM[gmOffset + chunkStart], dataBuf, copyParams);

            TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
            SetFlag<HardEvent::MTE3_V>(eventID2);
            WaitFlag<HardEvent::MTE3_V>(eventID2);

            remaining -= chunkLen;
            chunkStart += chunkLen;
        }
    }
}

// 标量 Sqrt
//
// 【为什么标量运算要这么麻烦】
//   Ascend C 的 Sqrt/Log/Exp 等数学函数只有"向量指令"版本（操作整个 LocalTensor），
//   没有直接对 float 标量运算的版本。因此要计算一个标量的 sqrt，必须：
//     1. 先把标量写入 UB buffer（tmpBuf）—— 用 SetValue（S 流水线）
//     2. 再用 V 流水线的 Sqrt 向量指令对这个 buffer 做计算
//     3. 最后从 buffer 读回标量结果 —— 用 GetValue（S 流水线）
//   整个流程：S(写标量) → V(Sqrt) → S(读标量)，中间需要 S_V 和 V_S 同步。
//
// 【同步含义】
//   S_V 同步：S 流水线(SetValue)写完 tmpBuf 后，V 流水线(Sqrt)才能读 —— 等 S 完成再启动 V。
//   V_S 同步：V 流水线(Sqrt)写完 tmpBuf 后，S 流水线(GetValue)才能读 —— 等 V 完成再启动 S。
//   不同流水线异步执行，必须显式同步防止数据竞争。
__aicore__ inline float ScalarSqrt(LocalTensor<float>& tmpBuf, float val)
{
    // 第1步：先清零 tmpBuf（V 流水线的 Duplicate 指令），再写入标量值
    Duplicate(tmpBuf, 0.0f, static_cast<int32_t>(FP32_ALIGN));
    {
        // V_S 同步：等 Duplicate(V) 写完 tmpBuf，S 流水线的 SetValue 才能写
        TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID0);
        WaitFlag<HardEvent::V_S>(eventID0);
    }
    // 第2步：S 流水线将标量 val 写入 tmpBuf[0]
    tmpBuf.SetValue(0, val);
    {
        // S_V 同步：等 SetValue(S) 写完 tmpBuf，V 流水线的 Sqrt 才能读
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    // 第3步：V 流水线执行 Sqrt 向量指令（虽然只用 tmpBuf[0]，但向量指令会处理整个 FP32_ALIGN 长度）
    Sqrt(tmpBuf, tmpBuf, static_cast<int32_t>(FP32_ALIGN));
    {
        // V_S 同步：等 Sqrt(V) 写完 tmpBuf，S 流水线的 GetValue 才能读
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    // 第4步：S 流水线从 tmpBuf[0] 读回标量结果
    return tmpBuf.GetValue(0);
}

// 标量 Pow(val, exponent) = exp(exponent * log(val))
// 同样走"写入UB→V指令计算→读回标量"的流程（详见 ScalarSqrt 的说明）。
// 计算链：Maxs(防log(0)) → Log → Muls(乘exponent) → Exp，等价于 val^exponent。
// 用 Maxs(val, eps) 防止 val=0 时 log(0)=-inf 导致后续结果异常。
__aicore__ inline float ScalarPow(LocalTensor<float>& tmpBuf, float val, float exponent, float eps)
{
    Duplicate(tmpBuf, 0.0f, static_cast<int32_t>(FP32_ALIGN));
    {
        TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID0);
        WaitFlag<HardEvent::V_S>(eventID0);
    }
    tmpBuf.SetValue(0, val);
    {
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    Maxs(tmpBuf, tmpBuf, eps, static_cast<int32_t>(FP32_ALIGN));
    PipeBarrier<PIPE_V>();
    Log(tmpBuf, tmpBuf, static_cast<int32_t>(FP32_ALIGN));
    PipeBarrier<PIPE_V>();
    Muls(tmpBuf, tmpBuf, exponent, static_cast<int32_t>(FP32_ALIGN));
    PipeBarrier<PIPE_V>();
    Exp(tmpBuf, tmpBuf, static_cast<int32_t>(FP32_ALIGN));
    {
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    return tmpBuf.GetValue(0);
}

// 标量 Log
// 同样走"写入UB→V指令计算→读回标量"的流程（详见 ScalarSqrt 的说明），V 指令为 Log。
__aicore__ inline float ScalarLog(LocalTensor<float>& tmpBuf, float val)
{
    Duplicate(tmpBuf, 0.0f, static_cast<int32_t>(FP32_ALIGN));
    {
        TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID0);
        WaitFlag<HardEvent::V_S>(eventID0);
    }
    tmpBuf.SetValue(0, val);
    {
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    Log(tmpBuf, tmpBuf, static_cast<int32_t>(FP32_ALIGN));
    {
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    return tmpBuf.GetValue(0);
}

// 标量 Exp
// 同样走"写入UB→V指令计算→读回标量"的流程（详见 ScalarSqrt 的说明），V 指令为 Exp。
__aicore__ inline float ScalarExp(LocalTensor<float>& tmpBuf, float val)
{
    Duplicate(tmpBuf, 0.0f, static_cast<int32_t>(FP32_ALIGN));
    {
        TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID0);
        WaitFlag<HardEvent::V_S>(eventID0);
    }
    tmpBuf.SetValue(0, val);
    {
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(eventID);
        WaitFlag<HardEvent::S_V>(eventID);
    }
    Exp(tmpBuf, tmpBuf, static_cast<int32_t>(FP32_ALIGN));
    {
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
    }
    return tmpBuf.GetValue(0);
}

} // namespace NsRenorm

#endif // _RENORM_COMMON_H_
