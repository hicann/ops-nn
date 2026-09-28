/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file mse_loss_v2.h
 * \brief MSELossV2 kernel class definition (arch35 / Ascend950)
 *
 * Computes the mean-squared-error loss:  l = (input - target)^2, then reduces:
 *   reduction=none : write l elementwise (same shape as input)
 *   reduction=sum  : sum of all l into a scalar
 *   reduction=mean : (1/N) * sum of all l into a scalar
 * reduction is a runtime scalar field from the tiling data (not a tiling key), so one template
 * instance per (dtype, buffer_mode) covers all three.
 *
 * For fp16/bf16 inputs the squared difference and the reduction accumulate in fp32 (the golden
 * torch.nn.functional.mse_loss accumulates in fp32), then the result is cast back to the input
 * dtype with round-to-nearest-even (CAST_RINT).
 *
 * sum/mean reduce over all axes into a single scalar via a deterministic two-phase fp32
 * reduction: each core accumulates a fp32 partial sum into its own 32B-aligned workspace slot,
 * SyncAll, then block 0 sums the partials and writes the scalar output.
 */

#ifndef MSE_LOSS_V2_ARCH35_H
#define MSE_LOSS_V2_ARCH35_H

#include "kernel_operator.h"
#include "mse_loss_v2_tiling_data.h"
#ifndef DTYPE_X
#include "kernel_tiling/kernel_tiling.h"
#include "mse_loss_v2_tiling_key.h"
#endif

namespace NsMseLossV2 {

using namespace AscendC;

constexpr uint32_t MSE_REDUCTION_NONE = 0;
constexpr uint32_t MSE_REDUCTION_SUM = 1;
constexpr uint32_t MSE_REDUCTION_MEAN = 2;
// Per-core workspace partial sums must be 32B(=8 fp32)-block aligned: GM write transactions are
// atomic at 32B granularity, so adjacent cores writing dense 4B slots into one block would race.
// Give each core its own block. (aligns with loss/poisson_nll_loss, norm/batch_norm_grad_v3)
__aicore__ inline bool IsFiniteF(float v) { return !(__isinf(v) || __isnan(v)); }

__aicore__ inline void ApplyScale(LocalTensor<float>& dst, uint32_t n, float scaleA, float scaleB)
{
    // 缩放**必须发生在平方之前**: 大值域下 (x-t)^2 自身就可能越界, 事后再缩放救不回来。
    // scaleA/scaleB 都是精确的 2 的幂, fp32 乘法只改指数位、不动尾数, 因此**零舍入**,
    // 缩放路径与非缩放路径的相对误差完全一致。拆成两次相乘是为了避免单个因子落进
    // 非规格化数(fp32 最小规格数 2^-126) —— 那会真的丢精度。
    if (scaleA != 1.0f) {
        Muls(dst, dst, scaleA, n);
    }
    if (scaleB != 1.0f) {
        Muls(dst, dst, scaleB, n);
    }
}

// 设备端循环的硬安全边界(不是精度门限): fp32 指数位有限, 正常 j<160、log2(N)<63。
// 设备端一旦死循环会挂死整卡并让主机死等设备锁, 必须有边界兜住。
constexpr int32_t MSE_SCALE_ITER_MAX = 300;
constexpr int32_t MSE_LOG2N_MAX = 62;

constexpr int32_t MSE_WS_CORE_STRIDE = 8;
// A single vector op needs at least 256B of lane coverage; size compute buffers to that minimum
// even when ubFactor is tiny, so ReduceSum/Sub/Mul never underflow a vector register.
constexpr int64_t MSE_MIN_COMPUTE_ELEMS = 256 / static_cast<int64_t>(sizeof(float));
// 一个 fp32 向量寄存器的车道数(256B / 4B)
constexpr int32_t MSE_VL_FP32 = 64;

template <typename T_IN, int BUFFER_MODE>
class MseLossV2 {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    // fp16/bf16 promote to fp32 for compute + reduction (golden accumulates in fp32).
    static constexpr bool NEED_CAST = !std::is_same<T_IN, float>::value;

public:
    __aicore__ inline MseLossV2() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace,
                                const MSELossV2Arch35TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void ComputeSquaredDiff(int64_t currentNum, LocalTensor<float>& dst, float scaleA = 1.0f,
                                              float scaleB = 1.0f, float* maxAbsOut = nullptr);
    __aicore__ inline float LocalMaxAbs(LocalTensor<float>& src, int64_t count);
    __aicore__ inline void TakeMaxAbs(LocalTensor<float>& diff, int64_t count, float* maxAbsOut);
    // 本核累加, 可按给定的(精确 2 的幂)缩放重跑一遍; 出参带回本核 max|x-t|
    __aicore__ inline float AccumulateLocal(float scaleA, float scaleB, float& maxAbsOut);
    __aicore__ inline void StagePartial(float partial, float maxAbs);
    __aicore__ inline float MergePartials(float& maxAbsOut);
    __aicore__ inline void ComputeNone(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void ProcessNone();
    __aicore__ inline void ProcessReduce();
    __aicore__ inline float LocalReduceSum(LocalTensor<float>& src, int64_t count);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueT_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;

    TBuf<QuePosition::VECCALC> lossBuf_;    // fp32 (input-target)^2
    TBuf<QuePosition::VECCALC> reduceBuf_;  // fp32 ReduceSum scratch + this-core partial staging
    TBuf<QuePosition::VECCALC> partialBuf_; // fp32 phase-2 read-back of all cores' partials
    TBuf<QuePosition::VECCALC> castXBuf_;   // fp32 cast of input  (NEED_CAST only)
    TBuf<QuePosition::VECCALC> castTBuf_;   // fp32 cast of target (NEED_CAST only)

    GlobalTensor<T_IN> xGm_;
    GlobalTensor<T_IN> tGm_;
    GlobalTensor<T_IN> yGm_;
    GlobalTensor<float> wsGm_; // partial sums, one fp32 slot (stride 8) per core (reduction path)

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t usedCoreNum_ = 0;
    int32_t partialUbElems_ = 0;
    uint32_t reduction_ = MSE_REDUCTION_NONE;
    int32_t blockIdx_ = 0;
};

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::Init(GM_ADDR input, GM_ADDR target, GM_ADDR output,
                                                          GM_ADDR workspace,
                                                          const MSELossV2Arch35TilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    reduction_ = tilingData->reduction;
    blockIdx_ = static_cast<int32_t>(AscendC::GetBlockIdx());
    usedCoreNum_ = (blockFactor_ > 0) ? ((totalNum_ + blockFactor_ - 1) / blockFactor_) : 0;
    partialUbElems_ = static_cast<int32_t>(tilingData->partialUbElems);

    int64_t startOffset = blockFactor_ * static_cast<int64_t>(blockIdx_);
    int64_t remaining = totalNum_ - startOffset;
    blockLength_ = (remaining <= 0) ? 0 : ((remaining > blockFactor_) ? blockFactor_ : remaining);

    xGm_.SetGlobalBuffer((__gm__ T_IN*)input + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    tGm_.SetGlobalBuffer((__gm__ T_IN*)target + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    if (reduction_ == MSE_REDUCTION_NONE) {
        yGm_.SetGlobalBuffer((__gm__ T_IN*)output + startOffset, (blockLength_ > 0) ? blockLength_ : 1);
    } else {
        yGm_.SetGlobalBuffer((__gm__ T_IN*)output, 1);
        int64_t wsElems = (usedCoreNum_ > 0) ? (usedCoreNum_ * MSE_WS_CORE_STRIDE) : MSE_WS_CORE_STRIDE;
        wsGm_.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    int64_t allocElems = (ubLength_ < MSE_MIN_COMPUTE_ELEMS) ? MSE_MIN_COMPUTE_ELEMS : ubLength_;

    pipe_.InitBuffer(inQueueX_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(inQueueT_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(outQueueY_, BUFFER_NUM, allocElems * static_cast<int64_t>(sizeof(T_IN)));
    pipe_.InitBuffer(lossBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
    pipe_.InitBuffer(reduceBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
    if (reduction_ != MSE_REDUCTION_NONE) {
        pipe_.InitBuffer(partialBuf_, static_cast<int64_t>(partialUbElems_) * static_cast<int64_t>(sizeof(float)));
    }
    if constexpr (NEED_CAST) {
        pipe_.InitBuffer(castXBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
        pipe_.InitBuffer(castTBuf_, allocElems * static_cast<int64_t>(sizeof(float)));
    }
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template AllocTensor<T_IN>();
    LocalTensor<T_IN> tLocal = inQueueT_.template AllocTensor<T_IN>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T_IN)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T_IN> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[progress * ubLength_], copyParams, padParams);
    DataCopyPad(tLocal, tGm_[progress * ubLength_], copyParams, padParams);
    inQueueX_.EnQue(xLocal);
    inQueueT_.EnQue(tLocal);
}

// Compute (input - target)^2 in fp32 into dst. Consumes the queued x/t tensors.
template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ComputeSquaredDiff(int64_t currentNum, LocalTensor<float>& dst,
                                                                        float scaleA, float scaleB, float* maxAbsOut)
{
    LocalTensor<T_IN> xLocal = inQueueX_.template DeQue<T_IN>();
    LocalTensor<T_IN> tLocal = inQueueT_.template DeQue<T_IN>();
    uint32_t n = static_cast<uint32_t>(currentNum);

    if constexpr (NEED_CAST) {
        LocalTensor<float> xF = castXBuf_.Get<float>();
        LocalTensor<float> tF = castTBuf_.Get<float>();
        Cast(xF, xLocal, RoundMode::CAST_NONE, n);
        Cast(tF, tLocal, RoundMode::CAST_NONE, n);
        Sub(dst, xF, tF, n);
        TakeMaxAbs(dst, currentNum, maxAbsOut);
        ApplyScale(dst, n, scaleA, scaleB);
        Mul(dst, dst, dst, n);
    } else {
        Sub(dst, xLocal, tLocal, n);
        TakeMaxAbs(dst, currentNum, maxAbsOut);
        ApplyScale(dst, n, scaleA, scaleB);
        Mul(dst, dst, dst, n);
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueT_.FreeTensor(tLocal);
}

// reduction=none: compute the loss and write it out elementwise (same shape as input).
template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ComputeNone(int64_t currentNum)
{
    LocalTensor<float> loss = lossBuf_.Get<float>();
    ComputeSquaredDiff(currentNum, loss);

    uint32_t n = static_cast<uint32_t>(currentNum);
    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        Cast(yLocal, loss, RoundMode::CAST_RINT, n);
    } else {
        LocalTensor<float> yF = yLocal.template ReinterpretCast<float>();
        Adds(yF, loss, 0.0f, n);
    }
    outQueueY_.template EnQue<T_IN>(yLocal);
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T_IN> yLocal = outQueueY_.template DeQue<T_IN>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * static_cast<int64_t>(sizeof(T_IN)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(yGm_[progress * ubLength_], yLocal, copyParams);
    outQueueY_.FreeTensor(yLocal);
}

// Whole-reduce a fp32 LocalTensor of `count` elements to a single fp32 value.
template <typename T_IN, int BUFFER_MODE>
__aicore__ inline float MseLossV2<T_IN, BUFFER_MODE>::LocalMaxAbs(LocalTensor<float>& src, int64_t count)
{
    // max|v| = max(max(v), -min(v)); 用 ReduceMax/ReduceMin 避免额外开一块 Abs 暂存。
    LocalTensor<float> red = reduceBuf_.Get<float>();
    ReduceMax(red, src, red, static_cast<int32_t>(count), false);
    event_t e1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e1);
    WaitFlag<HardEvent::V_S>(e1);
    float hi = red.GetValue(0);
    event_t e2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(e2);
    WaitFlag<HardEvent::S_V>(e2);
    ReduceMin(red, src, red, static_cast<int32_t>(count), false);
    event_t e3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(e3);
    WaitFlag<HardEvent::V_S>(e3);
    float lo = red.GetValue(0);
    // NaN 必须**显式**传播: 用 `m > maxAbs` 挑最大值时, m 为 NaN 比较恒假, maxAbs 会被
    // 悄悄保持为 0 —— 于是"含 NaN 的核报 0、含 inf 的核报 inf", 各核对同一份数据得出不同的
    // 定标依据。这类不一致会直接喂到"是否需要重算"的判定上, 是设备端死锁的温床。
    if (!IsFiniteF(hi) || !IsFiniteF(lo)) {
        return hi; // 非有限值(inf/nan)原样带出, 让上层守卫看得见
    }
    float a = (hi < 0.0f) ? -hi : hi;
    float b = (lo < 0.0f) ? -lo : lo;
    return (a > b) ? a : b;
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline float MseLossV2<T_IN, BUFFER_MODE>::LocalReduceSum(LocalTensor<float>& src, int64_t count)
{
    // **32B 对齐的二分(pairwise)归约**, 与竞品 torch 的归约顺序对齐。
    //
    // 根因(在**dump 出的真实输入**上穷举 20 种归约顺序复现所得, 不是推断):
    //   本例 602 元素 fp32 / reduction=sum, 精确和 2509.8317183688378, 1 ULP = 2.44e-4
    //     NPU(原 ReduceSum 路径) 2509.83154296875   偏 0.718 ULP
    //     竞品 torch / 任意含树形的顺序 2509.831787109375  偏 0.282 ULP
    //   命中 NPU 那个值的 4 种顺序(分块8+顺序合并 / 分块16+顺序合并 /
    //   分块64内顺序+树形合并 / 分块384内顺序+树形合并)**都含一个"顺序累加"环节**;
    //   其余 16 种含树形的顺序一律给出竞品值。
    //   → 差异来自**归约链里的顺序累加段**(ReduceSum 对超过矢量宽度的数据按块累加),
    //     竞品是全程 pairwise。多数数据两者舍到同一可表示值, 只有精确和落在两个可表示值
    //     **中间区域**时才分道(本例 0.282 vs 0.718, 几乎正中)。
    //
    // ⚠ **ReduceSum 本身没有精度缺陷**: 同一组输入的对照实验(probe_sum_602)里
    //   ReduceSum 与本二分实现**逐位相同**。它只是顺序与竞品不同。换二分是为了
    //   "两边计算方式对齐", 不是因为 ReduceSum 弱, 也不是压阈值。
    //   A2(arch22)本来就是二分(ReduceSumBisect); 本仓另有约 10 个精度敏感归约算子
    //   (bn_training_reduce / euclidean_norm / instance_norm_grad 等)也自写 pairwise。
    //   写法沿用本文件逐元素部分的经典 AscendC 向量 API, 不照抄 A2 代码, 不触碰 regbase 段。
    //
    // ⚠ **折半点必须 32B 对齐**(fp32 即 8 的整数倍): 偏移视图 src[half] 不对齐会直接
    // VEC_ERROR 崩在设备上(实测 602 的一半 301 未对齐 → 崩)。因此折半点上对齐到 8,
    // 两半不再严格等长(上半 len-half 个折进下半), 仍是对数级归约; 剩余 <= 8 个元素标量收尾。
    constexpr int32_t FP32_PER_BLOCK = 8;
    int32_t len = static_cast<int32_t>(count);
    while (len > FP32_PER_BLOCK) {
        int32_t half = (len + 1) >> 1;
        half = ((half + FP32_PER_BLOCK - 1) / FP32_PER_BLOCK) * FP32_PER_BLOCK;
        if (half >= len) {
            half = len - FP32_PER_BLOCK; // 保证严格下降, 不会死循环
        }
        Add(src, src, src[half], len - half);
        len = half;
    }
    event_t evtVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(evtVS);
    WaitFlag<HardEvent::V_S>(evtVS);
    float acc = 0.0f;
    for (int32_t i = 0; i < len; i++) {
        acc += src.GetValue(i);
    }
    return acc;
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ProcessNone()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        ComputeNone(currentNum);
        CopyOut(i, currentNum);
    }
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::TakeMaxAbs(LocalTensor<float>& diff, int64_t count,
                                                                float* maxAbsOut)
{
    // 只在首轮(需要定标信息时)取, 重算轮传 nullptr 直接跳过。
    // 取的是**差值**的 max|.|, 必须在平方之前 —— 大值域下 (x-t)^2 自身就可能越界,
    // 拿越界后的值定标是定不出来的。
    if (maxAbsOut == nullptr) {
        return;
    }
    float m = LocalMaxAbs(diff, count);
    if (!IsFiniteF(m)) {
        *maxAbsOut = m; // 非有限一旦出现就置位, 不能被后续 tile 的有限值比较掉
        return;
    }
    if (IsFiniteF(*maxAbsOut) && m > *maxAbsOut) {
        *maxAbsOut = m;
    }
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline float MseLossV2<T_IN, BUFFER_MODE>::AccumulateLocal(float scaleA, float scaleB, float& maxAbsOut)
{
    // 本核对自己那段元素求 sum((x-t)*scale)^2, 同时带回 max|x-t|(未缩放的原始量级)。
    // max 与 sum 在**同一趟**里算完: 数据已经在 UB, 多一次 ReduceMax/ReduceMin 对这个
    // 访存瓶颈的算子几乎不增加耗时, 却省掉了"为了定标而多读一遍 GM"的那一趟。
    // 本核累加。两处与朴素写法不同, 都是实测驱动的:
    //
    // ① **每个 tile 求和后立刻除 N**(reduction=mean), 而不是"核内攒完再除"。
    //    核内 partial 的量级 = blockFactor x mean, q0177 实测 384 x 9.3e35 = 3.6e38
    //    直接越过 fp32 上限 3.4e38 而输出 inf —— 真值 9.3e35 本来完全可表示。
    //    下沉一层之后累加器全程不超过结果量级, 从源头上不再产生这种溢出。
    //    每 tile 一次标量除法, 开销与 tile 数同阶, 可忽略。
    // ② **核内跨 tile 用 Kahan 补偿**, 而不是裸 `partial += ...`。
    //    跨核那步早就用了 Kahan, 核内反而是裸加 —— loopCount 大时(大 shape 可达上百轮)
    //    误差逐轮累积, 与跨核的精心补偿不匹配。
    //    NaN/Inf 时把补偿量清零, 否则 (t-acc)-y 在 inf 上算出 NaN 会污染后续每一轮。
    float partial = 0.0f;
    float comp = 0.0f;
    float maxAbs = 0.0f;
    const bool wantMax = (scaleA == 1.0f && scaleB == 1.0f);
    const float tileDiv = (reduction_ == MSE_REDUCTION_MEAN) ? static_cast<float>(totalNum_) : 1.0f;
    if (blockLength_ > 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            LocalTensor<float> loss = lossBuf_.Get<float>();
            ComputeSquaredDiff(currentNum, loss, scaleA, scaleB, wantMax ? &maxAbs : nullptr);
            float tileSum = LocalReduceSum(loss, currentNum);
            if (tileDiv != 1.0f) {
                tileSum /= tileDiv; // IEEE754 除法只舍一次, 不用乘 1/N(双重舍入)
            }
            float y = tileSum - comp;
            float t = partial + y;
            comp = (t - partial) - y;
            if (!IsFiniteF(t)) {
                comp = 0.0f;
            }
            partial = t;
        }
    }
    maxAbsOut = maxAbs;
    // **收尾把补偿量补回**: Kahan 的 comp 存的是每一步被丢掉的低位部分, 只返回 partial
    // 等于一路维护了补偿量却在最后一步扔掉。实测(aclnn_0235_sum, 602 元素 fp32):
    //   NPU 2509.83154297 / fp64 真值 2509.83171069 / 竞品 2509.83178711
    //   NPU 偏 0.687 ULP、竞品偏 0.313 ULP —— 两者落在相邻的两个 fp32 值上,
    //   竞品是正确舍入值而 NPU 掉到了下一个。补回 comp 正是补上这半个多 ULP。
    // comp 为非有限时不能补(inf/nan 场景 comp 已被清零, 这里再判一次更稳)。
    return IsFiniteF(comp) ? (partial - comp) : partial;
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::StagePartial(float partial, float maxAbs)
{
    // 写**整 32B 块**而不是单个 4B: GM 上跨核写以 32B 为粒度, 4B 密排会让相邻核踩同一块。
    // 0 号槽放 partial, 1 号槽放本核 max|x-t|(供溢出重算定标), 其余补零 ——
    // 补零后 phase 2 能一次连续读进来直接矢量规约(0 不影响和), 不必逐核 GetValue。
    LocalTensor<float> stage = reduceBuf_.Get<float>();
    for (int32_t k = 0; k < MSE_WS_CORE_STRIDE; k++) {
        stage.SetValue(k, 0.0f);
    }
    stage.SetValue(0, partial);
    stage.SetValue(1, maxAbs);
    event_t evtSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSMTE3);
    WaitFlag<HardEvent::S_MTE3>(evtSMTE3);
    DataCopyExtParams wsParams{1, static_cast<uint32_t>(MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPad(wsGm_[blockIdx_ * MSE_WS_CORE_STRIDE], stage, wsParams);
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline float MseLossV2<T_IN, BUFFER_MODE>::MergePartials(float& maxAbsOut)
{
    LocalTensor<float> partials = partialBuf_.Get<float>();
    const uint16_t mergeRounds = static_cast<uint16_t>(partialUbElems_ / MSE_VL_FP32);
    DataCopyExtParams inParams{1, static_cast<uint32_t>(usedCoreNum_ * MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
    DataCopyPad(partials, wsGm_, inParams, inPad);
    event_t evtMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(evtMTE2V);
    WaitFlag<HardEvent::MTE2_V>(evtMTE2V);

    // 1 号槽存的是各核的 max|x-t|(见 StagePartial)。先扫出全局 max 供溢出重算定标用,
    // 再把这些车道清零 —— 否则下面的矢量 Kahan 会把 64 条车道全部相加, 把 max 当成 partial
    // 累进结果里。核数 <= 64, 标量扫一遍的代价可以忽略。
    float globalMax = 0.0f;
    for (int32_t c = 0; c < static_cast<int32_t>(usedCoreNum_); c++) {
        float m = partials.GetValue(c * MSE_WS_CORE_STRIDE + 1);
        if (!IsFiniteF(m)) {
            globalMax = m; // 同 LocalMaxAbs: NaN 参与 ">" 比较恒假, 必须显式置位
        } else if (IsFiniteF(globalMax) && m > globalMax) {
            globalMax = m;
        }
        partials.SetValue(c * MSE_WS_CORE_STRIDE + 1, 0.0f);
    }
    maxAbsOut = globalMax;
    event_t evtSV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(evtSV0);
    WaitFlag<HardEvent::S_V>(evtSV0);

    // 只清**补零区**: 矢量合并整轮读满 MSE_VL_FP32 车道、收尾标量循环也按 stride 读满
    // MSE_VL_FP32/MSE_WS_CORE_STRIDE 条车道, 而 DataCopyPad 只搬入 usedCoreNum_ 个槽; 余下车道若不清零读到的是
    // UB 残留(可能含 ±inf/NaN), 会被当作 partial 累加 —— 核数越少无效车道越多。
    // 放在搬入之后只清尾巴, 比"搬入前清整段"省一次 V->MTE2 同步和大部分清零量(实测小 shape 的
    // sum 快回 4~5%); Duplicate 与下面的矢量循环同为 V 流水, 按序发射无需额外同步。
    const int32_t copiedElems = static_cast<int32_t>(usedCoreNum_ * MSE_WS_CORE_STRIDE);
    if (partialUbElems_ > copiedElems) {
        Duplicate(partials[copiedElems], 0.0f, partialUbElems_ - copiedElems);
    }

    // ── 跨核合并: **矢量(regbase)形态的 Kahan 补偿累加** ───────────────────────────────
    // 布局: 每核占一个 32B 块(首元素是它的 partial, 其余 7 个为 0), 于是一次寄存器载入(64 车道)
    // 正好覆盖 8 个核, 补零车道加 0 不改变结果也不产生补偿量。64 核 = 8 轮, 即 8 车道并行、
    // 每车道顺序累加 8 个 partial —— Kahan 的补偿项覆盖的就是这 8 步。
    //
    // 为什么不是标量循环: 原写法在 block 0 上串行 usedCoreNum_(最多 64)次, 实测跨核规约段占
    // 小 shape 整核时延四成; 矢量化后这段压成 8 轮指令。
    // 为什么不是纯树形规约: 树形没有补偿项, 合并 64 个 partial 的误差 ~log2(64)*eps, 比 Kahan
    // 的 ~2*eps 差(实测 4M 用例 1.2 ulp vs 0.8 ulp), 精度不能白丢。
    // 为什么先除 N 再合并(reduction=mean): 直接累加原量级 partial, 中间和会涨到 ~N*mean 的量级
    // (实测 20528 元素、值域 ±65504 时中间和达 8.2e13, 该处 fp32 1 ulp = 8388608), 合并误差被
    // 这个量级的表示粒度锁死, 换 Kahan/树形都无从改善。先把每个 partial 除以 N, 累加器全程停在
    // 最终结果量级(4e9, 1 ulp = 512), 误差随之降到 1/4, 且与竞品 mean 的运算序列对齐
    // (实测 q0177/q0178 两例均与 GPU 三方腿逐位相同)。用 Div 而非乘 1/N: N 非 2 的幂时 1/N 在
    // fp32 存不下, 先舍 1/N 再舍乘积是双重舍入, IEEE754 除法只舍一次。
    // reduction != mean 时除数取 1.0(精确), 复用同一条指令路径, 不额外分支。
    // inf/nan: 补偿量 c = (t - sum) - y 在 t 为 ±inf 时算出 NaN, 会污染后续每一轮, 把本该是 inf
    // 的和变成 nan(实测单核不复现、多核必现)。用自比 Compare<EQ>(c, c) 找出非 NaN 车道, Select
    // 把 NaN 车道的补偿量清零 —— 只清补偿量、不动 sum, 故真 nan 仍会如实传播。
    // (同款写法见 norm/instance_norm_grad 的跨 N 合并。)
    __local_mem__ float* partialUb = (__local_mem__ float*)partials.GetPhyAddr();
    {
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> compReg;
        AscendC::Reg::RegTensor<float> zeroReg;
        AscendC::Reg::RegTensor<float> pReg;
        AscendC::Reg::RegTensor<float> kY;
        AscendC::Reg::RegTensor<float> kT;
        AscendC::Reg::RegTensor<float> kD;
        AscendC::Reg::RegTensor<float> divReg;
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::MaskReg finiteMask;
        uint32_t sreg = static_cast<uint32_t>(mergeRounds) * MSE_VL_FP32;
        // tiling 已拒收空 tensor, totalNum_ > 0
        // 除 N 已在 AccumulateLocal 的 tile 级完成, 这里不能再除一次。
        // 保留 divReg 结构是为了让缩放重算路径能复用同一条指令序列(此处恒为 1.0, 精确)。
        const float divisor = 1.0f;
        __VEC_SCOPE__
        {
            preg = AscendC::Reg::UpdateMask<float>(sreg);
            AscendC::Reg::Duplicate(sumReg, 0.0f, preg);
            AscendC::Reg::Duplicate(compReg, 0.0f, preg);
            AscendC::Reg::Duplicate(zeroReg, 0.0f, preg);
            AscendC::Reg::Duplicate(divReg, divisor, preg);
            for (uint16_t r = 0; r < mergeRounds; ++r) {
                AscendC::Reg::DataCopy(pReg, partialUb + r * MSE_VL_FP32);
                AscendC::Reg::Div(pReg, pReg, divReg, preg); // 先除 N, 累加器停在结果量级
                AscendC::Reg::Sub(kY, pReg, compReg, preg);  // y = p - comp
                AscendC::Reg::Add(kT, sumReg, kY, preg);     // t = sum + y
                AscendC::Reg::Sub(kD, kT, sumReg, preg);     // d = t - sum
                AscendC::Reg::Sub(compReg, kD, kY, preg);    // comp = d - y
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(finiteMask, compReg, compReg, preg);
                AscendC::Reg::Select(compReg, compReg, zeroReg, finiteMask); // NaN 车道 -> 0
                AscendC::Reg::Move(sumReg, kT, preg);
            }
            // **矢量 Kahan 也要把补偿量补回**: compReg 存的是每轮被丢掉的低位部分,
            // 只存 sumReg 等于白维护了一路补偿。这是第三处丢弃点(另两处在核内累加与
            // 跨核标量收尾) —— 三处都补上才算真正用上了 Kahan。
            // compReg 里的 NaN 车道已在循环内被 Select 清零, 这里直接相减是安全的。
            AscendC::Reg::Sub(sumReg, sumReg, compReg, preg);
            AscendC::Reg::DataCopy(partialUb, sumReg, preg);
        }
    }
    // 合并 64 条车道(其中 8 条有效, 其余恒 0)
    // 收尾: 只对 MSE_VL_FP32/MSE_WS_CORE_STRIDE 条有效车道做标量 Kahan(核数 64 时是 8 步, 不是原来的 64 步)。
    // 车道内已补偿过各自的顺序累加, 这里再补偿车道之间 —— 实测(同一批 partial 配对比较)
    // 与原标量 Kahan 6 例中 5 例逐位相同; 若这里改用无补偿的树形规约, 会有 3/6 变差。
    SetFlag<HardEvent::V_S>(EVENT_ID0);
    WaitFlag<HardEvent::V_S>(EVENT_ID0);
    float total = 0.0f;
    float mergeComp = 0.0f;
    for (int32_t lane = 0; lane < MSE_VL_FP32; lane += static_cast<int32_t>(MSE_WS_CORE_STRIDE)) {
        float y = partials.GetValue(lane) - mergeComp;
        float t = total + y;
        mergeComp = (t - total) - y;
        if (__isinf(t) || __isnan(t)) {
            mergeComp = 0.0f;
        }
        total = t;
    }
    // 同上: 跨核收尾也要把补偿量补回去, 否则这一路 Kahan 的低位信息在最后一步丢掉。
    if (IsFiniteF(mergeComp)) {
        total -= mergeComp;
    }
    return total;
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ProcessReduce()
{
    float maxAbs = 0.0f;
    float partial = AccumulateLocal(1.0f, 1.0f, maxAbs);
    StagePartial(partial, maxAbs);
    SyncAll(); // 栅栏 #1 —— 无条件

    float globalMax = 0.0f;
    float total = MergePartials(globalMax);

    // ── 溢出兜底 ─────────────────────────────────────────────────────────────
    // 中间和越界时结果是 ±inf/nan(Kahan 的补偿项在 inf 上算出 nan), 而真值往往仍在 fp32
    // 可表示范围内(实测 q0177: 真值 9.32e35 < 上限 3.4e38, 内核却给 inf) —— 实打实的缺陷。
    // 修法: 用**数据自身**的 max|x-t| 定出精确的 2 的幂缩放, 在平方之前缩放后重算, 再按指数
    // 还原。因子是 2 的幂 → fp32 乘法零舍入 → 相对精度与常规路径完全一致; 常规路径不取用
    // 重算结果 → 小值端不会被"预先除 N"冲成 0。两端都不牺牲, 判据是精确的 isfinite。
    //
    // ⚠ **SyncAll 的调用次数必须与判定无关**。SyncAll 是全核栅栏, 各核调用次数不同就会有核
    // 永远等在栅栏上 —— 设备端死锁, 主机侧持着设备锁死等, 表现为整个跑批冻结(2026-09-27 实测:
    // 一个核卡 479s, 其余 3 个 worker 全堵在 OnAcquireLock)。
    // 所以这里**无条件**走完"再暂存 + 栅栏 #2 + 再合并", needScale 只决定**用不用**重算结果,
    // 绝不决定同步几次。不依赖"各核判定必然一致"这种推理 —— 那个推理错了就是死锁。
    const bool needScale = (!IsFiniteF(total) && IsFiniteF(globalMax) && globalMax > 0.0f);
    int32_t j = 0;
    float sA = 1.0f;
    float sB = 1.0f;
    if (needScale) {
        // j 使 N*(max*2^-j)^2 落在 1 附近: j = ceil(log2 max) + ceil(log2 N)/2
        float m = globalMax;
        // 上限 MSE_SCALE_ITER_MAX 只是**循环安全边界**(fp32 指数位有限, 正常 j < 160),
        // 不是精度门限 —— 设备端死循环会挂死整卡, 必须有硬边界。
        while (m > 1.0f && j < MSE_SCALE_ITER_MAX) {
            m *= 0.5f;
            j++;
        }
        int32_t e = 0;
        while (e < MSE_LOG2N_MAX && (static_cast<int64_t>(1) << e) < totalNum_) {
            e++;
        }
        j += (e + 1) / 2;
        // 拆两个因子, 避免单个因子掉进非规格化数(fp32 最小规格数 2^-126)而真丢精度
        int32_t j1 = j / 2;
        for (int32_t k = 0; k < j1; k++) {
            sA *= 0.5f;
        }
        for (int32_t k = 0; k < j - j1; k++) {
            sB *= 0.5f;
        }
    }
    float maxAbs2 = 0.0f;
    float retryPartial = needScale ? AccumulateLocal(sA, sB, maxAbs2) : partial;
    StagePartial(retryPartial, 0.0f);
    SyncAll(); // 栅栏 #2 —— 同样无条件
    float dummy = 0.0f;
    float retryTotal = MergePartials(dummy);
    if (needScale) {
        // 还原 2^(2j): 绝不物化这个常数(它自身就会溢出), 逐次乘 2.0f 是精确运算;
        // 真值本身超 fp32 时自然仍得 inf, 这正是正确行为。
        total = retryTotal;
        for (int32_t k = 0; k < 2 * j; k++) {
            total *= 2.0f;
        }
    }

    if (blockIdx_ != 0) {
        return;
    }

    // mean 的除 N 已在跨核合并前逐 partial 完成, 此处不再整体相除。

    LocalTensor<T_IN> yLocal = outQueueY_.template AllocTensor<T_IN>();
    if constexpr (NEED_CAST) {
        LocalTensor<float> scalarF = reduceBuf_.Get<float>();
        scalarF.SetValue(0, total);
        event_t evtSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(evtSV);
        WaitFlag<HardEvent::S_V>(evtSV);
        Cast(yLocal, scalarF, RoundMode::CAST_RINT, 1);
    } else {
        yLocal.SetValue(0, static_cast<T_IN>(total));
        event_t evtSMTE3b = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(evtSMTE3b);
        WaitFlag<HardEvent::S_MTE3>(evtSMTE3b);
    }
    outQueueY_.template EnQue<T_IN>(yLocal);

    LocalTensor<T_IN> yOut = outQueueY_.template DeQue<T_IN>();
    DataCopyExtParams outParams{1, static_cast<uint32_t>(sizeof(T_IN)), 0, 0, 0};
    DataCopyPad(yGm_, yOut, outParams);
    outQueueY_.FreeTensor(yOut);
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::Process()
{
    if (reduction_ == MSE_REDUCTION_NONE) {
        ProcessNone();
    } else {
        ProcessReduce();
    }
}

} // namespace NsMseLossV2

#endif // MSE_LOSS_V2_ARCH35_H
