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
    __aicore__ inline void StagePartial(float partial, float maxAbs, int32_t jCore);
    __aicore__ inline float MergePartialsRestore();
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
        // 两轮各占互不重叠的区域: 复用同一块会让快核改写后、慢核读到混合两轮的数据,
        // 导致各核缩放判定不一致、合并出 inf。
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
    // 32B 对齐的二分(pairwise)归约: 归约顺序决定舍入点, 与参照实现保持一致。
    // 折半点必须按 32B(fp32 即 8 元素)上对齐, 否则偏移视图不对齐会触发 VEC_ERROR;
    // 两半因此不严格等长, 剩余不足 8 个元素标量收尾。
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
    // 两点与朴素写法不同: ① reduction=mean 时每个 tile 求和后立刻除 N, 使累加器量级不超过
    // 结果量级, 避免核内 partial 溢出 fp32; ② 核内跨 tile 用 Kahan 补偿, 与跨核那步一致。
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
    // 收尾把 Kahan 补偿量补回: comp 是每步被丢掉的低位, 只返回 partial 等于白维护补偿。
    // comp 非有限时不补(inf/nan 场景下 comp 已清零, 此处再判一次)。
    return IsFiniteF(comp) ? (partial - comp) : partial;
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::StagePartial(float partial, float maxAbs, int32_t jCore)
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
    stage.SetValue(2, static_cast<float>(jCore)); // 本核的缩放指数, 供 0 号核还原
    event_t evtSMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(evtSMTE3);
    WaitFlag<HardEvent::S_MTE3>(evtSMTE3);
    DataCopyExtParams wsParams{1, static_cast<uint32_t>(MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPad(wsGm_[blockIdx_ * MSE_WS_CORE_STRIDE], stage, wsParams);
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline float MseLossV2<T_IN, BUFFER_MODE>::MergePartialsRestore()
{
    // 只由 0 号核调用: 逐核读 (partial, j), 按各核自带的 j 还原后做 Kahan 累加。
    // 还原用"逐次乘 2.0f"而非物化 2^(2j)(那个常数自身会溢出); 2 的幂乘法零舍入。
    LocalTensor<float> partials = partialBuf_.Get<float>();
    DataCopyExtParams inParams{1, static_cast<uint32_t>(usedCoreNum_ * MSE_WS_CORE_STRIDE * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> inPad{false, 0, 0, 0};
    // 读别核槽位前逐槽失效本核数据缓存: SyncAll 只保证都写完, 不保证本核读到 GM 最新值。
    for (int64_t off = 0; off < usedCoreNum_ * MSE_WS_CORE_STRIDE; off += MSE_WS_CORE_STRIDE) {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(wsGm_[off]);
    }
    DataCopyPad(partials, wsGm_, inParams, inPad);
    event_t evtMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(evtMTE2S);
    WaitFlag<HardEvent::MTE2_S>(evtMTE2S);

    float total = 0.0f;
    float comp = 0.0f;
    for (int32_t c = 0; c < static_cast<int32_t>(usedCoreNum_); c++) {
        float v = partials.GetValue(c * MSE_WS_CORE_STRIDE);
        const int32_t jc = static_cast<int32_t>(partials.GetValue(c * MSE_WS_CORE_STRIDE + 2));
        for (int32_t k = 0; k < 2 * jc; k++) {
            v *= 2.0f; // 精确: 只改指数位
        }
        float y = v - comp;
        float t = total + y;
        comp = (t - total) - y;
        if (!IsFiniteF(t)) {
            comp = 0.0f; // inf/nan 上的补偿项会算出 NaN, 必须清零
        }
        total = t;
    }
    // 收尾把补偿量补回(同核内 Kahan): 否则一路维护的低位信息在最后一步被丢掉。
    return IsFiniteF(comp) ? (total - comp) : total;
}

template <typename T_IN, int BUFFER_MODE>
__aicore__ inline void MseLossV2<T_IN, BUFFER_MODE>::ProcessReduce()
{
    // 跨核规约: 缩放判定是纯本地决策 —— 每核只看自己的 partial 是否溢出, 溢出就按本核的
    // max|x-t| 定出 2 的幂缩放因子并只重算本核那段, 把该指数随 partial 上报, 0 号核还原后累加。
    // 若改成各核都读全部 partial 再各自判定, 跨核读的可见性差异会让判定分叉、合并出 inf。
    float maxAbs = 0.0f;
    float partial = AccumulateLocal(1.0f, 1.0f, maxAbs);
    int32_t jCore = 0;
    if (!IsFiniteF(partial) && IsFiniteF(maxAbs) && maxAbs > 0.0f) {
        // j 使 blockFactor*(max*2^-j)^2 落在 1 附近: j = ceil(log2 max) + ceil(log2 N)/2
        float m = maxAbs;
        // MSE_SCALE_ITER_MAX 是循环安全边界(设备端死循环会挂死整卡), 不是精度门限。
        while (m > 1.0f && jCore < MSE_SCALE_ITER_MAX) {
            m *= 0.5f;
            jCore++;
        }
        int32_t e = 0;
        while (e < MSE_LOG2N_MAX && (static_cast<int64_t>(1) << e) < totalNum_) {
            e++;
        }
        jCore += (e + 1) / 2;
        // 拆两个因子, 避免单个因子掉进非规格化数(fp32 最小规格数 2^-126)而真丢精度
        float sA = 1.0f;
        float sB = 1.0f;
        int32_t j1 = jCore / 2;
        for (int32_t k = 0; k < j1; k++) {
            sA *= 0.5f;
        }
        for (int32_t k = 0; k < jCore - j1; k++) {
            sB *= 0.5f;
        }
        float dummyMax = 0.0f;
        partial = AccumulateLocal(sA, sB, dummyMax);
    }
    StagePartial(partial, maxAbs, jCore);
    // 过栅栏前排空 MTE3: StagePartial 的 DataCopyPad 是异步写, 不排空则 0 号核可能读到旧值。
    PipeBarrier<PIPE_MTE3>();
    SyncAll(); // 唯一一道跨核栅栏, 无条件

    if (blockIdx_ != 0) {
        return;
    }

    // mean 的除 N 已在跨核合并前逐 partial 完成, 此处不再整体相除。
    float total = MergePartialsRestore();

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
