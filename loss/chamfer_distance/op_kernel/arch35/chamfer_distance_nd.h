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
 * \file chamfer_distance_nd.h
 * \brief ChamferDistance arch35 内核: 逐查询点对另一点集做最小平方距离 + argmin 归约
 *
 * 输入布局 (2, B, N): xyz[0] 为全部 x 坐标、xyz[1] 为全部 y 坐标(见 01 §6.1)。
 * 调度: B*N 个查询点摊平分核; 每个查询点内, 被查集合按段过 UB, 段内一次
 *       ReduceMin(calIndex=true) 出(最小值, 段内下标), 跨段用标量严格小于更新。
 */

#ifndef CHAMFER_DISTANCE_ND_H
#define CHAMFER_DISTANCE_ND_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "chamfer_distance_tiling_data.h"

namespace ChamferDistance {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::UpdateMask;

// fp16/bf16 → fp32 的随路 cast trait(与 activation/clipped_swiglu 同款)
constexpr static AscendC::MicroAPI::CastTrait CAST_B16_TO_FP32 = {
    AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN};

// VF 计算段: 一次算一段被查点到查询点的平方距离。
// 范式照 activation/relu6_d 的工作实现 —— __simd_vf__ 自由函数 + 每轮 UpdateMask + 裸指针推进;
// 成员函数里包 __VEC_SCOPE__ 再配 AddrReg/三元选 mask 会触发后端 "Unsupported Inst must be hoisted"。
// fp16/bf16 在这里随路 DIST_UNPACK_B16 载入 + Cast 到 fp32, 不另开 fp32 暂存缓冲。
template <typename T>
__simd_vf__ inline void ChamferDistVF(__ubuf__ float* distAddr, __ubuf__ T* xAddr, __ubuf__ T* yAddr, float negX1,
                                      float negY1, uint32_t count, uint32_t fp32Lane, uint16_t repeatTimes)
{
    AscendC::MicroAPI::RegTensor<float> dx;
    AscendC::MicroAPI::RegTensor<float> dy;
    AscendC::MicroAPI::RegTensor<float> sqX;
    AscendC::MicroAPI::RegTensor<float> sqY;
    AscendC::MicroAPI::MaskReg mask;
    uint32_t remain = count; // UpdateMask 每轮自行按 VL 递减, 无需外部再算余量

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = AscendC::MicroAPI::UpdateMask<float>(remain);
        if constexpr (IsSameType<T, float>::value) {
            AscendC::MicroAPI::LoadAlign(dx, xAddr + i * fp32Lane);
            AscendC::MicroAPI::LoadAlign(dy, yAddr + i * fp32Lane);
        } else {
            AscendC::MicroAPI::RegTensor<T> rawX;
            AscendC::MicroAPI::RegTensor<T> rawY;
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(rawX, xAddr + i * fp32Lane);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(rawY, yAddr + i * fp32Lane);
            AscendC::MicroAPI::Cast<float, T, CAST_B16_TO_FP32>(dx, rawX, mask);
            AscendC::MicroAPI::Cast<float, T, CAST_B16_TO_FP32>(dy, rawY, mask);
        }
        // d = (x2 - x1)^2 + (y2 - y1)^2, 标量用加负值
        AscendC::MicroAPI::Adds(dx, dx, negX1, mask);
        AscendC::MicroAPI::Adds(dy, dy, negY1, mask);
        AscendC::MicroAPI::Mul(sqX, dx, dx, mask);
        AscendC::MicroAPI::Mul(sqY, dy, dy, mask);
        AscendC::MicroAPI::Add(sqX, sqX, sqY, mask);
        AscendC::MicroAPI::StoreAlign(distAddr + i * fp32Lane, sqX, mask);
    }
}

// 标量转换: half 有到/自 float 的转换, bfloat16_t 没有, 必须走 ToFloat/ToBfloat16
template <typename T>
__aicore__ inline float ScalarToFloat(const T& v)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        return AscendC::ToFloat(v);
    } else {
        return static_cast<float>(v);
    }
}

template <typename T>
__aicore__ inline T ScalarFromFloat(float v)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        return AscendC::ToBfloat16(v);
    } else {
        return static_cast<T>(v);
    }
}

template <typename T>
class ChamferDistanceND {
public:
    __aicore__ inline ChamferDistanceND() {}

    __aicore__ inline void Init(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist1, GM_ADDR dist2, GM_ADDR idx1, GM_ADDR idx2,
                                const ChamferDistanceArch35TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    // 一个方向: query 集合逐点扫 scan 集合, 结果落 dist/idx
    __aicore__ inline void RunDirection(const GlobalTensor<T>& queryGm, const GlobalTensor<T>& scanGm,
                                        const GlobalTensor<T>& distGm, const GlobalTensor<int32_t>& idxGm);
    __aicore__ inline void CopyInQuery(const GlobalTensor<T>& queryGm, int64_t taskBase, int64_t len);
    __aicore__ inline void CopyInScan(const GlobalTensor<T>& scanGm, int64_t base, int64_t len);
    __aicore__ inline void CalcDistVec(int64_t len, float x1, float y1);
    // 候选块要被查询 tile 内多个查询点复用, 故把"取缓冲/算距离/放缓冲"拆开
    __aicore__ inline void AcquireScan();
    __aicore__ inline void CalcDistVecKeep(int64_t len, float x1, float y1);
    __aicore__ inline void FreeScan();
    __aicore__ inline void ScanBatchSegment(const GlobalTensor<T>& scanGm, const LocalTensor<T>& queryXBuf,
                                            const LocalTensor<T>& queryYBuf, const LocalTensor<float>& bestBuf,
                                            const LocalTensor<int32_t>& bestIdxBuf, int64_t bIdx, int64_t k0,
                                            int64_t k1);
    __aicore__ inline void ReduceChunk(int64_t len, float& segMin, int32_t& segIdx);
    __aicore__ inline void FlushOut(const GlobalTensor<T>& distGm, const GlobalTensor<int32_t>& idxGm, int64_t taskBase,
                                    int64_t count);

    __aicore__ inline int64_t AlignUp(int64_t x, int64_t base) const { return (x + base - 1) / base * base; }

    TPipe* pipe_ = nullptr;
    GlobalTensor<T> xyz1Gm_;
    GlobalTensor<T> xyz2Gm_;
    GlobalTensor<T> dist1Gm_;
    GlobalTensor<T> dist2Gm_;
    GlobalTensor<int32_t> idx1Gm_;
    GlobalTensor<int32_t> idx2Gm_;

    // x/y 分成两块独立缓冲: 放同一块里用元素偏移取 y, 在 fp16/bf16 下偏移只有 VL 的一半字节数,
    // 不满足向量访存对齐, 真机报 VEC_ERROR。
    TQue<TPosition::VECIN, 1> scanXQue_; // 被查段的 x(原始 dtype, fp16/bf16 在 VF 内随路转 fp32)
    TQue<TPosition::VECIN, 1> scanYQue_; // 被查段的 y(同上)
    // 查询点 x/y 也必须分成两块: 同一块里用元素偏移取 y, 在 fp16/bf16 下偏移只有 16B,
    // 不满足 UB 搬运的 32B 对齐(与 scan 缓冲同源的坑)
    TBuf<TPosition::VECCALC> queryXBuf_;
    TBuf<TPosition::VECCALC> queryYBuf_; // 当前输出块的查询点 x/y(原始 dtype)
    TBuf<TPosition::VECCALC> distBuf_;   // 段内距离(fp32)
    TBuf<TPosition::VECCALC> workBuf_;   // ReduceMin 的 work tensor
    TBuf<TPosition::VECCALC> redBuf_;    // ReduceMin 的输出(值 + 下标)
    LocalTensor<T> scanX_;               // 当前候选块(tile 内复用, 用完再 FreeScan)
    LocalTensor<T> scanY_;
    TBuf<TPosition::VECCALC> bestBuf_;    // 查询 tile 内每点的跨段最小值
    TBuf<TPosition::VECCALC> bestIdxBuf_; // 对应下标
    TQue<TPosition::VECOUT, 1> outDistQue_;
    TQue<TPosition::VECOUT, 1> outIdxQue_;

    int64_t b_ = 0;
    int64_t n_ = 0;
    int64_t planeSize_ = 0; // B*N, x 平面与 y 平面的间距
    int64_t taskNum_ = 0;
    int64_t colsPerChunk_ = 0;
    int64_t chunkNum_ = 0;
    int64_t startTask_ = 0;
    int64_t endTask_ = 0;
    int64_t outTileLen_ = 0;

protected:
    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
    // 查询点分块大小(32B 的整数倍; 决定候选块的复用次数)
    constexpr static int64_t QUERY_TILE = 64;
    // 跨段最小值的初值必须是 +inf, 不能用 3.4e38 之类的"极大值"哨兵:
    // 坐标取到 fp32 极值域时真实距离会溢出成 inf, 而 `inf < 3.4e38` 为假 → 哨兵反而赢过真实值,
    // 输出 3.4e38 而不是 inf(真机实测过的缺陷)。
    constexpr static float DIST_INIT_VALUE = __builtin_inff();
};

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::Init(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist1, GM_ADDR dist2,
                                                  GM_ADDR idx1, GM_ADDR idx2,
                                                  const ChamferDistanceArch35TilingData* tiling, TPipe* pipe)
{
    pipe_ = pipe;
    b_ = tiling->b;
    n_ = tiling->n;
    planeSize_ = b_ * n_;
    taskNum_ = tiling->taskNum;
    colsPerChunk_ = tiling->colsPerChunk;
    chunkNum_ = tiling->chunkNum;

    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    startTask_ = blockIdx * tiling->tasksPerCore;
    int64_t myTasks = (blockIdx == tiling->realCoreNum - 1) ? tiling->tailTasks : tiling->tasksPerCore;
    endTask_ = startTask_ + myTasks;

    xyz1Gm_.SetGlobalBuffer((__gm__ T*)xyz1);
    xyz2Gm_.SetGlobalBuffer((__gm__ T*)xyz2);
    dist1Gm_.SetGlobalBuffer((__gm__ T*)dist1);
    dist2Gm_.SetGlobalBuffer((__gm__ T*)dist2);
    idx1Gm_.SetGlobalBuffer((__gm__ int32_t*)idx1);
    idx2Gm_.SetGlobalBuffer((__gm__ int32_t*)idx2);

    // 查询点分块: 候选块从 GM 搬一次要服务整个查询 tile(见 Process 的循环次序),
    // tile 越大 GM 访存越省 —— 原实现每个查询点都重搬一遍全部候选, 访存量是 O(查询数×候选数)。
    // 仍保持 32B 的整数倍, 输出按整块落盘。
    outTileLen_ = static_cast<int64_t>(QUERY_TILE);
    int64_t chunkAlign = AlignUp(colsPerChunk_, static_cast<int64_t>(VL_FP32));
    pipe_->InitBuffer(scanXQue_, 1, static_cast<uint32_t>(chunkAlign * sizeof(T)));
    pipe_->InitBuffer(scanYQue_, 1, static_cast<uint32_t>(chunkAlign * sizeof(T)));
    pipe_->InitBuffer(distBuf_, static_cast<uint32_t>(chunkAlign * sizeof(float)));
    pipe_->InitBuffer(workBuf_, static_cast<uint32_t>(chunkAlign * sizeof(float)));
    pipe_->InitBuffer(queryXBuf_, static_cast<uint32_t>(outTileLen_ * sizeof(T)));
    pipe_->InitBuffer(queryYBuf_, static_cast<uint32_t>(outTileLen_ * sizeof(T)));
    pipe_->InitBuffer(redBuf_, BLOCK_SIZE);
    // 跨候选块的逐查询点最小值/下标累加器(tile 内每个查询点各一份)
    pipe_->InitBuffer(bestBuf_, static_cast<uint32_t>(outTileLen_ * sizeof(float)));
    pipe_->InitBuffer(bestIdxBuf_, static_cast<uint32_t>(outTileLen_ * sizeof(int32_t)));
    pipe_->InitBuffer(outDistQue_, 1, static_cast<uint32_t>(outTileLen_ * sizeof(T)));
    pipe_->InitBuffer(outIdxQue_, 1, static_cast<uint32_t>(outTileLen_ * sizeof(int32_t)));
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::Process()
{
    if (taskNum_ <= 0 || n_ <= 0 || startTask_ >= endTask_) {
        return;
    }
    RunDirection(xyz1Gm_, xyz2Gm_, dist1Gm_, idx1Gm_); // set1 -> set2
    RunDirection(xyz2Gm_, xyz1Gm_, dist2Gm_, idx2Gm_); // set2 -> set1
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::CopyInQuery(const GlobalTensor<T>& queryGm, int64_t taskBase, int64_t len)
{
    LocalTensor<T> xBuf = queryXBuf_.template Get<T>();
    LocalTensor<T> yBuf = queryYBuf_.template Get<T>();
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(len * sizeof(T));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> pad;
    pad.isPad = true;
    pad.leftPadding = 0;
    pad.rightPadding = 0;
    pad.paddingValue = ScalarFromFloat<T>(0.0f);

    // x 平面与 y 平面在 GM 中相隔 B*N 个元素(布局 (2, B, N))
    AscendC::DataCopyPad(xBuf, queryGm[taskBase], params, pad);
    AscendC::DataCopyPad(yBuf, queryGm[planeSize_ + taskBase], params, pad);
    // 随后要用标量 GetValue 读这两块 UB, 只需 MTE2→S 一个方向的同步
    event_t eventMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    AscendC::SetFlag<HardEvent::MTE2_S>(eventMte2ToS);
    AscendC::WaitFlag<HardEvent::MTE2_S>(eventMte2ToS);
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::CopyInScan(const GlobalTensor<T>& scanGm, int64_t base, int64_t len)
{
    LocalTensor<T> xBuf = scanXQue_.template AllocTensor<T>();
    LocalTensor<T> yBuf = scanYQue_.template AllocTensor<T>();

    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = static_cast<uint32_t>(len * sizeof(T));
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> pad;
    pad.isPad = true;
    pad.leftPadding = 0;
    pad.rightPadding = static_cast<uint8_t>(AlignUp(len, static_cast<int64_t>(BLOCK_SIZE / sizeof(T))) - len);
    pad.paddingValue = ScalarFromFloat<T>(0.0f);

    // x 平面与 y 平面在 GM 中相隔 B*N 个元素(布局 (2, B, N))
    AscendC::DataCopyPad(xBuf, scanGm[base], params, pad);
    AscendC::DataCopyPad(yBuf, scanGm[planeSize_ + base], params, pad);
    scanXQue_.template EnQue<T>(xBuf);
    scanYQue_.template EnQue<T>(yBuf);
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::AcquireScan()
{
    scanX_ = scanXQue_.template DeQue<T>();
    scanY_ = scanYQue_.template DeQue<T>();
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::FreeScan()
{
    scanXQue_.FreeTensor(scanX_);
    scanYQue_.FreeTensor(scanY_);
}

// 用已驻留的候选块算一个查询点的整段距离(不动缓冲, 供 tile 内多个查询点复用)
template <typename T>
__aicore__ inline void ChamferDistanceND<T>::CalcDistVecKeep(int64_t len, float x1, float y1)
{
    LocalTensor<float> distBuf = distBuf_.template Get<float>();
    uint32_t fp32Lane = VL_FP32;
    uint16_t repeatTimes = static_cast<uint16_t>((len + fp32Lane - 1) / fp32Lane);
    // fp16/bf16 的 cast 在 VF 内随路做(DIST_UNPACK_B16 + Cast), 不再整段预转、也不需要 PIPE_V
    ChamferDistVF<T>((__ubuf__ float*)distBuf.GetPhyAddr(), (__ubuf__ T*)scanX_.GetPhyAddr(),
                     (__ubuf__ T*)scanY_.GetPhyAddr(), -x1, -y1, static_cast<uint32_t>(len), fp32Lane, repeatTimes);
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::CalcDistVec(int64_t len, float x1, float y1)
{
    AcquireScan();
    CalcDistVecKeep(len, x1, y1);
    FreeScan();
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::ReduceChunk(int64_t len, float& segMin, int32_t& segIdx)
{
    LocalTensor<float> distBuf = distBuf_.template Get<float>();
    LocalTensor<float> workBuf = workBuf_.template Get<float>();
    LocalTensor<float> redBuf = redBuf_.template Get<float>();

    // calIndex=true: redBuf[0] 为最小值, redBuf[1] 为其下标(按位存放, 需重解释为整数)
    AscendC::ReduceMin<float>(redBuf, distBuf, workBuf, static_cast<int32_t>(len), true);
    // 归约结果由标量读出, 是 S 等 V, 不是 V 等 V
    event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    AscendC::SetFlag<HardEvent::V_S>(eventVToS);
    AscendC::WaitFlag<HardEvent::V_S>(eventVToS);
    segMin = redBuf.GetValue(0);
    segIdx = static_cast<int32_t>(redBuf.template ReinterpretCast<uint32_t>().GetValue(1));
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::FlushOut(const GlobalTensor<T>& distGm, const GlobalTensor<int32_t>& idxGm,
                                                      int64_t taskBase, int64_t count)
{
    LocalTensor<T> outDist = outDistQue_.template DeQue<T>();
    LocalTensor<int32_t> outIdx = outIdxQue_.template DeQue<int32_t>();

    DataCopyExtParams distParams;
    distParams.blockCount = 1;
    distParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
    distParams.srcStride = 0;
    distParams.dstStride = 0;
    AscendC::DataCopyPad(distGm[taskBase], outDist, distParams);

    DataCopyExtParams idxParams = distParams;
    idxParams.blockLen = static_cast<uint32_t>(count * sizeof(int32_t));
    AscendC::DataCopyPad(idxGm[taskBase], outIdx, idxParams);

    outDistQue_.FreeTensor(outDist);
    outIdxQue_.FreeTensor(outIdx);
}

// 同一 batch 内: 逐候选块搬入(只搬一次), 块内服务 [k0, k1) 这些查询点, 更新各自的跨段最小值
template <typename T>
__aicore__ inline void ChamferDistanceND<T>::ScanBatchSegment(
    const GlobalTensor<T>& scanGm, const LocalTensor<T>& queryXBuf, const LocalTensor<T>& queryYBuf,
    const LocalTensor<float>& bestBuf, const LocalTensor<int32_t>& bestIdxBuf, int64_t bIdx, int64_t k0, int64_t k1)
{
    for (int64_t c = 0; c < chunkNum_; ++c) {
        int64_t colOff = c * colsPerChunk_;
        int64_t len = (n_ - colOff) < colsPerChunk_ ? (n_ - colOff) : colsPerChunk_;
        CopyInScan(scanGm, bIdx * n_ + colOff, len);
        AcquireScan();
        for (int64_t k = k0; k < k1; ++k) {
            float x1 = ScalarToFloat<T>(queryXBuf.GetValue(k));
            float y1 = ScalarToFloat<T>(queryYBuf.GetValue(k));
            CalcDistVecKeep(len, x1, y1);
            float segMin = DIST_INIT_VALUE;
            int32_t segIdx = 0;
            ReduceChunk(len, segMin, segIdx);
            // 严格小于: 并列时保留更早的段, 与"取最小下标"一致
            if (segMin < bestBuf.GetValue(k)) {
                bestBuf.SetValue(k, segMin);
                bestIdxBuf.SetValue(k, static_cast<int32_t>(colOff) + segIdx);
            }
        }
        FreeScan();
    }
}

template <typename T>
__aicore__ inline void ChamferDistanceND<T>::RunDirection(const GlobalTensor<T>& queryGm, const GlobalTensor<T>& scanGm,
                                                          const GlobalTensor<T>& distGm,
                                                          const GlobalTensor<int32_t>& idxGm)
{
    LocalTensor<T> queryXBuf = queryXBuf_.template Get<T>();
    LocalTensor<T> queryYBuf = queryYBuf_.template Get<T>();
    int64_t task = startTask_;
    while (task < endTask_) {
        int64_t tileLen = (endTask_ - task) < outTileLen_ ? (endTask_ - task) : outTileLen_;
        CopyInQuery(queryGm, task, tileLen);

        LocalTensor<T> outDist = outDistQue_.template AllocTensor<T>();
        LocalTensor<int32_t> outIdx = outIdxQue_.template AllocTensor<int32_t>();
        // ⚠️ 循环次序: 候选块在外、查询点在内。候选块从 GM 搬一次即服务 tile 内全部查询点,
        // GM 访存量从 O(查询数×候选数) 降到 O(候选数×块数)。反过来写(查询点在外)会让每个
        // 查询点都重搬一遍全部候选 —— 那正是大规模形状上落后竞品的原因。
        LocalTensor<float> bestBuf = bestBuf_.template Get<float>();
        LocalTensor<int32_t> bestIdxBuf = bestIdxBuf_.template Get<int32_t>();
        for (int64_t k = 0; k < tileLen; ++k) {
            bestBuf.SetValue(k, DIST_INIT_VALUE);
            bestIdxBuf.SetValue(k, 0);
        }
        // tile 内的查询点可能跨 batch(taskNum = B*N), 逐 batch 段处理以保证候选集合正确
        int64_t k0 = 0;
        while (k0 < tileLen) {
            int64_t bIdx = (task + k0) / n_;
            int64_t k1 = k0;
            while (k1 < tileLen && (task + k1) / n_ == bIdx) {
                ++k1;
            }
            ScanBatchSegment(scanGm, queryXBuf, queryYBuf, bestBuf, bestIdxBuf, bIdx, k0, k1);
            k0 = k1;
        }
        for (int64_t k = 0; k < tileLen; ++k) {
            outDist.SetValue(k, ScalarFromFloat<T>(bestBuf.GetValue(k)));
            outIdx.SetValue(k, bestIdxBuf.GetValue(k));
        }
        outDistQue_.template EnQue<T>(outDist);
        outIdxQue_.template EnQue<int32_t>(outIdx);
        FlushOut(distGm, idxGm, task, tileLen);
        task += tileLen;
    }
}

} // namespace ChamferDistance

#endif // CHAMFER_DISTANCE_ND_H
