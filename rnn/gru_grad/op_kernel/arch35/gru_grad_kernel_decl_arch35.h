/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_kernel_decl_arch35.h
 * \brief GRU 反向 Kernel（dav_3510）类声明：常量、VF 前向声明、GGemmPlan、
 * GruGradKernel 成员声明与数据成员。成员定义见各 gru_grad_*_arch35.h 实现文件
 * （经 gru_grad_arch35.h 聚合 include）。
 */
#ifndef _GRU_GRAD_KERNEL_DECL_ARCH35_
#define _GRU_GRAD_KERNEL_DECL_ARCH35_

#include "kernel_operator.h"
#include "../gru_grad_tiling_data.h" // 公用 TilingData（struct GruGradTilingData + CutBatchTiling，本实现全量消费）
#include "gru_grad_sync.h"
#include "gru_grad_cube.h"
using namespace AscendC;
constexpr int64_t GRU_GATE_SIZE = 3;
constexpr int64_t GATE_IDX_RESET = 0;  // [r|z|n] staging 列区首门：reset 门
constexpr int64_t GATE_IDX_UPDATE = 1; // update 门
constexpr int64_t GATE_IDX_NEW = 2;    // new 门
constexpr int64_t AIV_PER_AIC = 2;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t DEFAULT_UB_BUF_ELEMENTS = 4096;
constexpr int64_t L1_ALIGN_BYTES = 512;      // L1/L0C 分形粒度对齐（512B）
constexpr int64_t DG_RESIDENT_MIN_RATIO = 4; // dgate B 侧驻留率至少 1/4 才值得切换
constexpr int64_t ALIGN_32B_FP32 = 8;        // UB fp32 tile 行距对齐（元素）
constexpr int64_t ALIGN_32B_FP32_MASK = 7;
constexpr int64_t BLOCK_BYTES = 32;           // DataCopyPad UB 侧 stride 单位（32B 块）
constexpr int64_t UB_TRANS_BUF_BYTES = 16384; // 转置 pass 的 UB tile 预算（ubTmp）
constexpr int64_t TRANS_GROUP = 16;           // 转置 pass 的 TS 维分组
constexpr int64_t TRANS_STRIP_COLS = 256;     // 条带 run 源行数（staged [8×256]=2048 elem 单次 Gather）
constexpr int64_t TRANS_TILE_COLS = 8;    // 转置列 tile 宽（8 列 fp32 = 32B 块，子带/条带分配粒度）
constexpr int64_t BAND_NUM = 2;           // 带化路径的 N 向等分带数（带宽 hAb=H/2）
constexpr int64_t BAND_H_ALIGN = 128;     // 带化域 H 对齐门槛：hAb=H/2 须 %VL==0 且 >=VL
constexpr int64_t DBUF_SLOTS = 2;         // L1/L0C 双缓冲（乒乓）槽位数
constexpr int64_t RESIDENT_MIN_SLOTS = 2; // dgate B 侧驻留最少槽位（scratch 交替需成对）
constexpr int64_t PIPELINE_MAX_HIDDEN = 256; // 深度-1 流水域门：H 超此值 L1 预算超限（实证承重，勿放开）
constexpr int64_t FP16_BYTES = FLOAT_BYTES / 2; // fp16 元素字节数（sizeof 分派判据）
constexpr int64_t WNZ_STAGE_BUFFS = 4;          // wNz 段预装的 MTE2/MTE3 四级中转缓冲槽数
// ------------------------------------------------------------------------------------------------
// GateGradsVF —— 门梯度链 VF 寄存器级实现（同仓其他 RNN 类算子的
// __simd_vf__ + asc_vf_call；asc_vf_call 仅 __NPU_ARCH__==3510 展开，其他编译目标为空）。
// 数值与 基线 UB 链逐位一致（同 Sub/Mul 指令序、同操作数配对与舍入点）；
// 门控梯度链的 UB 访问块数显著降低，中间量驻留向量寄存器。
// ------------------------------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void GateGradsVF(__ubuf__ T* drpAddr,  // dst0: drp = dhn·hn·r·(1−r)
                                    __ubuf__ T* dzpAddr,  // dst1: dzp = A·(hp−n)·z
                                    __ubuf__ T* dhnAddr,  // dst2: dhn = A·(1−n²)
                                    __ubuf__ T* dghnAddr, // dst3: dGh_n = r·dhn
                                    __ubuf__ T* dhfhAddr, // dst4: dhFromH = gradH·z
                                    __ubuf__ T* gradHAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr, __ubuf__ T* hpAddr,
                                    __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t count, uint32_t oneRepeatSize,
                                    uint16_t repeatTimes);
// ------------------------------------------------------------------------------------------------
// GateGradsScatterVF —— 该计算链 + [r|z|n] staging 布局直写（消除 6 次 UB→UB MTE2
// 装配与 V→M2→M3 双同步，串行链收敛为单 V→M3）。前置条件 hAligned % oneRepeatSize == 0
// （整拍无尾掩码，StoreAlign 地址恒 32B 对齐）；不满足走 原装配路径。
// dGi/dGh staging 各按行写 [drp|dzp|dhn] / [drp|dzp|dGh_n]（drp/dzp 同值双平面双写）；
// dhFromH 就地覆写 hp（chunk 内先载后存、跨 chunk 地址带不相交）。
// ------------------------------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void GateGradsScatterVF(__ubuf__ T* dGiAddr,    // dGi staging [bRows, 3hA]
                                           __ubuf__ T* dGhAddr,    // dGh staging [bRows, 3hA]
                                           __ubuf__ T* dhfhAddr,   // dhFromH 平面（就地覆写 hp）
                                           __ubuf__ T* hpSaveAddr, // hp 保留平面（Tail 消费免 GM 重读）
                                           __ubuf__ T* gradHAddr, __ubuf__ T* zAddr, __ubuf__ T* nAddr,
                                           __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr, uint32_t bRows,
                                           uint32_t hAligned, uint32_t oneRepeatSize);
// ------------------------------------------------------------------------------------------------
// GateGradsBandVF —— 带化 VF：gradH 折叠 + 全宽 staging 带列直写。
// 相对 GateGradsScatterVF：
//  - 输入 +3：gemmRes（本带 M2 装载于 ubTmp3，带紧凑 [rows, hAb]）、dy（ubNDy 带切片）、
//    dhFromH(t+1)（换出 ubGate 带切片——乒乓换出恰存上步输出，Tail 预取在其后覆写）
//  - 折叠 2 Add：gradH = (gemmRes + dhFromH) + dy（与现行 Accumulate+LoadGradH 同序，
//    IEEE 加法交换律保证逐位一致）后接同 15 算子链
//  - staging 全宽 [rows, 3hA]，本带只写列区 [g*H + bandOff, +hAb)（带间列区不相交，
//    M3 仍全宽单次——免 12 次窄 M3 重构）
//  - dhFromH(t) 就地写 ubHPrev 带列（hp(t) 本带已消费）；hp 保留带列同 scatter
// ------------------------------------------------------------------------------------------------
template <typename T>
__simd_vf__ inline void GateGradsBandVF(__ubuf__ T* dGiAddr,     // dGi staging 全宽 [bRows, 3hA]
                                        __ubuf__ T* dGhAddr,     // dGh staging 全宽 [bRows, 3hA]
                                        __ubuf__ T* dhfhOutAddr, // dhFromH(t) 带写出（ubHPrev 带列就地）
                                        __ubuf__ T* hpSaveAddr,  // hp 保留带写（ubDHPrevFromH）
                                        __ubuf__ T* gemmResAddr, // gemmRes 带紧凑 [bRows, hAb]
                                        __ubuf__ T* dyAddr, __ubuf__ T* dhfhInAddr, __ubuf__ T* zAddr,
                                        __ubuf__ T* nAddr, __ubuf__ T* hpAddr, __ubuf__ T* hnAddr, __ubuf__ T* rAddr,
                                        uint32_t bRows, uint32_t hA, uint32_t hAb, uint32_t bandOff,
                                        uint32_t oneRepeatSize);
struct GGemmPlan {
    int64_t M{0};
    int64_t N{0};
    int64_t K{0};
    int64_t singleCoreM{0};
    int64_t singleCoreN{0};
    int64_t usedCoreNum{0};
    int64_t mCoreLoop{0};
    int64_t nCoreLoop{0};
    int64_t mStep{0}; // L0C 子 tile 步长（16 对齐，mStep*nStep <= 64K 元素）
    int64_t nStep{0};
    int64_t kChunk{0}; // L1 K 分块（c0 对齐）
    int64_t l1AElems{0};
    int64_t l1BElems{0};
    int64_t l0CElems{0};
};

template <typename DTYPE>
class GruGradKernel {
public:
    __aicore__ inline int64_t AlignUp512(int64_t bytes);

    __aicore__ inline GruGradKernel() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h, GM_ADDR output_h,
                                GM_ADDR reset_gate, GM_ADDR update_gate, GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy,
                                GM_ADDR dh, GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input,
                                GM_ADDR dw_hidden, GM_ADDR db_input, GM_ADDR db_hidden, const GruGradTilingData* tiling,
                                GM_ADDR workspace);

    __aicore__ inline void Process();

    TPipe pipe;

private:
    __aicore__ inline void PlanDgateResident();

    __aicore__ inline void PlanWNz();

    __aicore__ inline void PreloadDgateB();

    __aicore__ inline void RunDgateResident(GlobalTensor<DTYPE>& aGm, GlobalTensor<float>& cGm, int64_t mExt);

    __aicore__ inline void RunGemmTiles(const GGemmPlan& p, GlobalTensor<DTYPE>& aGm, int64_t aRowPitch,
                                        GlobalTensor<DTYPE>& bGm, int64_t bRowPitch, GlobalTensor<float>& cGm,
                                        int64_t cRowPitch, int64_t m0, int64_t n0, int64_t mExt, int64_t nExt,
                                        bool batched = false, bool pipelined = false);

    __aicore__ inline void RunDgateSliceGEMM(int64_t aBase, int64_t kFrom, int64_t kTo);

    __aicore__ inline int64_t VecRowStart(int64_t aivIdx);

    __aicore__ inline int64_t VecRowCount(int64_t aivIdx);

    __aicore__ inline void DgateStep(int64_t tIdx);

    __aicore__ inline void RunDxGemm();

    __aicore__ inline void RunDwGemm2D(const GGemmPlan& p, GlobalTensor<DTYPE>& aGm, int64_t aRowPitch,
                                       GlobalTensor<DTYPE>& bGm, int64_t bRowPitch, GlobalTensor<float>& cGm,
                                       int64_t cRowPitch);

    __aicore__ inline void RunDwGmms();

    __aicore__ inline void CubeProcess();

    __aicore__ inline void ReduceS1();

    __aicore__ inline void DcciStepSlices(int64_t tIdx);

    __aicore__ inline void WriteWNzImage();

    __aicore__ inline void VecProcess();

    __aicore__ inline void VecProcessFinalize();

    __aicore__ inline int64_t GetCompactRowOffset(int64_t seqIdx);

    __aicore__ inline bool GetCoreRows(int64_t& mStart, int64_t& mCnt);

    __aicore__ inline void SyncM2toV();

    __aicore__ inline void SyncVtoM3();

    __aicore__ inline void SyncVtoM2();

    __aicore__ inline void SyncM3toV();

    __aicore__ inline void SyncM3toM2();

    __aicore__ inline void SyncM2toM3();

    // ---- GM<->UB 行搬运（行距修复版）------------------------------------------------------------
    // 语义：把 GM 中以 gmStride（元素）为行距的 rows 行、每行 length 个 T 元素，
    //       搬到 UB 中以 rowAlign（元素，32B 对齐垫后行距）为行距的 tile。
    // fp16 时先落入本 buffer 后半的 half 暂存区（fp32 偏移 allocLength），再 Cast 回 fp32 布局保持。
    template <typename T>
    __aicore__ inline void CopyInRow(GlobalTensor<T>& gm, const LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                     int64_t length, int64_t rowAlign, int64_t gmStride);

    // UB（fp32 布局，行距 rowAlign 个 fp32 元素）-> GM（行距 gmStride 个 T 元素）。
    // fp16 时先在 ubTmp2 的 half 视图做位置保持 Cast，再按 half 行距搬运。
    template <typename T>
    __aicore__ inline void CopyOutRow(GlobalTensor<T>& gm, const LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                      int64_t length, int64_t gmStride, int64_t rowAlign, bool vSyncPreDone = false);

    __aicore__ inline void ReadHPrev(int64_t tIdx, int64_t mStart, int64_t bOff, int64_t bRows, int64_t hOff,
                                     int64_t hLen, int64_t hAligned, LocalTensor<float>& ub);

    __aicore__ inline void InitFastPathCache();

    __aicore__ inline bool DhPrevUbPath();

    __aicore__ inline void LoadGradH(int64_t tIdx, int64_t ghOff, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                     int64_t hAligned);

    __aicore__ inline void LoadGatesAndHPrev(int64_t tIdx, int64_t ghOff, int64_t mStart, int64_t bOff, int64_t bRows,
                                             int64_t hOff, int64_t hLen, int64_t hAligned);

    __aicore__ inline void ComputeDINew(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                        int64_t hAligned);

    __aicore__ inline void ComputeDReset(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                         int64_t hAligned);

    __aicore__ inline void ComputeDUpdate(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                          int64_t hAligned);

    __aicore__ inline void ComputeDGhNewAndDhFromH(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                                   int64_t gmGateStride, int64_t hAligned);

    __aicore__ inline void ProcessVectorHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart, int64_t bOff,
                                              int64_t bRows, int64_t ht, int64_t compactGateRow, int64_t gm3H,
                                              int64_t gmGateStride, int64_t hTiles);

    __aicore__ inline void StoreHPrevHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart, int64_t bOff, int64_t bRows,
                                           int64_t ht, int64_t compactGateRow, int64_t hTiles, bool hpInUb);

    __aicore__ inline void ProcessVector(int64_t tIdx);

    __aicore__ inline void ProcessVectorTail(int64_t tIdx);

    __aicore__ inline void PrefetchFallback(int64_t tIdx, bool isReverse, int64_t nextGhOff, int64_t mStart,
                                            int64_t mCnt, int64_t hAlignedFull);

    __aicore__ inline void ConsumePrefetchSwaps();

    __aicore__ inline void VecProcessBand();

    __aicore__ inline void InitDhPrev();

    __aicore__ inline bool GateGradsMergedOk(int64_t bRows, int64_t hLen, int64_t hAligned);

    __aicore__ inline void ComputeGateGradsMerged(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                                  int64_t hAligned);

    __aicore__ inline bool GateGradsScatterOk(int64_t bRows, int64_t hLen, int64_t hAligned);

    __aicore__ inline void ComputeGateGradsScatter(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                                   int64_t hAligned);

    __aicore__ inline void AccumulateDhPrev(int64_t tIdx);

    __aicore__ inline void StoreDhPrev();

    // ------------------------------------------------------------------------------------------------
    // TransposeGiGhT：dGi/dGh [TS,3H] -> dGiT/dGhT [3H,TS]（dW GEMM 的 A 侧，输入 dtype 存储）。
    // 16 行分组（TS 维）× 列逐根处理：
    //   对每个输出行 m（=源列 m）：CopyInRow 按列收集 rCnt 个元素（GM 行距 cols，length=1 逐元素
    //   fallback 路径）落 UB（行距 8 fp32 元素），再 CopyOutRow 按行散布到 dstGm[m*TS + r0..]。
    // 注意：写侧须按 dst tile 行基址取数——若按源行基址取数，落进 dst 的是源的
    // "行片段"而非"列"，dGiT 实为 dGi 的行拷贝（f2a ws 实证 dGiT[0:4]==dGi[0:4]），dW 两路 A 侧全错。
    // 每任务写完即对本块 DataCacheCleanAndInvalid（cv-sync 纪律 5：生产者侧刷新）。

    __aicore__ inline void InitTransGatherOffsets();

    __aicore__ inline void TransposeOnePlane(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                             int64_t cols, int64_t colStart, int64_t colEnd, int64_t localIdx,
                                             int64_t pairStride, int64_t dstRowPitch = 0);

    __aicore__ inline void TransposeGiGhT();

    __aicore__ inline void ConvertOnePlane(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                           int64_t cols, int64_t rowStart, int64_t rowEnd, int64_t localIdx,
                                           int64_t pairStride);

    __aicore__ inline void ConvertTransposedPlane(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm,
                                                  int64_t srcRows, int64_t srcCols, int64_t rowStart, int64_t rowEnd,
                                                  int64_t localIdx, int64_t pairStride);

    __aicore__ inline void ConvertOutputs();

    __aicore__ inline void ProcessBiasReduce(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                             int64_t cols);

    __aicore__ inline void InitWorkBuffers(__gm__ uint8_t* ws);

    __aicore__ inline void setD(GlobalTensor<DTYPE>& g, __gm__ uint8_t* base, int64_t off, int64_t elems);

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h,
                                             GM_ADDR output_h, GM_ADDR reset_gate, GM_ADDR update_gate,
                                             GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh, GM_ADDR batch_sizes,
                                             GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input, GM_ADDR dw_hidden,
                                             GM_ADDR db_input, GM_ADDR db_hidden, GM_ADDR workspace);

    __aicore__ inline void InitVectorBuf();

    __aicore__ inline int64_t AlignDown16(int64_t v);

    __aicore__ inline void SwapTilingMN(TCubeTiling& t);

    __aicore__ inline void BuildOnePlan(GGemmPlan& p, const TCubeTiling& t);

    __aicore__ inline void BuildPlans();

private:
    struct InputGm {
        __aicore__ inline InputGm() = default;
        GlobalTensor<DTYPE> xGm;
        GlobalTensor<DTYPE> wInputGm;
        GlobalTensor<DTYPE> wHiddenGm;
        GlobalTensor<DTYPE> outputHGm;
        GlobalTensor<DTYPE> dyGm;
        GlobalTensor<DTYPE> dhGm;
        GlobalTensor<DTYPE> initHGm;
        GlobalTensor<DTYPE> resetGateGm;
        GlobalTensor<DTYPE> updateGateGm;
        GlobalTensor<DTYPE> newGateGm;
        GlobalTensor<DTYPE> hNGm;
        GlobalTensor<int64_t> batchSizesGm;
    };
    struct OutputGm {
        __aicore__ inline OutputGm() = default;
        GlobalTensor<DTYPE> dxGm;
        GlobalTensor<DTYPE> dhPrevGm;
        GlobalTensor<DTYPE> dwInputGm;
        GlobalTensor<DTYPE> dwHiddenGm;
        GlobalTensor<DTYPE> dbInputGm;
        GlobalTensor<DTYPE> dbHiddenGm;
    };
    struct WorkGm {
        __aicore__ inline WorkGm() = default;
        GlobalTensor<DTYPE> dGhGm;
        GlobalTensor<DTYPE> dGiGm;
        GlobalTensor<DTYPE> hPrevWsGm;
        GlobalTensor<DTYPE> dhPrevWsGm;
        GlobalTensor<DTYPE> dhFromHGm;
    };

    InputGm inputGm;
    OutputGm outputGm;
    WorkGm workGm;
    // 扩展 workspace
    GlobalTensor<DTYPE> xTGm;
    GlobalTensor<DTYPE> hPrevTGm;
    GlobalTensor<float> dhPrevF32Gm;
    GlobalTensor<float> dwIF32Gm;
    GlobalTensor<float> dwHF32Gm;
    GlobalTensor<float> dxF32Gm;
    GlobalTensor<DTYPE> wNzGm;    // w_hh NZ 像预暂存段
    bool wNzGmValid_{false};      // host 已分配该段
    bool wNzUsable_{false};       // kernel 复核通过(writer/consumer 双方共同判据)
    GlobalTensor<float> s1PartGm; // dgate K 切分部分和段 [aicoreCnt, B, H]
    bool s1SegValid_{false};      // host 已分配该段
    bool s1Usable_{false};        // 复核通过(K 切片非空 + wNz 域)

    TBuf<TPosition::VECCALC> vbGradH, vbHPrev, vbReset, vbUpdate, vbNew, vbHN, vbDHPrevFromH;
    TBuf<TPosition::VECCALC> vbTmp, vbTmp2, vbTmp3;
    TBuf<TPosition::VECCALC> vbGate;
    // 门平面预取双缓冲(下一拍 r/z/n/hn/hPrev)
    TBuf<TPosition::VECCALC> vbNReset, vbNUpdate, vbNNew, vbNHN, vbNHPrev;
    LocalTensor<float> ubGradH, ubHPrev, ubReset, ubUpdate, ubNew, ubHN, ubDHPrevFromH;
    LocalTensor<float> ubTmp, ubTmp2, ubTmp3;
    LocalTensor<float> ubGate;
    LocalTensor<float> ubNReset, ubNUpdate, ubNNew, ubNHN, ubNHPrev;
    LocalTensor<float> ubNDy; // dy 预取(复用 ubNHPrev 槽位)
    // bias 累加器——ubGate 尾部别名视图（band 路径驻留安全区）
    // 无新 TBuf（VECCALC 池 248KB 已满），[2048, 2048+6hA) 区段与 hp 预取 [0, mCnt*hA) 无交叠
    bool gatesPrefetched{false}; // 门平面预取单次消费标志
    bool dyPrefetched{false};    // dy 预取独立标志（与 gatesPrefetched 解耦，容量门独立）
    // 快路径判定缓存（形状派生量 kernel 内恒定，VecProcess 入口一次算齐）
    bool coreRowsInit_{false}; // GetCoreRows 缓存有效位
    bool coreRowsOk_{false};
    int64_t mStart_{0};
    int64_t mCnt_{0};
    bool fastPathInit_{false};  // 谓词缓存有效位
    bool mergedShapeOk_{false}; // H%alignPara==0（merged/scatter 共同前置）
    bool scatterVOk_{false};    // hiddenSizeAligned % VL == 0（整拍约束）
    int64_t stagingCapRows_{0}; // allocLength / (3*hiddenSizeAligned)
    bool dhPrevUbPath_{false};  // DhPrevUbPath 全判定
    bool pfCapOk_{false};       // 预取容量/单tile判定
    bool pfDyCapOk_{false};     // dy 预取专属容量门（ubNDy 槽位仅 8KB，超域回退 GM 直读）
    bool hAlign8Ok_{false};     // H%alignPara==0（hp 预取门）
    bool vfMte2Dirty_{false};   // Accumulate 同步后是否有新 MTE2 喂 VF（scatter 据此条件同步）
    bool band0Early_{false};    // 本步已发 B0 提前信号（CubeProcess 补发配对）
    bool band1Early_{false};    // 本步已发 B1 末带提前信号（同上补发配对）
    bool bandPath_{false};      // 带化 AIV 主循环分派（InitFastPathCache 一次算齐）
    bool dhFromHInHp_{false};   // dhFromH 常驻缓冲位置（true=ubHPrev 合并路径）
    bool hpPreserved_{false};   // 本步 scatter 已将 hp 保留至 ubDHPrevFromH（Tail 消费）
    bool hpPrefetched{false};   // hp(t-1) 已预取至 ubGate（单次消费，乒乓换入）

    const GruGradTilingData* tiling{nullptr};
    int64_t T{0};
    int64_t B{0};
    int64_t H{0};
    int64_t I{0};
    int64_t vecBTile_{0};
    int64_t vecHTile_{0};
    int64_t allocLength{DEFAULT_UB_BUF_ELEMENTS};
    int64_t totalSteps_{0};
    int64_t alignPara{ALIGN_32B_FP32};
    int64_t alignMask{ALIGN_32B_FP32_MASK};

    GGemmPlan dgatePlan;
    GGemmPlan dwIhPlan;
    GGemmPlan dwHhPlan;
    GGemmPlan dxPlan;
    // dgate B 侧 L1 驻留规划（0 = 回退 原始）
    int64_t dgResident_{0};
    int64_t dgSlotBBytes_{0};
    int64_t dgSlotABytes_{0};
    int64_t dgChunkCnt_{0};
    int64_t dgTileCnt_{0};
    int64_t l1AElems_{0};
    int64_t l1BElems_{0};
    int64_t l0AElems_{0};
    int64_t l0BElems_{0};
    int64_t l0CElems_{0};
};

#endif // _GRU_GRAD_KERNEL_DECL_ARCH35_
