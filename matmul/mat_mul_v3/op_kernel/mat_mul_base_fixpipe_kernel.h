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
 * \file mat_mul_base_fixpipe_kernel.h
 * \brief Intermediate base class for fixpipe-based matmul kernels with AIC+AIV cooperation.
 *        Extracts common code from MatmulBaseVectorNz2NdKernel and MatmulBaseUnalignedNKernel.
 */
 #ifndef OP_KERNEL_MATMUL_V3_BASE_FIXPIPE_KERNEL_H
 #define OP_KERNEL_MATMUL_V3_BASE_FIXPIPE_KERNEL_H
 
 #include "mat_mul_base_kernel.h"
 #include "mat_mul_nz2nd.h"
 
 using namespace AscendC;
 using namespace matmul;
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
     const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
 class MatmulBaseFixPipeKernel : public MatmulBaseKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB> {
 public:
     using C_T = typename C_TYPE::T;
     __aicore__ inline MatmulBaseFixPipeKernel() {}
 
     __aicore__ inline void AicProcess(GlobalTensor<C_T>& cTensor, uint8_t enAtomic, bool aicNeedWaitAiv,
         uint8_t pingPongId);
     __aicore__ inline void UpdateOffsetC(bool isNd);
 
     // AicProcess 的两段公共实现，拆分点即派生类可注入差异（如切换 orgM）的位置：
     //   - AicLoadAndIterate: 设置 single shape、A/B/bias，执行 Iterate，并按需等待 AIV；
     //   - AicStoreC: 将结果写出（GetTensorC）并发起 AIC->AIV 同步。
     __aicore__ inline void AicLoadAndIterate(bool aicNeedWaitAiv, uint8_t pingPongId);
     __aicore__ inline void AicStoreC(GlobalTensor<C_T>& cTensor, uint8_t enAtomic, uint8_t pingPongId);
 
     // 计算 AIV 侧 NZ2ND 转换所需的公共块参数，返回 false 表示本核无需处理。
     __aicore__ inline bool PrepareNz2NdConversion(uint8_t pingPongId, LocalTensor<C_T>& tensorNZ,
         LocalTensor<C_T>& tensorND, uint64_t& ubProcessMNum, uint64_t& srcGmOffset, uint64_t& dstGmOffset);
 
     // AIV 侧 NZ2ND 的公共拷入与转换流程。
     __aicore__ inline void Nz2NdCopyAndConvert(GlobalTensor<C_T> cNzGlobal, LocalTensor<C_T> tensorNZ,
         LocalTensor<C_T> tensorND, uint64_t ubProcessMNum, uint64_t srcGmOffset, uint8_t pingPongId,
         uint64_t gatherNBlocks);
 
     // 公共的两级 tiling 主调度循环（AIC+AIV 混合）。DERIVED 为派生 kernel 类型，
     // 循环内通过 static_cast<DERIVED*> 直接调用派生类的 AicProcess 与 AivStep，
     template <class DERIVED>
     __aicore__ inline void RunFixpipeTileLoop(uint64_t index, uint8_t enAtomic);
 
 protected:
     GlobalTensor<C_T> tempCGlobal_;
     GatherMaskParams params_;
     uint64_t baseSize_ = 0UL;
     uint64_t alignedN_ = 0UL;
     uint64_t c0Size_ = BLOCK_SIZE;
 };
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AicLoadAndIterate(
     bool aicNeedWaitAiv, uint8_t pingPongId)
 {
     this->mm_.SetSingleShape(this->block_.params_.singleCoreM, this->block_.params_.singleCoreN,
                              this->block_.matmulTilingData_->matmulTiling.singleCoreK);
     this->mm_.SetTensorA(this->aGlobal_[this->block_.offset_.offsetA], A_TYPE::isTrans);
     this->mm_.SetTensorB(this->bGlobal_[this->block_.offset_.offsetB], B_TYPE::isTrans);
     if (this->block_.matmulTilingData_->matmulTiling.isBias) {
         this->mm_.SetBias(this->biasGlobal_[this->block_.offset_.offsetBias]);
     }
     this->mm_.Iterate();
     if (aicNeedWaitAiv) {
         CrossCoreWaitFlag(AIC_SYNC_AIV_FLAG + pingPongId);
     }
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AicStoreC(
     GlobalTensor<C_T>& cTensor, uint8_t enAtomic, uint8_t pingPongId)
 {
     this->mm_.GetTensorC(cTensor, enAtomic);
 #if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
     CrossCoreSetFlag<0x2, PIPE_FIX>(AIV_SYNC_AIC_FLAG + pingPongId);
 #endif
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AicProcess(
     GlobalTensor<typename C_TYPE::T>& cTensor, uint8_t enAtomic, bool aicNeedWaitAiv, uint8_t pingPongId)
 {
     if ASCEND_IS_AIC {
         this->AicLoadAndIterate(aicNeedWaitAiv, pingPongId);
         this->AicStoreC(cTensor, enAtomic, pingPongId);
     }
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::UpdateOffsetC(
     bool isNd)
 {
     Nz2NdUpdateOffsetC(this->block_.params_, this->block_.offset_, this->block_.matmulTilingData_, isNd);
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline bool MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
     MM_CB>::PrepareNz2NdConversion(uint8_t pingPongId, LocalTensor<C_T>& tensorNZ, LocalTensor<C_T>& tensorND,
     uint64_t& ubProcessMNum, uint64_t& srcGmOffset, uint64_t& dstGmOffset)
 {
     uint32_t vBlockIndex = GetBlockIdx();
     if (vBlockIndex >= (this->block_.matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
         return false;
     }
     CrossCoreWaitFlag(AIV_SYNC_AIC_FLAG + pingPongId);
     LocalTensor<C_T> ubTensor = this->ubBuf_.template Get<C_T>();
     uint64_t cDtypeSize = sizeof(C_T);
     ubProcessMNum = min(MMV3CeilAlign(this->block_.params_.singleCoreM / NUM_TWO, this->c0Size_),
         static_cast<uint64_t>(this->block_.params_.singleCoreM));
     int64_t subIdx = GetSubBlockIdx();
     srcGmOffset = 0UL;
     dstGmOffset = 0UL;
     if (subIdx == 1) {
         srcGmOffset = ubProcessMNum * ALIGNED_H;
         dstGmOffset = ubProcessMNum * this->block_.matmulTilingData_->matmulTiling.N;
         ubProcessMNum = this->block_.params_.singleCoreM - ubProcessMNum;
     }
     uint64_t ndOffset = (TOTAL_UB_SIZE >> 2) / cDtypeSize;
     uint64_t pingpongOffset = (TOTAL_UB_SIZE >> 1) / cDtypeSize;
     tensorNZ = ubTensor[pingPongId * pingpongOffset];
     tensorND = ubTensor[pingPongId * pingpongOffset + ndOffset];
     return true;
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
     MM_CB>::Nz2NdCopyAndConvert(GlobalTensor<C_T> cNzGlobal, LocalTensor<C_T> tensorNZ, LocalTensor<C_T> tensorND,
     uint64_t ubProcessMNum, uint64_t srcGmOffset, uint8_t pingPongId, uint64_t gatherNBlocks)
 {
     Nz2NdCopyInWithNz<C_T>(tensorNZ, cNzGlobal[srcGmOffset], ubProcessMNum, pingPongId,
         this->alignedN_, this->block_.matmulTilingData_->matmulTiling.baseM);
     CrossCoreSetFlag<0x2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + pingPongId);
     Nz2NdMulsAndGatherMask<C_T>(tensorND, tensorNZ, ubProcessMNum, pingPongId,
         this->alignedN_, this->c0Size_, gatherNBlocks, this->params_);
     this->UpdateOffsetC(true);
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 template <class DERIVED>
 __aicore__ inline void MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
     MM_CB>::RunFixpipeTileLoop(uint64_t index, uint8_t enAtomic)
 {
     int8_t pingPongId = 0;
     bool aicNeedWaitAiv = false;
     bool reverse = true;
     GlobalTensor<C_T> tempCGlobal = this->tempCGlobal_;
     for (uint64_t mTileIndex = 0; mTileIndex < this->block_.params_.mTileCntL2; mTileIndex++) {
         reverse = !reverse;
         for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < this->block_.params_.nTileCntL2; nTileIndexTemp++) {
             uint64_t nTileIndex = reverse ? (this->block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
             this->block_.UpdateBlockCnt(mTileIndex, nTileIndex);
             this->block_.InitBlockIndex(index);
             SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG));
             SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + 1));
             for (uint64_t j = 0; j < this->block_.params_.realRound; j++) {
                 tempCGlobal = this->tempCGlobal_[this->baseSize_ * (GetCurrentBlockIdx() * 2 + pingPongId)];
                 if (this->block_.params_.rowOrder == 0) {
                     this->block_.UpdateBasicIndex(j);
                 }
                 if (this->block_.params_.index < this->block_.params_.totalTileCnt) {
                     this->block_.UpdateBlockParams(mTileIndex, nTileIndex);
                     this->block_.template CalcGMOffset<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(mTileIndex, nTileIndex);
                     static_cast<DERIVED*>(this)->AicProcess(tempCGlobal, enAtomic, aicNeedWaitAiv, pingPongId);
                     static_cast<DERIVED*>(this)->AivStep(tempCGlobal, pingPongId);
                     aicNeedWaitAiv = aicNeedWaitAiv || bool(pingPongId);
                     pingPongId = (pingPongId + 1) & 1;
                 }
                 this->block_.UpdateBlockIndex();
             }
             WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG));
             WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + 1));
         }
     }
     if (this->block_.params_.isHf32) {
         this->mm_.SetHF32(false, 0);
     }
     PipeBarrier<PIPE_ALL>();
 }
 
 #endif
 