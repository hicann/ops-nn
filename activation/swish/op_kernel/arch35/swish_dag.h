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
 * \file swish_dag.h
 * \brief swish_dag
 */

#ifndef SWISH_DAG_H
#define SWISH_DAG_H
#include "atvoss/util/dag.h"
#include "atvoss/util/vec.h"
#include "atvoss/util/placeholder.h"
using namespace Ops::Base;

 namespace SwishDag {
#ifdef __CCE_AICORE__
    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };
    constexpr static AscendC::MicroAPI::CastTrait castTrait1 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
#endif
     const int PLACEHOLDER_INDEX_0 = 0;

     template<class T>
     struct SwishNegOneDagCalc : public Vec::ElemwiseBinaryOP<T, T, float> {
         __aicore__ inline SwishNegOneDagCalc(LocalTensor<T>& dst, LocalTensor<T>& src1, float scale, uint32_t count) {
     #ifdef __CCE_AICORE__
             uint32_t dtypeSize = sizeof(float);
             constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
             uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
             uint32_t loopNum = (count + vl - 1) / vl;
             uint32_t vlSize = vl;

             __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
             __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInputfloat;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> expResult;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> addsResult;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> divResult;
             AscendC::MicroAPI::MaskReg mask;

             if constexpr (std::is_same_v<T, float>) {
                 __VEC_SCOPE__ {
                      for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                          mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);
                          AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregInputfloat, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                          AscendC::MicroAPI::Exp(expResult, vregInputfloat, mask);
                          AscendC::MicroAPI::Adds(addsResult, expResult, 1, mask);
                          AscendC::MicroAPI::Div(divResult, vregInputfloat, addsResult, mask);
                          AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), divResult, mask);
                      }
                  }
             } else {
                 AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregInputT;
                 AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> divResultT;
                 __VEC_SCOPE__ {
                     // 不需要cast
                     for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                         mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);
                         AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInputT, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                         AscendC::MicroAPI::Cast<float, T, castTrait0>(vregInputfloat, vregInputT, mask);
                         AscendC::MicroAPI::Exp(expResult, vregInputfloat, mask);
                         AscendC::MicroAPI::Adds(addsResult, expResult, 1, mask);
                         AscendC::MicroAPI::Div(divResult, vregInputfloat, addsResult, mask);
                         AscendC::MicroAPI::Cast<T, float, castTrait1>(divResultT, divResult, mask);
                         AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), divResultT, mask);
                     }
                 }
             }
     #endif
         }
     };

     template<class T>
     struct SwishCalc : public Vec::ElemwiseBinaryOP<T, T, float> {
         __aicore__ inline SwishCalc(LocalTensor<T>& dst, LocalTensor<T>& src1, float scale, uint32_t count) {
     #ifdef __CCE_AICORE__
             uint32_t dtypeSize = sizeof(float);
             constexpr uint64_t VECTOR_REG_WIDTH = 256UL;
             constexpr float NEGATIVE_ONE = -1;
             uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
             uint32_t loopNum = (count + vl - 1) / vl;
             uint32_t vlSize = vl;

             __ubuf__ T* src1Addr = (__ubuf__ T*)src1.GetPhyAddr();
             __ubuf__ T* dstAddr = (__ubuf__ T*)dst.GetPhyAddr();

             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> vregInputfloat;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> MulsResult;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> expResult;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> addsResult;
             AscendC::MicroAPI::RegTensor<float, AscendC::MicroAPI::RegTraitNumOne> divResult;
             AscendC::MicroAPI::MaskReg mask;

             if constexpr (std::is_same_v<T, float>) {
                __VEC_SCOPE__ {
                     for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                         mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);
                         AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_NORM>(vregInputfloat, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                         AscendC::MicroAPI::Muls(MulsResult, vregInputfloat, NEGATIVE_ONE, mask);
                         AscendC::MicroAPI::Muls(MulsResult, MulsResult, scale, mask);
                         AscendC::MicroAPI::Exp(expResult, MulsResult, mask);
                         AscendC::MicroAPI::Adds(addsResult, expResult, 1, mask);
                         AscendC::MicroAPI::Div(divResult, vregInputfloat, addsResult, mask);
                         AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), divResult, mask);
                     }
                 }
             } else {
                 AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> vregInputT;
                 AscendC::MicroAPI::RegTensor<T, AscendC::MicroAPI::RegTraitNumOne> divResultT;
                 __VEC_SCOPE__ {
                     // 不需要cast
                     for (uint16_t loopIdx = 0; loopIdx < static_cast<uint16_t>(loopNum); loopIdx++) {
                         mask = AscendC::MicroAPI::UpdateMask<float, AscendC::MicroAPI::RegTraitNumOne>(count);
                         AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregInputT, (__ubuf__ T*)(src1Addr + loopIdx * vlSize));
                         AscendC::MicroAPI::Cast<float, T, castTrait0>(vregInputfloat, vregInputT, mask);
                         AscendC::MicroAPI::Muls(MulsResult, vregInputfloat, NEGATIVE_ONE, mask);
                         AscendC::MicroAPI::Muls(MulsResult, MulsResult, scale, mask);
                         AscendC::MicroAPI::Exp(expResult, MulsResult, mask);
                         AscendC::MicroAPI::Adds(addsResult, expResult, 1, mask);
                         AscendC::MicroAPI::Div(divResult, vregInputfloat, addsResult, mask);
                         AscendC::MicroAPI::Cast<T, float, castTrait1>(divResultT, divResult, mask);
                         AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>((__ubuf__ T*)(dstAddr + loopIdx * vlSize), divResultT, mask);
                     }
                 }
             }
     #endif
         }
     };

     template <typename T>
     struct SwishNegOne{
         // scale输入为-1场景
         // 数据搬入
         using InputX1T = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

         // 计算
         using OpResult = Bind<SwishNegOneDagCalc<T>, InputX1T, Placeholder::Var<float, 0>>;

         // Copy out
         using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

         using Outputs = Elems<OpCopyOut>;
         using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
         using OpDag = DAGSch<Outputs, void, MemCfg>;
     };
     template <typename T>
     struct SwishZero{
         // scale输入为0场景
         // 数据搬入
         using InputX1T = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;
         // cast
         using InputX1 = Bind<Vec::Cast<float, T, 0>, InputX1T>;
         // 计算
         using ConstValue = MAKE_CONST(float, 0.5);
         using OpResult = Bind<Vec::Muls<float>, InputX1, ConstValue>;
         using OpResultCast = Bind<Vec::Cast<T, float, 1>, OpResult>;

         // Copy out
         using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResultCast>;

         using Outputs = Elems<OpCopyOut>;
         using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
         using OpDag = DAGSch<Outputs, void, MemCfg>;
     };
     template <typename T>
     struct SwishOther{
         // scale输入为其他数字场景
         // 数据搬入
         using InputX1T = Bind<Vec::CopyIn<T>, Placeholder::In0<T>>;

         // 计算
         using OpResult = Bind<SwishCalc<T>, InputX1T, Placeholder::Var<float, 0>>;

         // Copy out
         using OpCopyOut = Bind<Vec::CopyOut<T>, Placeholder::Out0<T>, OpResult>;

         using Outputs = Elems<OpCopyOut>;
         using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
         using OpDag = DAGSch<Outputs, void, MemCfg>;
     };
}

 #endif // SWISH_DAG_H
