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
 * \file conv_bp_input_sub_func_sync.h
 * \brief Cross-core synchronization primitives for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_SYNC_H
#define CONV3D_BP_INPUT_SUB_FUNC_SYNC_H

#include "conv_bp_input_sub_func_utils.h"

namespace Convolution3DBackpropFunc {

template <class Intf, pipe_t srcPipe, pipe_t dstPipe>
__aicore__ inline void CvCrossCoreSet(Intf* self, uint8_t flagId)
{
#if __CUBE_VECTOR_FUSION_ONLY__
    AscendC::TQueSync<srcPipe, dstPipe> sync;
    sync.SetFlag(flagId);
#else
    CrossCoreSetFlag<SYNC_MODE, srcPipe>(flagId);
#endif
}

template <class Intf, pipe_t srcPipe, pipe_t dstPipe>
__aicore__ inline void CvCrossCoreWait(Intf* self, uint8_t flagId)
{
#if __CUBE_VECTOR_FUSION_ONLY__
    AscendC::TQueSync<srcPipe, dstPipe> sync;
    sync.WaitFlag(flagId);
#else
    CrossCoreWaitFlag<SYNC_MODE, dstPipe>(flagId);
#endif
}

template <class Intf>
__aicore__ inline void WaitForVecBeforeLoadToB2(Intf* self)
{
#ifndef __CCE_KT_TEST__
    CvCrossCoreWait<Intf, PIPE_MTE3, PIPE_MTE1>(self, FLAG_MTE1_ID_1);
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        CvCrossCoreWait<Intf, PIPE_MTE3, PIPE_MTE1>(self, FLAG_MTE1_ID_1 + CROSS_CORE_FLAG_ID_MAX);
    }
#endif
}

template <class Intf>
__aicore__ inline void NotifyVecAfterLoadToB2(Intf* self)
{
#ifndef __CCE_KT_TEST__
    CvCrossCoreSet<Intf, PIPE_MTE1, PIPE_V>(self, FLAG_MTE1_ID_2);
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        CvCrossCoreSet<Intf, PIPE_MTE1, PIPE_MTE3>(self, FLAG_MTE1_ID_2 + CROSS_CORE_FLAG_ID_MAX);
    }
#endif
}

template <class Intf>
__aicore__ inline void WaitForCubeBeforeLoadToB1(Intf* self)
{
#ifndef __CCE_KT_TEST__
    CvCrossCoreWait<Intf, PIPE_MTE1, PIPE_MTE3>(self, FLAG_MTE1_ID_2);
#endif
}

template <class Intf>
__aicore__ inline void NotifyCubeAfterLoadToB1(Intf* self)
{
#ifndef __CCE_KT_TEST__
    CvCrossCoreSet<Intf, PIPE_MTE3, PIPE_MTE1>(self, FLAG_MTE1_ID_1);
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreSetHead(Intf* self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIC_SCALAR {
        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            CvCrossCoreSet<Intf, PIPE_MTE1, PIPE_V>(self, FLAG_MTE1_ID_2);
#if __CUBE_VECTOR_FUSION_ONLY__
            if (self->ctx.tiling_->bl1Pbuffer > 1) {
                CvCrossCoreSet<Intf, PIPE_MTE1, PIPE_V>(self, FLAG_MTE1_ID_2);
            }
#endif
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreWaitTail(Intf* self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIV_SCALAR {
        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            if (GetSubBlockIdx() == 0) {
                CvCrossCoreWait<Intf, PIPE_MTE1, PIPE_V>(self, FLAG_MTE1_ID_2);
#if __CUBE_VECTOR_FUSION_ONLY__
                if (self->ctx.tiling_->bl1Pbuffer > 1) {
                    CvCrossCoreWait<Intf, PIPE_MTE1, PIPE_V>(self, FLAG_MTE1_ID_2);
                }
#endif
            }
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreSetHeadForMix(Intf* self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIV_SCALAR {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            CvCrossCoreSet<Intf, PIPE_MTE3, PIPE_FIX>(self, FLAG_FIXP_ID);
        } else if (self->ctx.enableSplitDk_) {
            CvCrossCoreSet<Intf, PIPE_MTE3, PIPE_FIX>(self, FLAG_FIXP_ID);
        }
    }
    if ASCEND_IS_AIC_SCALAR {
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            CvCrossCoreSet<Intf, PIPE_MTE1, PIPE_MTE3>(self, FLAG_MTE1_ID_2);
            CvCrossCoreSet<Intf, PIPE_MTE1, PIPE_MTE3>(self, FLAG_MTE1_ID_2 + CROSS_CORE_FLAG_ID_MAX);
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreWaitTailForMix(Intf* self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIC_SCALAR {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            CvCrossCoreWait<Intf, PIPE_MTE3, PIPE_FIX>(self, FLAG_FIXP_ID);
        } else if (self->ctx.enableSplitDk_) {
            CvCrossCoreWait<Intf, PIPE_MTE3, PIPE_FIX>(self, FLAG_FIXP_ID);
        }
    }
    if ASCEND_IS_AIV_SCALAR {
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            CvCrossCoreWait<Intf, PIPE_MTE1, PIPE_MTE3>(self, FLAG_MTE1_ID_2);
        }
    }
#endif
}

} // namespace Convolution3DBackpropFunc

#endif
