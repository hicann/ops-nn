# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# aclnn_exclude: this operator adds no public aclnn interface. It is reached from aclnnLSTM, which
# lives in rnn/dynamic_rnn/op_api/ and dispatches here by SoC and capability, so no aclnnSingleLayerLstm
# may be generated from the OpDef. op_api/single_layer_lstm.cpp -- the L0 wrapper aclnnLSTM calls -- is
# compiled anyway.
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
add_all_modules_sources(
    OPTYPE single_layer_lstm
    ACLNNTYPE aclnn_exclude
    COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT}
    TILING_DIR ${SUPPORT_TILING_DIR}
)

# OPTIONS is not a tuning knob. The pair below ASSERTS AIC:AIV == 2, i.e. it is part of the kernel
# BEING a MIX 1:2 kernel, and it is bound to the KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
# line in arch35/single_layer_lstm.cpp -- change one and the other must change with it.
#
# Both directions of a mismatch fail SILENTLY WITH WRONG NUMBERS, not with an error:
#   1_2 entry without the macros -> the L0C -> UB drain no-ops, and the UB -> L1 feedback degrades
#                                   to a GM round trip plus a cross-core message
#   1_1 entry with the macros    -> equally wrong
# Measured both ways on a MIX 1:2 kernel: with the macros bad=0, without them bad=26949/32768.
add_kernel_sources(
    KERNEL_SRC arch35/single_layer_lstm.cpp
    COMPUTE_UNITS ascend950
    OPTIONS
        -DENABLE_CV_COMM_VIA_SSBUF=1
        -D__MIX_CORE_AIC_RATION__=2
)
