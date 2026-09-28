# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# The compile options are NOT optional and NOT a tuning knob: the pair asserts AIC:AIV == 2, matching
# the KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) line in the kernel entry and the OPTIONS in
# op_kernel/CMakeLists.txt. Without them the L0C -> UB drain no-ops and the UB -> L1 feedback degrades
# to a GM round trip, so the kernel under test would not be the kernel that ships.
if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    # -DSINGLE_LAYER_LSTM_ONLY_PHASE_A=1 keeps the registered cases to what the CPU simulator can attribute: it
    # faults inside its own predicate-register model on the range-split tanh, so phases B and C are
    # scored on hardware instead. See the note at the top of test_single_layer_lstm.cpp.
    # The test fills TilingData by calling the host pickers directly -- those are header only, so no
    # tiling translation unit is needed here.
    # -DDTYPE_W names the caller's element width, the same way the ops packaging does for each entry
    # of config/ascend950/single_layer_lstm_binary.json. The kernel has no default for it on purpose:
    # a build that omits it must fail to compile rather than silently produce an fp32 kernel behind
    # an fp16 signature. float here because this case is scored against an fp32 oracle; the fp16 and
    # bf16 paths are scored on hardware through tests/st.
    AddOpTestCase(single_layer_lstm "ascend950pr_9599"
                  "-DENABLE_CV_COMM_VIA_SSBUF=1 -D__MIX_CORE_AIC_RATION__=2 -DSINGLE_LAYER_LSTM_ONLY_PHASE_A=1 -DDTYPE_W=float")
endif()
