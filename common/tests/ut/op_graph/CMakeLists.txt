# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_GRAPH_UT)
    if(NOT TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
        add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
    endif()
    target_sources(${OP_GRAPH_MODULE_NAME}_cases_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/test_bias_optimize_quant_rollback_base.cpp)
endif()
