# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

message(STATUS "=== Debug: start ops.norm.add_layer_norm_quant.graph_plugin.CMakeLists.txt ")
add_graph_plugin_sources()

# Operator prototypes used to construct fusion patterns.
target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE
    ${PROJECT_SOURCE_DIR}/quant/dynamic_quant/op_graph/dynamic_quant_proto.h
    ${PROJECT_SOURCE_DIR}/quant/quantize/op_graph/quantize_proto.h
    ${PROJECT_SOURCE_DIR}/quant/ascend_quant_v2/op_graph/ascend_quant_v2_proto.h
)
