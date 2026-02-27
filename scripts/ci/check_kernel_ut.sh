#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

# 查找符合条件的二级目录：
#    1. 一级目录由cmake/variables.cmake中的OP_CATEGORY_LIST直接提供
#    2. 二级目录下需要包含/tests/ut/op_kernel文件夹
#    3. 构建出valid_dirs清单

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
pr_file=$(realpath "${1:-pr_filelist.txt}")
# 提取 OP_CATEGORY_LIST 的值
op_category_list=$(grep -oP 'set\(OP_CATEGORY_LIST\s*\K".*"' $current_dir/$variables_cmake | sed 's/"//g')
IFS=' ' read -r -a op_categories <<< "$op_category_list"

#op_categories=("activation" "conv" "foreach" "vfusion" "index" "loss" "matmul" "norm" "optim" "pooling" "quant" "rnn" "control")

valid_dirs=()
for category in "${op_categories[@]}"
do
    category_path="$current_dir/$category"
    if [ -d "$category_path" ]; then
        for dir in "$category_path"/*/
        do
            if [ -d "$dir/tests/ut/op_kernel" ]; then
                dir_name=$(basename "$dir")
                if [[ "$dir_name" != *"common"* ]]; then
                    valid_dirs+=("$dir_name")
                fi
            fi
        done
    fi
done


#搜索变更文件清单，将存在于valid_dirs清单中的算子保存到ops_name中，路径中需要出现'op_kernel'

ops_name=()
found_mirror_update=false
mapfile -t lines < ${pr_file}
for file_path in "${lines[@]}"
do
    # 去除前后空格
    file_path=$(echo "$file_path" | xargs)
    # 跳过空行
    if [ -z "$file_path" ]; then
        continue
    fi
    # 跳过 .md 后缀的文件
    if [[ "$file_path" == *.md ]]; then
        continue
    fi
    # 检查该路径是否包含 valid_dirs 中的任意一个目录名
    for dir in "${valid_dirs[@]}"
    do
        if [[ "$file_path" == *"/$dir/"* ]]; then
            if [[ "$file_path" == *"op_kernel"* ]]; then
                # 去重后添加到 ops_name 数组中
                if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                    ops_name+=("$dir")
                fi
            fi
            break
        fi
    done

    if [[ "$file_path" == "scripts/ci/mirror_update_time.txt" ]]; then 
        found_mirror_update=true;
    fi

done

supportedSocVersion=("ascend910b" "ascend310p" "ascend950")

for name in "${ops_name[@]}"
do
    for soc_version in "${supportedSocVersion[@]}"
    do
        echo "[EXECUTE_COMMAND] bash build.sh -u --opkernel --ops=$name --cov --soc=$soc_version"
        bash build.sh -u --opkernel --ops=$name --cov --soc=$soc_version --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?
        if [ $status -ne 0 ]; then
            echo "${name} kernel ut fail"
            exit 1
        fi
    done
done

#更新镜像验证全量算子UT
if [[ "$found_mirror_update" == "true" ]]; then
    foreach_ops_910b=("foreach_abs" "foreach_add_list")
    control_ops_910b=("assert")
    index_ops_910b=("embedding_dense_grad_v2" "gather_elements_v2" "index_put_v2" "scatter_elements_v2")
    vfusion_ops_910b=("scaled_masked_softmax_v2")
    rnn_ops_910b=("dynamic_rnn")
    pooling_ops_910b=("avg_pool3_d" "adaptive_max_pool3d")
    activation_ops_910b=("ge_glu_v2" "ge_glu_grad_v2")
    loss_ops_910b=("logit" "logit_grad")
    optim_ops_910b=("apply_adam_w_v2" "apply_fused_ema_adam")
    norm_ops_910b=("rms_norm" "add_layer_norm_grad")
    quant_ops_910b=("flat_quant")
    matmul_ops_910b=("addmv" "batch_mat_mul_v3" "fused_linear_cross_entropy_loss_grad" "fused_linear_online_max_sum" "fused_quant_mat_mul" "gemm" "mat_mul_v3" "mv" "quant_batch_matmul_v3" "weight_quant_batch_matmul_v2")
    conv_ops_910b=("conv2d_v2" "conv3d_v2")
    operator_list_910b=("${foreach_ops_910b[@]}" "${control_ops_910b[@]}" "${index_ops_910b[@]}" "${vfusion_ops_910b[@]}" "${rnn_ops_910b[@]}" "${pooling_ops_910b[@]}" "${activation_ops_910b[@]}" "${loss_ops_910b[@]}" "${optim_ops_910b[@]}" "${norm_ops_910b[@]}" "${quant_ops_910b[@]}" "${matmul_ops_910b[@]}" "${conv_ops_910b[@]}")

    foreach_ops_950=("foreach_abs" "foreach_add_list")
    control_ops_950=("assert")
    index_ops_950=("embedding_dense_grad_v2" "scatter_elements_v2")
    vfusion_ops_950=("scaled_masked_softmax_v2")
    rnn_ops_950=()
    pooling_ops_950=("avg_pool3_d" "max_pool_grad_with_argmax_v3")
    activation_ops_950=("ge_glu_grad_v2" "swi_glu")
    loss_ops_950=("logit" "logit_grad")
    optim_ops_950=("apply_adam_w_v2" "apply_fused_ema_adam")
    norm_ops_950=("add_layer_norm" "rms_norm_grad")
    quant_ops_950=("dequant_swiglu_quant" "dynamic_quant")
    matmul_ops_950=("mat_mul_v3" "quant_batch_matmul_v4" "weight_quant_batch_matmul_v2")
    conv_ops_950=("conv2d_v2" "conv3d_v2" "conv3d_backprop_filter_v2" "conv3d_backprop_input_v2" "conv3d_transpose_v2")
    operator_list_950=("${foreach_ops_950[@]}" "${control_ops_950[@]}" "${index_ops_950[@]}" "${vfusion_ops_950[@]}" "${rnn_ops_950[@]}" "${pooling_ops_950[@]}" "${activation_ops_950[@]}" "${loss_ops_950[@]}" "${optim_ops_950[@]}" "${norm_ops_950[@]}" "${quant_ops_950[@]}" "${matmul_ops_950[@]}" "${conv_ops_950[@]}")

    for name in "${operator_list_910b[@]}"
    do
        echo "[EXECUTE_COMMAND] bash build.sh -u --ops=$name --soc=ascend910b"
        bash build.sh -u --ops=$name --soc=ascend910b --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?
        if [ $status -ne 0 ]; then
            echo "${name} ascend910b ut fail"
            exit 1
        fi
    done

    for name in "${operator_list_950[@]}"
    do
        echo "[EXECUTE_COMMAND] bash build.sh -u --opapi --ophost --ops=$name --soc=ascend950"
        bash build.sh -u --opapi --ophost --ops=$name --soc=ascend950 --cann_3rd_lib_path=${ASCEND_3RD_LIB_PATH} -j16
        status=$?
        if [ $status -ne 0 ]; then
            echo "${name} ascend950 api/host ut fail"
            exit 1
        fi
    done
fi