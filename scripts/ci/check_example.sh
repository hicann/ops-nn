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
#    1. 二级目录下存在examples文件夹
#    2. examples文件夹中存在前缀为test_aclnn_的文件
#    3. 构建出valid_dirs清单

current_dir=$(pwd)
variables_cmake="cmake/variables.cmake"
op_category_list=$(grep -oP 'set\(OP_CATEGORY_LIST\s*\K".*"' $current_dir/$variables_cmake | sed 's/"//g')
IFS=' ' read -r -a op_categories <<< "$op_category_list"
valid_dirs=()
pr_file=$(realpath "${1:-pr_filelist.txt}")

for category in "${op_categories[@]}"
do  
    first_level="$current_dir/$category"
    if [ -d "$first_level" ]; then
        for second_level in "$first_level"/*/
        do
            examples_dir="$second_level""examples"
            # 检查 examples 目录是否存在
            if [ -d "$examples_dir" ]; then
                # 检查 examples 目录中是否有以 test_aclnn_ 开头的文件
                if ls "$examples_dir"/test_aclnn_* 1> /dev/null 2> /dev/null; then
                    dir_name=$(basename "$second_level")
                    valid_dirs+=("$dir_name")
                fi
            fi
        done
    fi
done


#搜索变更的文件名，将存在于valid_dirs清单中的算子保存到ops_name中

ops_name=()
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
            # 去重后添加到 ops_name 数组中
            if [[ ! " ${ops_name[@]} " =~ " $dir " ]]; then
                ops_name+=("$dir")
            fi
            break
        fi
    done

    #如果file_path为scripts/ci/mirror_update_time.txt，则将准备好的需验证算子列表加到ops_name中
    if [[ "$file_path" == "scripts/ci/mirror_update_time.txt" ]]; then

        foreach_ops_910b=("foreach_abs" "foreach_add_list")
        control_ops_910b=("assert")
        index_ops_910b=("embedding_dense_grad_v2" "gather_elements_v2" "scatter_elements_v2")
        vfusion_ops_910b=("scaled_masked_softmax_v2")
        rnn_ops_910b=("dynamic_rnn")
        pooling_ops_910b=("avg_pool3_d" "adaptive_max_pool3d")
        activation_ops_910b=("ge_glu_v2" "ge_glu_grad_v2")
        loss_ops_910b=("logit" "logit_grad")
        optim_ops_910b=("apply_adam_w_v2" "apply_fused_ema_adam")
        norm_ops_910b=("rms_norm" "add_layer_norm_grad")
        quant_ops_910b=("flat_quant" "dynamic_quant")
        matmul_ops_910b=("addmv" "batch_mat_mul_v3" "fused_linear_cross_entropy_loss_grad" "fused_linear_online_max_sum" "fused_quant_mat_mul" "gemm" "mat_mul_v3" "mv" "quant_batch_matmul_v3" "weight_quant_batch_matmul_v2")
        conv_ops_910b=("conv2d_v2" "conv3d_v2")
        operator_list_910b=("${foreach_ops_910b[@]}" "${index_ops_910b[@]}" "${vfusion_ops_910b[@]}" "${rnn_ops_910b[@]}" "${pooling_ops_910b[@]}" "${activation_ops_910b[@]}" "${loss_ops_910b[@]}" "${optim_ops_910b[@]}" "${norm_ops_910b[@]}" "${quant_ops_910b[@]}" "${matmul_ops_910b[@]}" "${conv_ops_910b[@]}")

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
        
        for op in "${operator_list_910b[@]}"; do
            op_exists=0
            for existing_op in "${ops_name[@]}"; do
                if [ "$existing_op" = "$op" ]; then
                    op_exists=1
                    break
                fi
            done
            if [ "$op_exists" -eq 0 ]; then
                ops_name+=("$op")
            fi
        done
    fi

done


#自定义算子包验证run_example
for name in "${ops_name[@]}"
do
    #安装指定路径的自定义算子包
    echo "--------------------------------"
    echo "${name}"
    ./single/cann-ops-nn-${name}-linux.*.run
    echo "[EXECUTE_COMMAND] bash build.sh --run_example $name eager cust --vendor_name=$name"
    bash build.sh --run_example $name eager cust --vendor_name=$name
    status=$?
    if [ $status -ne 0 ]; then
        echo "${name} example fail"
        exit 1
    fi
done