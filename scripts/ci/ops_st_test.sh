#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# Map soc version to arch-specific ST case directory name
declare -A SOC_TO_ST_ARCH
SOC_TO_ST_ARCH=(["ascend910b"]="arch22" ["ascend950"]="arch35")

# Ops allowed in CI_ONLINE_ST mode
ONLINE_ST_WHITELIST=(
    "mat_mul_v3"
    "batch_mat_mul_v3"
)

# Check if an op is in the online ST mode whitelist
is_op_in_online_whitelist() {
    local op="$1"
    for whitelisted in "${ONLINE_ST_WHITELIST[@]}"; do
        [[ "${op}" == "${whitelisted}" ]] && return 0
    done
    return 1
}

# Check if CI_ONLINE_ST environment variable is set and enabled
is_online_st_mode() {
    [[ -n "${CI_ONLINE_ST:-}" ]] || return 1
    local val_lower
    val_lower=$(echo "${CI_ONLINE_ST}" | tr '[:upper:]' '[:lower:]')
    [[ "${val_lower}" != "0" && "${val_lower}" != "false" ]]
}

dotted_line="----------------------------------------------------------------"

# Print info message with timestamp to stderr
print_msg() {
    local msg="$1"
    local date_time
    date_time=$(date +%Y-%m-%d/%H.%M.%S)
    echo "[INFO]${date_time}: ${msg}" >&2
}

# Print error message with red color to stderr
print_error() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[31m[ERROR] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

# Print success message with green color to stderr
print_success() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[32m[SUCCESS] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

# Print warning message with yellow color to stderr
print_warning() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[33m[WARNING] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

# Get op category list from cmake/variables.cmake, fallback to hardcoded default
get_op_categories() {
    local cmake_file="${framework_path}/cmake/variables.cmake"
    local categories=""

    if [[ -f "${cmake_file}" ]]; then
        categories=$(grep "OP_CATEGORY_LIST" "${cmake_file}" | \
            sed -n 's/set(OP_CATEGORY_LIST "\(.*\)")/\1/p' | \
            tr -d '"')
    else
        categories="matmul conv activation foreach hash vfusion index loss norm optim pooling quant rnn control"
    fi

    categories="${categories} common"
    echo "${categories}"
}

# Extract op name from a file path based on directory structure
extract_op_from_path() {
    local file_path="$1"
    local op_categories="$2"
    local op_name=""

    local rel_path="${file_path#${framework_path}/}"
    local parts=(${rel_path//\// })

    if [[ ${#parts[@]} -ge 2 ]]; then
        local first_dir="${parts[0]}"
        local second_dir="${parts[1]}"

        local is_category=0
        for cat in ${op_categories}; do
            if [[ "${first_dir}" == "${cat}" ]]; then
                is_category=1
                break
            fi
        done

        if [[ ${is_category} -eq 1 ]]; then
            if [[ "${second_dir}" == "common" ]]; then
                op_name="${first_dir}.common"
            else
                op_name="${second_dir}"
            fi
        elif [[ "${first_dir}" == "experimental" && ${#parts[@]} -ge 3 ]]; then
            local exp_type="${parts[1]}"
            local exp_name="${parts[2]}"
            for cat in ${op_categories}; do
                if [[ "${exp_type}" == "${cat}" ]]; then
                    if [[ "${exp_name}" == "common" ]]; then
                        op_name="${exp_type}.common"
                    else
                        op_name="${exp_name}"
                    fi
                    break
                fi
            done
        fi
    fi

    if [[ -n "${op_name}" ]]; then
        echo "${op_name}"
    fi
}

# Parse changed file list and extract unique op names
parse_ops_from_filelist() {
    local pr_filelist="$1"

    if [[ ! -f "${pr_filelist}" ]]; then
        print_error "pr_filelist not found: ${pr_filelist}"
        return 1
    fi

    local op_categories=$(get_op_categories)
    local changed_files=$(cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' || echo "")

    if [[ -z "${changed_files}" ]]; then
        print_msg "No changed files in pr_filelist"
        return 0
    fi

    local ops_set=""
    while IFS= read -r file_line; do
        [[ -z "${file_line}" ]] && continue
        file_line=$(echo "${file_line}" | sed 's/^[MADRC]\t//')
        local op_name=$(extract_op_from_path "${file_line}" "${op_categories}")
        if [[ -n "${op_name}" ]]; then
            if [[ -z "${ops_set}" ]]; then
                ops_set="${op_name}"
            elif [[ ",${ops_set}," != *",${op_name},"* ]]; then
                ops_set="${ops_set},${op_name}"
            fi
        fi
    done <<< "${changed_files}"

    echo "${ops_set}"
}

# Merge two comma-separated op lists with deduplication
merge_ops_lists() {
    local list1="$1"
    local list2="$2"
    local merged=""

    for op in ${list1//,/ }; do
        if [[ -z "${merged}" ]]; then
            merged="${op}"
        elif [[ ",${merged}," != *",${op},"* ]]; then
            merged="${merged},${op}"
        fi
    done

    for op in ${list2//,/ }; do
        if [[ -z "${merged}" ]]; then
            merged="${op}"
        elif [[ ",${merged}," != *",${op},"* ]]; then
            merged="${merged},${op}"
        fi
    done

    echo "${merged}"
}

# Print usage information and exit
usage() {
    echo "Usage: bash ops_st_test.sh [--soc_version=ascend950] [--ops=op1,op2,op3] [--test_type=kernel,aclnn,e2e] [--pr_filelist=pr_filelist.txt]"
    echo "       bash ops_st_test.sh pr_filelist.txt"
    echo "Options:"
    echo "    --soc_version   (Optional) Specify soc version. Supported: ascend910b, ascend950. If not specified, auto-detect via 'python3 -m ttk info'."
    echo "    --ops           (Optional) Specify operators to test (comma-separated). If not specified, extract from git diff."
    echo "    --test_type     (Optional) Specify test types to run (comma-separated). Supported: kernel, aclnn, e2e. Default: all types."
    echo "    --pr_filelist   (Optional) Path to file containing list of changed files (one per line). If not specified, extract from git diff."
    echo "    --case_path     (Optional) Custom base path for test cases. If specified, st_path will be {case_path}/\${op_type}/\${op_name}"
    echo "    --testcase, -t  (Optional) Specify testcase name(s) to run (comma-separated). Mutually exclusive; cannot specify both or repeat."
    echo "    --update_ttk    (Optional) Force re-download ops-test-kit by removing existing directory and cloning again."
    echo "Examples:"
    echo "    bash ops_st_test.sh"
    echo "    bash ops_st_test.sh pr_filelist.txt"
    echo "    bash ops_st_test.sh --soc_version=ascend950"
    echo "    bash ops_st_test.sh --pr_filelist=pr_filelist.txt"
    echo "    bash ops_st_test.sh --soc_version=ascend910b --ops=mat_mul_v3,conv2d_v2"
    echo "    bash ops_st_test.sh --soc_version=ascend950 --test_type=kernel"
    echo "    bash ops_st_test.sh --soc_version=ascend950 --test_type=kernel,e2e"
    echo "    bash ops_st_test.sh --update_ttk"
}

# Detect changed ops via git diff and dependency analysis
get_changed_ops() {
    local base_branch="master"
    local changed_files

    changed_files=$(git diff --name-only "${base_branch}...HEAD" 2>/dev/null || git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")

    if [[ -z "${changed_files}" ]]; then
        print_msg "No changed files detected"
        return 0
    fi

    mkdir -p "${build_path}"
    cd "${build_path}"
    rm -f CMakeCache.txt
    cmake -DENABLE_EXPERIMENTAL=FALSE -DPREPROCESS_ONLY=ON "${framework_path}" >/dev/null 2>&1 || {
        print_error "Failed to run cmake preprocess"
        return 1
    }
    cd "${framework_path}"

    local changed_files_tmp="${build_path}/tmp/changed_files_tmp.txt"
    mkdir -p "${build_path}/tmp"
    echo "${changed_files}" > "${changed_files_tmp}"

    export BASE_PATH="${framework_path}"
    export BUILD_PATH="${build_path}"

    local is_experimental="FALSE"

    local result
    result=$(python3 "${framework_path}/scripts/util/parse_compile_changed_files.py" "${changed_files_tmp}" "${is_experimental}" 2>&1)
    rm -f "${changed_files_tmp}"

    if [[ -z "${result}" ]]; then
        print_msg "No ops detected from changed files"
        return 0
    fi

    if [[ "${DEBUG_DEPENDENCIES:-}" == "TRUE" ]]; then
        local reverse_deps=$(echo "${result}" | cut -d':' -f1)
        local compile_deps=$(echo "${result}" | cut -d':' -f2)
        print_msg "Reverse dependencies (ops that depend on changed ops): ${reverse_deps}"
        print_msg "Compile dependencies (ops needed to compile): ${compile_deps}"
    fi

    local ops_list=$(echo "${result}" | cut -d':' -f1)
    echo "${ops_list}"
}

# Clone ops-test-kit repository to ttk_path
clone_ops_test_kit() {
    print_msg "Downloading ops-test-kit..."
    git clone https://gitcode.com/cann/ops-test-kit.git "${ttk_path}" || {
        print_error "Failed to clone ops-test-kit"
        exit 1
    }
    print_msg "ops-test-kit downloaded successfully"
}

# Ensure ops-test-kit is available: clone if missing/corrupted, or re-clone if --update_ttk
download_ops_test_kit() {
    print_msg "Preparing build environment..."

    mkdir -p "${build_path}"

    if [[ ! -d "${ttk_path}" ]]; then
        clone_ops_test_kit
    elif [[ ! -f "${ttk_path}/ttk/__init__.py" ]]; then
        print_warning "ops-test-kit directory exists but appears corrupted, re-downloading..."
        rm -rf "${ttk_path}"
        clone_ops_test_kit
    elif [[ "${update_ttk}" == "TRUE" ]]; then
        print_msg "--update_ttk specified, re-downloading ops-test-kit..."
        rm -rf "${ttk_path}"
        clone_ops_test_kit
    else
        print_msg "ops-test-kit already exists, skipping download"
    fi
}

# Detect and validate SOC version via ttk info
detect_soc_version() {
    local ttk_info_output
    ttk_info_output=$(cd "${ttk_path}" && python3 -m ttk info 2>/dev/null || echo "")

    local detected_soc=""
    if echo "${ttk_info_output}" | grep -qi "Ascend910B\|Ascend 910"; then
        detected_soc="ascend910b"
    elif echo "${ttk_info_output}" | grep -qi "Ascend950\|Ascend 950"; then
        detected_soc="ascend950"
    fi

    if [[ -z "${soc_version}" ]]; then
        if [[ -z "${detected_soc}" ]]; then
            print_error "Failed to detect SOC version via 'python3 -m ttk info'. Please specify --soc_version manually."
            exit 1
        fi
        soc_version="${detected_soc}"
        print_msg "Auto-detected soc_version: ${soc_version}"
    else
        if [[ "${soc_version}" != "ascend910b" && "${soc_version}" != "ascend950" ]]; then
            print_error "Unsupported soc_version: ${soc_version}. Supported: ascend910b, ascend950"
            exit 1
        fi

        if [[ -n "${detected_soc}" && "${detected_soc}" != "${soc_version}" ]]; then
            print_error "SOC version mismatch: specified '${soc_version}' but detected '${detected_soc}' from 'python3 -m ttk info'"
            exit 1
        fi
    fi

    print_msg "soc_version: ${soc_version}"
}

# Source CANN environment with fallback chain: ASCEND_HOME_PATH -> ASCEND_TOOLKIT_HOME -> /usr/local/Ascend/cann
setup_cann_env() {
    if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
        print_msg "ASCEND_HOME_PATH already set: ${ASCEND_HOME_PATH}"
        return 0
    fi

    local cann_dirs=("${ASCEND_HOME_PATH:-}" "${ASCEND_TOOLKIT_HOME:-}" "/usr/local/Ascend/cann")
    for cann_dir in "${cann_dirs[@]}"; do
        [[ -z "${cann_dir}" ]] && continue
        if [[ -f "${cann_dir}/bin/setenv.bash" ]]; then
            print_msg "Sourcing CANN environment from: ${cann_dir}"
            set +u
            source "${cann_dir}/bin/setenv.bash"
            set -u
            return 0
        fi
    done

    print_error "CANN environment not found. Please set ASCEND_HOME_PATH or install CANN to /usr/local/Ascend/cann"
    exit 1
}

# Find the source directory of an op by name
find_op_code_path() {
    local op_name="$1"
    local code_path=$(find "${framework_path}" -type d -name "${op_name}" -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/build_out/*" | head -1)

    if [[ -z "${code_path}" ]]; then
        return 1
    fi

    echo "${code_path}"
}

# Get op category type (e.g. matmul, conv) from its source directory path
get_op_type() {
    local code_path="$1"
    local subdir_path=$(realpath "${code_path}")
    local op_type=$(basename "$(dirname "${subdir_path}")")
    echo "${op_type}"
}

# Find test case CSV files for an op, including arch-specific cases
find_test_cases() {
    local op_name="$1"
    local op_type="$2"
    local arch="$3"
    local test_case_files=()

    local st_path
    if [[ -n "${case_path}" ]]; then
        st_path="${case_path}/${op_type}/${op_name}/"
    else
        st_path="${framework_path}/${op_type}/${op_name}/tests/st"
    fi

    if [[ ! -d "${st_path}" ]]; then
        [[ "${DEBUG_DEPENDENCIES:-}" == "TRUE" ]] && print_msg "No st test directory found for ${op_name} at ${st_path}"
        return 0
    fi

    local all_prefixes=("ttk_kernel" "ttk_aclnn" "ttk_e2e")
    local search_prefixes=()

    if [[ -n "${test_type_list}" ]]; then
        IFS=',' read -r -a input_types <<< "${test_type_list}"
        for input_type in "${input_types[@]}"; do
            # Deprecated alias 'pta' is mapped to 'e2e'
            if [[ "${input_type}" == "pta" ]]; then
                search_prefixes+=("ttk_e2e")
            else
                search_prefixes+=("ttk_${input_type}")
            fi
        done
    else
        search_prefixes=("${all_prefixes[@]}")
    fi

    # Common cases: st/ttk_<type>_*.csv
    for prefix in "${search_prefixes[@]}"; do
        local csv_files=$(find "${st_path}" -maxdepth 1 -name "${prefix}_*.csv" -type f 2>/dev/null)
        for csv_file in ${csv_files}; do
            local test_type="${prefix#ttk_}"
            test_case_files+=("${test_type}:${csv_file}")
        done
    done

    # Arch-specific cases: st/<arch>/ttk_<type>_*.csv
    if [[ -n "${arch}" ]]; then
        local arch_path="${st_path}/${arch}"
        if [[ -d "${arch_path}" ]]; then
            for prefix in "${search_prefixes[@]}"; do
                local csv_files=$(find "${arch_path}" -maxdepth 1 -name "${prefix}_*.csv" -type f 2>/dev/null)
                for csv_file in ${csv_files}; do
                    local test_type="${prefix#ttk_}"
                    test_case_files+=("${test_type}:${csv_file}")
                done
            done
        fi
    fi

    echo "${test_case_files[*]}"
}

# Get the tests directory path for an op
get_ops_test_path() {
    local op_name="$1"
    local op_type="$2"
    local ops_test_path="${framework_path}/${op_type}/${op_name}/tests"

    if [[ ! -d "${ops_test_path}" ]]; then
        print_msg "No tests directory found for ${op_name} at ${op_type}/${op_name}/tests"
        return 1
    fi

    echo "${ops_test_path}"
}

# Check precision status of test results in a CSV file
check_precision_status() {
    local result_csv="$1"
    local op_name="$2"
    local testcase_name="$3"

    if [[ ! -f "${result_csv}" ]]; then
        print_warning "Result csv file not found: ${result_csv}"
        return 1
    fi

    python3 "${framework_path}/scripts/ci/ops_test_util.py" \
        --action=check_precision \
        --result_csv="${result_csv}" \
        --op_name="${op_name}" \
        --testcase_name="${testcase_name}"

    return $?
}

# Verify that plugin assets directory contains required .py files
check_plugin_assets() {
    local plugin_path="$1"
    local op_name="$2"

    local assets_path="${plugin_path}/assets"

    if [[ ! -d "${assets_path}" ]]; then
        print_warning "assets directory not found for ${op_name}: ${assets_path}"
        return 1
    fi

    local py_files=$(find "${assets_path}" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -1)
    if [[ -z "${py_files}" ]]; then
        print_warning "No .py files found in assets directory for ${op_name}: ${assets_path}"
        return 1
    fi

    return 0
}

# Run kernel-level test for an op using ttk kernel mode
run_kernel_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi

    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running kernel test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi
    local cmd="python3 -m ttk kernel -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} -d=false -b=release --pc=16 --run 1 --compare close --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" > /dev/null
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "kernel test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    else
        print_msg "kernel test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Run aclnn API test for an op using ttk aclnn mode
run_aclnn_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi

    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running aclnn test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi
    local cmd="python3 -m ttk aclnn -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} --pc=16 --run 1 --compare close --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" > /dev/null
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "aclnn test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    else
        print_msg "aclnn test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Run end-to-end test for an op using ttk e2e mode
run_e2e_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"

    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi

    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi

    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi

    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"

    print_msg "Running e2e test for ${op_name}, testcase: ${testcase_name}"

    cd "${ttk_path}"

    local testcase_arg=""
    if [[ -n "${testcase_filter}" ]]; then
        testcase_arg="-t ${testcase_filter}"
    fi
    local cmd="python3 -m ttk e2e -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} --pc=16 --run 1 --compare close --task-prof false --no-memory-check ${testcase_arg}"
    print_msg "Executing: ${cmd}"

    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" > /dev/null
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    if [[ ${test_failed} -ne 0 ]]; then
        print_error "e2e test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    else
        print_msg "e2e test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi

    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"

    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

# Summarize test results to CSV and print pass rate for an op
summarize_op_results() {
    local op_name="$1"
    local test_type="$2"
    local result_csvs="$3"

    local summary_file="${log_path}/${test_type}_summary.csv"
    local summary_header="op_name,testcase_name,test_type,result_csv,status"

    if [[ ! -f "${summary_file}" ]]; then
        echo "${summary_header}" > "${summary_file}"
    fi

    if [[ -z "${result_csvs}" ]]; then
        return 0
    fi

    for result_csv in ${result_csvs}; do
        if [[ ! -f "${result_csv}" ]]; then
            continue
        fi

        python3 "${framework_path}/scripts/ci/ops_test_util.py" \
            --action=summarize \
            --result_csv="${result_csv}" \
            --op_name="${op_name}" \
            --test_type="${test_type}" \
            --summary_file="${summary_file}"
    done

    local total_cases=0
    local passed_cases=0
    while IFS=',' read -r csv_op csv_testcase csv_type csv_result csv_status; do
        if [[ "${csv_op}" == "${op_name}" && "${csv_type}" == "${test_type}" ]]; then
            ((total_cases++))
            if [[ "${csv_status}" == "PASS" ]]; then
                ((passed_cases++))
            fi
        fi
    done < <(tail -n +2 "${summary_file}")

    if [[ ${total_cases} -gt 0 ]]; then
        print_msg "${op_name} ${test_type}: ${passed_cases}/${total_cases} passed"
    fi
}

# Print a formatted summary table of all test results
print_summary_table() {
    python3 "${framework_path}/scripts/ci/ops_test_util.py" \
        --action=print_table \
        --log_path="${log_path}"
}

# Generate and print reproduction commands for failed test cases
print_reproduction_commands() {
    local repro_file="${log_path}/repro_commands.txt"
    : > "${repro_file}"
    local has_failed=0

    {
        echo ""
        echo "${dotted_line}"
        echo "[ERROR] Reproduction commands for failed test cases:"
        echo "${dotted_line}"
        echo ""
    } >&2

    local summary_files=("kernel_summary.csv" "aclnn_summary.csv" "e2e_summary.csv")
    for sf in "${summary_files[@]}"; do
        local summary_path="${log_path}/${sf}"
        [[ ! -f "${summary_path}" ]] && continue
        local test_type="${sf%%_summary.csv}"

        while IFS=',' read -r f_op f_tc f_tt f_csv f_status || [[ -n "${f_op}" ]]; do
            [[ "${f_op}" == "op_name" ]] && continue
            [[ "${f_status}" == "PASS" ]] && continue
            [[ -z "${f_op}" ]] && continue

            has_failed=1

            local code_path="" op_type="" ops_test_path="" input_csv=""
            code_path=$(find_op_code_path "${f_op}" 2>/dev/null) || code_path=""
            if [[ -n "${code_path}" ]]; then
                op_type=$(get_op_type "${code_path}")
                ops_test_path=$(get_ops_test_path "${f_op}" "${op_type}" 2>/dev/null) || ops_test_path=""
            fi

            local result_basename csv_basename
            result_basename=$(basename "${f_csv}")
            csv_basename="${result_basename%_result.csv}"
            if [[ -n "${csv_basename}" ]]; then
                input_csv=$(find "${framework_path}" -name "${csv_basename}.csv" -type f 2>/dev/null | head -1)
            fi

            local ttk_mode extra_flags
            case "${test_type}" in
                kernel) ttk_mode="kernel"; extra_flags="-d=false -b=release";;
                aclnn)  ttk_mode="aclnn";  extra_flags="";;
                e2e)    ttk_mode="e2e";    extra_flags="";;
                *)      ttk_mode="${test_type}"; extra_flags="";;
            esac

            local repro_cmd="cd ${ttk_path} && python3 -m ttk ${ttk_mode} -i ${input_csv} -o ${f_csv} --plugin ${ops_test_path} ${extra_flags} --pc=16 --run 1 --compare close --task-prof false --no-memory-check -t ${f_tc}"

            {
                echo "  # [${test_type}] ${f_op} / ${f_tc}"
                echo "  ${repro_cmd}"
                echo ""
            } >&2

            echo "[${test_type}] ${f_op} / ${f_tc}" >> "${repro_file}"
            echo "${repro_cmd}" >> "${repro_file}"
            echo "" >> "${repro_file}"
        done < "${summary_path}"
    done

    if [[ ${has_failed} -eq 0 ]]; then
        echo "  (no failed test cases found)" >&2
    fi
    echo "${dotted_line}" >&2
    print_msg "Reproduction commands also saved to: ${repro_file}"
}

# Run all test types for a single op and collect results
run_single_op_test() {
    local op_name="$1"

    print_msg "=== Testing op: ${op_name} ==="

    local code_path=$(find_op_code_path "${op_name}")
    if [[ -z "${code_path}" ]]; then
        print_warning "Cannot find op directory for ${op_name}, skipping"
        return 0
    fi

    local op_type=$(get_op_type "${code_path}")
    print_msg "op_type: ${op_type}, op_name: ${op_name}"

    local arch="${SOC_TO_ST_ARCH[${soc_version}]:-}"
    if [[ -n "${arch}" ]]; then
        print_msg "soc_version: ${soc_version}, arch: ${arch}"
    fi

    local ops_test_path=$(get_ops_test_path "${op_name}" "${op_type}")
    if [[ -z "${ops_test_path}" ]]; then
        print_msg "No tests directory found, skipping ${op_name}"
        return 0
    fi

    local test_cases=$(find_test_cases "${op_name}" "${op_type}" "${arch}")

    if [[ -z "${test_cases}" ]]; then
        print_msg "No test cases found for ${op_name}"
        return 0
    fi

    local result_csvs=()
    local kernel_csvs=()
    local aclnn_csvs=()
    local e2e_csvs=()
    local test_case_array=(${test_cases})
    local result_csv
    local testcase_name
    local op_error_flag=0

    declare -A test_runners=(
        ["kernel"]="run_kernel_test"
        ["aclnn"]="run_aclnn_test"
        ["e2e"]="run_e2e_test"
    )

    for test_item in "${test_case_array[@]}"; do
        local test_type=$(echo "${test_item}" | cut -d':' -f1)
        local test_csv=$(echo "${test_item}" | cut -d':' -f2-)
        local test_ret=0

        if [[ -z "${test_runners[${test_type}]+_}" ]]; then
            print_warning "Unknown test type: ${test_type}, skipping"
            continue
        fi

        local runner_func="${test_runners[${test_type}]}"
        result_csv=$(${runner_func} "${op_name}" "${test_csv}" "${ops_test_path}")
        test_ret=$?

        if [[ ${test_ret} -ne 0 ]]; then
            op_error_flag=1
        fi

        if [[ -n "${result_csv}" ]]; then
            result_csvs+=("${result_csv}")
            # Append result to the per-type array dynamically, e.g. kernel_csvs
            eval "${test_type}_csvs+=(\"\${result_csv}\")"
        fi
    done

    summarize_op_results "${op_name}" "kernel" "${kernel_csvs[*]}"
    summarize_op_results "${op_name}" "aclnn" "${aclnn_csvs[*]}"
    summarize_op_results "${op_name}" "e2e" "${e2e_csvs[*]}"

    # Emit lines in 'op_name:testcase_name:result_csv' format for the caller
    for csv in "${result_csvs[@]}"; do
        testcase_name=$(basename "${csv}" _result.csv)
        echo "${op_name}:${testcase_name}:${csv}"
    done

    if [[ ${op_error_flag} -ne 0 ]]; then
        return 1
    fi
}

# Parse command-line arguments and validate inputs
parse_args() {
    ops_list=""
    soc_version=""
    test_type_list=""
    pr_filelist=""
    case_path=""
    testcase_filter=""
    update_ttk=""

    for arg in "$@"; do
        case "${arg}" in
            --ops=*)
                ops_list="${arg#*=}"
                ;;
            --soc_version=*)
                soc_version="${arg#*=}"
                ;;
            --test_type=*)
                test_type_list="${arg#*=}"
                ;;
            --pr_filelist=*)
                pr_filelist="${arg#*=}"
                ;;
            --case_path=*)
                case_path="${arg#*=}"
                ;;
            --testcase=*|-t=*)
                if [[ -n "${testcase_filter}" ]]; then
                    print_error "Cannot specify --testcase and -t together (or repeat either one)"
                    usage
                    exit 1
                fi
                testcase_filter="${arg#*=}"
                ;;
            --update_ttk)
                update_ttk="TRUE"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                print_error "Unknown argument: ${arg}"
                usage
                exit 1
                ;;
            *)
                if [[ -z "${pr_filelist}" ]]; then
                    pr_filelist="${arg}"
                else
                    print_error "Multiple pr_filelist arguments: ${pr_filelist} and ${arg}"
                    usage
                    exit 1
                fi
                ;;
        esac
    done

    if [[ -n "${pr_filelist}" && ! -f "${pr_filelist}" ]]; then
        print_error "pr_filelist not found: ${pr_filelist}"
        exit 1
    fi

    if [[ -n "${test_type_list}" ]]; then
        IFS=',' read -r -a valid_types <<< "kernel,aclnn,e2e"
        IFS=',' read -r -a input_types <<< "${test_type_list}"
        for input_type in "${input_types[@]}"; do
            local found=0
            for valid_type in "${valid_types[@]}"; do
                if [[ "${input_type}" == "${valid_type}" ]]; then
                    found=1
                    break
                fi
            done
            if [[ ${found} -eq 0 ]]; then
                print_error "Unsupported test_type: ${input_type}. Supported: kernel, aclnn, e2e"
                exit 1
            fi
        done
    fi

    print_msg "ops_list: ${ops_list:-'auto detect from git diff or pr_filelist'}"
    print_msg "test_type_list: ${test_type_list:-'all types'}"
    if [[ -n "${pr_filelist}" ]]; then
        print_msg "pr_filelist: ${pr_filelist}"
    fi
    if [[ -n "${case_path}" ]]; then
        print_msg "case_path: ${case_path}"
    fi
    if [[ -n "${update_ttk}" ]]; then
        print_msg "update_ttk: ${update_ttk}"
    fi
}

# Global paths used across functions
framework_path="$(cd "$(dirname "$0")/../.." && pwd)"
build_path="${framework_path}/build"
log_path="${framework_path}/st/log"
ttk_path="${build_path}/third_party/ops-test-kit"

parse_args "$@"

# Normalize pr_filelist to absolute path
if [[ -n "${pr_filelist}" && "${pr_filelist}" != /* ]]; then
    pr_filelist="$(pwd)/${pr_filelist}"
fi

rm -rf "${log_path:?}"/*
mkdir -p "${log_path}"

# Stage 1: ensure ops-test-kit is available
download_ops_test_kit

if [[ "${update_ttk}" == "TRUE" ]]; then
    print_success "ops-test-kit updated successfully."
    exit 0
fi

# Stage 2: source CANN environment (required by ttk info and tests)
setup_cann_env

# Stage 3: detect SOC version via ttk info
detect_soc_version

# Stage 4: determine ops to test (--ops, pr_filelist or git diff)
if [[ -n "${ops_list}" && -z "${pr_filelist}" ]]; then
    print_msg "Using ops from --ops parameter: ${ops_list}"
elif [[ -z "${ops_list}" && -n "${pr_filelist}" ]]; then
    print_msg "Extracting ops from pr_filelist..."
    print_msg "pr_filelist content:"
    cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' | sed 's/^[MADRC]\t//' >&2
    ops_list=$(parse_ops_from_filelist "${pr_filelist}")
elif [[ -n "${ops_list}" && -n "${pr_filelist}" ]]; then
    print_msg "Merging ops from pr_filelist and --ops parameter..."
    print_msg "--ops input: ${ops_list}"
    print_msg "pr_filelist content:"
    cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' | sed 's/^[MADRC]\t//' >&2
    ops_from_filelist=$(parse_ops_from_filelist "${pr_filelist}")
    ops_list=$(merge_ops_lists "${ops_from_filelist}" "${ops_list}")
else
    print_msg "Extracting ops from git diff..."
    ops_list=$(get_changed_ops)
    ops_list=$(echo "${ops_list}" | tr ';' ',')
fi

if [[ -z "${ops_list}" ]]; then
    print_msg "No ops to test"
    exit 0
fi

print_msg "Ops to test: ${ops_list}"

IFS=',' read -r -a op_name_array <<< "${ops_list}"

# Stage 5: run tests for each op
all_result_csvs=()
result_flag=0
for op_name in "${op_name_array[@]}"; do
    if is_online_st_mode; then
        if ! is_op_in_online_whitelist "${op_name}"; then
            print_msg "CI_ONLINE_ST mode: skipping ${op_name} (not in whitelist)"
            continue
        fi
    fi
    op_results=$(run_single_op_test "${op_name}") || result_flag=1
    if [[ -n "${op_results}" ]]; then
        while IFS= read -r line; do
            all_result_csvs+=("${line}")
        done <<< "${op_results}"
    fi
done

# Stage 6: precision check on all result CSVs
print_msg "=== Starting precision check for all test cases ==="
precision_flag=0
for result_info in "${all_result_csvs[@]}"; do
    op_name=$(echo "${result_info}" | cut -d':' -f1)
    testcase_name=$(echo "${result_info}" | cut -d':' -f2)
    result_csv=$(echo "${result_info}" | cut -d':' -f3)
    check_precision_status "${result_csv}" "${op_name}" "${testcase_name}" || precision_flag=1
done

print_summary_table

# Stage 7: print summary table and decide final exit code
if [[ ${result_flag} -ne 0 || ${precision_flag} -ne 0 ]]; then
    print_reproduction_commands
    print_error "Some tests or precision checks failed. See reproduction commands above."
    if is_online_st_mode; then
        print_warning "CI_ONLINE_ST mode: exiting with 0 despite failures"
        exit 0
    fi
    exit 1
else
    print_success "All tests and precision checks passed."
    exit 0
fi
