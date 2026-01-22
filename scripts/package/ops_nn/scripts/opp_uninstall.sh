#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

PARAM_INVALID="0x0002"
PARAM_INVALID_DES="Invalid input parameter."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
OPERATE_FAILED="0x0001"

_CURR_PATH=$(dirname $(readlink -f $0))
_COMMON_INC_FILE="${_CURR_PATH}/common_func.inc"
_OPP_COMMON_FILE="${_CURR_PATH}/opp_common.sh"
. "${_COMMON_INC_FILE}"
. "${_OPP_COMMON_FILE}"

logwitherrorlevel() {
    _ret_status="$1"
    _level="$2"
    _msg="$3"
    if [ "${_ret_status}" != 0 ]; then
        if [ "${_level}" = "error" ]; then
            logandprint "${_msg}"
            exit 1
        else
            logandprint "${_msg}"
        fi
    fi
}

checkdirectoryexist() {
    _path="${1}"
    if [ ! -d "${_path}" ]; then
        logandprint "[ERROR]: ERR_NO:${FILE_READ_FAILED};ERR_DES:Installation directroy [${_path}] does not exist, uninstall failed."
        return 1
    else
        return 0
    fi
}

checkfileexist() {
    _path_param="${1}"
    if [ ! -f "${_path_param}" ];then
        logandprint "[ERROR]: ERR_NO:${FILE_READ_FAILED};ERR_DES:The file (${_path_param}) does not existed."
        return 1
    else
        return 0
    fi
}

# DFS sub-folders cleaner
deleteemptyfolders() {
    _init_dir="$1"
    _aicpu_filter="$2"
    find "${_init_dir}" -mindepth 1 -maxdepth 1 -type d ! \
        -path "${_aicpu_filter}" 2> /dev/null | while read -r dir
    do
        if [ "$(echo "${dir}" | grep "custom")" = "" ]; then
            deleteemptyfolders "${dir}"

            if [ "$(find "${dir}" -mindepth 1 -type d)" = "" ] && \
                [ "$(ls -A "${dir}")" = "" ] >/dev/null; then
                rm -rf -d "${dir}"
            fi
        else
            # remove custom folders which not contains sub-folder or any files
            if [ "$(ls -A "${dir}")" = "" ]; then
                rm -rf -d "${dir}"
            fi
        fi
    done
}

checkinstalledtype() {
    _type="$1"
    if [ "${_type}" != "run" ] &&
    [ "${_type}" != "full" ] &&
    [ "${_type}" != "devel" ]; then
        logandprint "[ERROR]: ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Install type \
[${_ugroup}] of ops_nn module is not right!"
        return 1
    else
        return 0
    fi
}

getinstallpath() {
    docker_root_tmp="$(echo "${docker_root}" | sed "s#/\+\$##g")"
    docker_root_regex="$(echo "${docker_root_tmp}" | sed "s#\/#\\\/#g")"
    relative_path_val=$(echo "${_ABS_INSTALL_PATH}" | sed "s/^${docker_root_regex}//g" | sed "s/\/\+\$//g")
    return
}

unsetenv() {
    logandprint "[INFO]: Unset the environment path [ export ASCEND_OPS_NN_PATH=${relative_path_val}/${ops_nn_platform_dir} ]."
    target_username=$(getinstalledinfo "${KEY_INSTALLED_UNAME}")
    target_usergroup=$(getinstalledinfo "${KEY_INSTALLED_UGROUP}")
    if [ "${is_docker_install}" = y ] ; then
        uninstall_option="--docker-root=${docker_root}"
    else
        uninstall_option=""
    fi
}

installed_path="$1"
uninstall_mode="$2"
is_quiet="$3"
_CHIP_TYPE="$4"
is_docker_install="$5"
docker_root="$6"
pkg_version_dir="${7}"
paramter_num="$#"

logandprint "[INFO]: Command ops_nn_uninstall"

if [ "${paramter_num}" != 0 ]; then
    if [ "${installed_path}" = "" ] ||
    [ "${uninstall_mode}" = "" ] ||
    [ "${is_quiet}" = "" ] ; then
        logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Empty paramters is invalid\
for call uninstall functions."
        exit 1
    fi
fi

SCENE_FILE="${_CURR_PATH}""/../scene.info"
platform_data=$(grep -e "arch" "$SCENE_FILE" | cut --only-delimited -d"=" -f2-)
ops_nn_platform_old_dir=ops_nn_$platform_data-linux
ops_nn_platform_dir=ops_nn
upper_opp_platform=$(echo "${ops_nn_platform_dir}" | tr 'a-z' 'A-Z')
_FILELIST_FILE="${_CURR_PATH}""/filelist.csv"
_COMMON_PARSER_FILE="${_CURR_PATH}""/install_common_parser.sh"
_TARGET_INSTALL_PATH="${_CURR_PATH}""/../../../../"
_INSTALL_INFO_SUFFIX="share/info/${ops_nn_platform_dir}/ascend_install.info"
_VERSION_INFO_SUFFIX="share/info/${ops_nn_platform_dir}/version.info"

# avoid relative path casued errors by delete floders
_ABS_INSTALL_PATH=$(cd ${_TARGET_INSTALL_PATH}; pwd)
getinstallpath
relative_path_info=${relative_path}
# init log file path
_INSTALL_INFO_FILE="${_ABS_INSTALL_PATH}/${_INSTALL_INFO_SUFFIX}"
if [ ! -f "${_INSTALL_INFO_FILE}" ]; then
    _INSTALL_INFO_FILE="/etc/ascend_install.info"
fi
# this is ops_nn verion info file
_VERSION_INFO_FILE="${_ABS_INSTALL_PATH}/${_VERSION_INFO_SUFFIX}"

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${upper_opp_platform}_INSTALL_TYPE"
KEY_INSTALLED_FEATURE="${upper_opp_platform}_Install_Feature"
KEY_INSTALLED_PATH="${upper_opp_platform}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${upper_opp_platform}_VERSION"
getinstalledinfo() {
    _key="$1"
    _res=""
    if [ -f "${_INSTALL_INFO_FILE}" ]; then
        chmod 644 "${_INSTALL_INFO_FILE}"> /dev/null 2>&1
        case "${_key}" in
        USERNAME)
            res=$(cat ${_INSTALL_INFO_FILE} | grep "USERNAME" | awk -F = '{print $2}')
            ;;
        USERGROUP)
            res=$(cat ${_INSTALL_INFO_FILE} | grep "USERGROUP" | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_TYPE)
            type="INSTALL_TYPE"
            res=$(cat ${_INSTALL_INFO_FILE} | grep "${type}" | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_PATH_VAL)
            val="INSTALL_PATH_VAL"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${val} | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_VERSION)
            version="VERSION"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${version} | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_PATH_PARAM)
            param="INSTALL_PATH_PARAM"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${param} | awk -F = '{print $2}')
            ;;
        esac
    fi
    echo "${res}"
}

logandprint "[INFO]: Begin uninstall the ops_nn module."

# check install folder existed
checkfileexist "${_INSTALL_INFO_FILE}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall ops_nn module failed."
checkfileexist "${_FILELIST_FILE}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall ops_nn module failed."
checkfileexist "${_COMMON_PARSER_FILE}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall ops_nn module failed."
ops_nn_sub_dir="${_ABS_INSTALL_PATH}""/share/info/${ops_nn_platform_dir}"
checkdirectoryexist "${ops_nn_sub_dir}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall ops_nn module failed."

installed_type=$(getinstalledinfo "${KEY_INSTALLED_TYPE}")
checkinstalledtype "${installed_type}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall ops_nn module failed."

_CUSTOM_PERM="755"
_BUILTIN_PERM="555"
# make the ops_nn and the upper folder can write files
is_change_dir_mode="false"
if [ "$(id -u)" != 0 ] && [ ! -w "${_TARGET_INSTALL_PATH}" ]; then
    chmod u+w "${_TARGET_INSTALL_PATH}" 2> /dev/null
    is_change_dir_mode="true"
fi

# change installed folder's permission except aicpu
subdirs=$(ls "${_TARGET_INSTALL_PATH}/${ops_nn_platform_dir}" 2> /dev/null)
for dir in ${subdirs}; do
    if [ "${dir}" != "Ascend310" ] && [ "${dir}" != "Ascend310RC" ] && [ "${dir}" != "Ascend910" ] && [ "${dir}" != "Ascend310P" ] && [ "${dir}" != "Ascend" ] && [ "${dir}" != "aicpu" ]; then
        chmod -R "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}/${ops_nn_platform_dir}/${dir}" 2> /dev/null
    fi
done
chmod "${_CUSTOM_PERM}" "${_TARGET_INSTALL_PATH}/${ops_nn_platform_dir}" 2> /dev/null

remove_init_py() {
  local built_in_impl_path=${_ABS_INSTALL_PATH}/opp/built-in/op_impl/ai_core/tbe/impl/ops_nn

  [ -e ${built_in_impl_path}/__init__.py ] && rm ${built_in_impl_path}/__init__.py > /dev/null 2>&1

  [ -e ${built_in_impl_path}/dynamic/__init__.py ] && rm ${built_in_impl_path}/dynamic/__init__.py > /dev/null 2>&1
}

remove_init_py

get_version "pkg_version" "$_VERSION_INFO_FILE"

# delete ops_nn source files
unsetenv

is_multi_version_pkg "pkg_is_multi_version" "$_VERSION_INFO_FILE "

if [ "${pkg_version_dir}" = "" ]; then
    FINAL_INSTALL_PATH=${_ABS_INSTALL_PATH}
else
    TMP_PATH="${_ABS_INSTALL_PATH}/.."
    FINAL_INSTALL_PATH=$(cd ${TMP_PATH}; pwd)
fi

# 赋可写权限
chmod +w -R "${_COMMON_PARSER_FILE}"

sh "${_COMMON_PARSER_FILE}" --package="${ops_nn_platform_dir}" --uninstall --recreate-softlink --username="${target_username}" --usergroup="${target_usergroup}" --version=$pkg_version \
    --use-share-info --version-dir=$pkg_version_dir --remove-install-info ${uninstall_option} "${installed_type}" "${FINAL_INSTALL_PATH}" "${_FILELIST_FILE}" "${_CHIP_TYPE}" --recreate-softlink
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Uninstall ops_nn module failed."


# delete install.info file
if [ "${uninstall_mode}" != "upgrade" ]; then
    logandprint "[INFO]: Delete the install info file (${_INSTALL_INFO_FILE})."
    rm -f "${_INSTALL_INFO_FILE}"
    logwitherrorlevel "$?" "warn" "[WARNING]Delete ops_nn install info file failed, \
please delete it by yourself."
fi

deleteopp(){
    if [ -s "$1" -a -d "$1" ];then
       for file in $(ls -a "$1")
       do
         if [ -f "$1/$file" ];then
           return 1
         fi
        if test -d "$1/$file";then
            if [ "$file" != '.' -a "$file" != '..' ];then
                   return 1
            fi
        fi
       done
       rm -rf -d "$1"
    fi
}

# delete the empty ops_nn folder it'self
res_val=$(ls "${ops_nn_sub_dir}" 2> /dev/null)
if [ "${res_val}" = "" ]; then
    rm -rf -d "${ops_nn_sub_dir}" >> /dev/null 2>&1
fi

# change installed folder's permission except aicpu
subdirs_param=$(ls "${_ABS_INSTALL_PATH}/${ops_nn_platform_dir}" 2> /dev/null)
for dir in ${subdirs_param}; do
    if [ "${dir}" != "Ascend310" ] && [ "${dir}" != "Ascend310RC" ] && [ "${dir}" != "Ascend910" ] && [ "${dir}" != "Ascend310P" ] && [ "${dir}" != "Ascend" ] && [ "${dir}" != "aicpu" ]; then
        chmod "${_BUILTIN_PERM}" "${_ABS_INSTALL_PATH}/${ops_nn_platform_dir}/${dir}" 2> /dev/null
    fi
done

if [ "${is_change_dir_mode}" = "true" ]; then
    chmod u-w "${_ABS_INSTALL_PATH}" 2> /dev/null
fi

temp=$(ls "${_ABS_INSTALL_PATH}/${ops_nn_platform_dir}" 2> /dev/null)
if [ -d "${_ABS_INSTALL_PATH}/${ops_nn_platform_dir}/" ]; then
    # find custom file in path and print log
    for file in $(ls -A ${_ABS_INSTALL_PATH}/${ops_nn_platform_dir}/* 2> /dev/null)
    do
        if [ "${file##*.}" != "info" ] && [ "${file}" != "Ascend310" ] && [ "${file}" != "Ascend310RC" ] && [ "${file}" != "Ascend910" ] && [ "${file}" != "Ascend310P" ] && [ "${file}" != "Ascend" ] && [ "${file}" != "aicpu" ];then
            logandprint "[WARNING]: ${file}, has files changed by users, cannot be delete."
        fi
    done
fi
# delete scene.info
scene_dir="${_ABS_INSTALL_PATH}/${ops_nn_platform_dir}/scene.info"
if [ -f ${scene_dir} ]; then
    rm -f ${scene_dir}
fi

# delete the upper folder when it is empty
dir_existed=$(ls "${_ABS_INSTALL_PATH}" 2> /dev/null)
if [ "${dir_existed}" = "" ] && [ "${uninstall_mode}" != "upgrade" ]; then
    rm -rf -d "${_ABS_INSTALL_PATH}" >> /dev/null 2>&1
fi

subdirs_param_install=$(ls "${installed_path}" 2> /dev/null)
if [ "${subdirs_param_install}" = "" ]; then
    [ -n "${installed_path}" ] && rm -rf "${installed_path}"
fi

logandprint "[INFO]: OPS_NN package uninstalled successfully! Uninstallation takes effect immediately."
exit 0

