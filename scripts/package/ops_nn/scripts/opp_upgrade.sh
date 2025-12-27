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

_CURR_OPERATE_USER="$(id -nu 2> /dev/null)"
_CURR_OPERATE_GROUP="$(id -ng 2> /dev/null)"
_DEFAULT_INSTALL_PATH=/usr/local/Ascend
# defaults for general user
if [ "$(id -u)" != "0" ]; then
    _DEFAULT_USERNAME="${_CURR_OPERATE_USER}"
    _DEFAULT_USERGROUP="${_CURR_OPERATE_GROUP}"
    _DEFAULT_INSTALL_PATH="${HOME}/Ascend"
fi
# run package's files info
_CURR_PATH=$(dirname $(readlink -f $0))
_FILELIST_FILE="${_CURR_PATH}""/filelist.csv"
_COMMON_PARSER_FILE="${_CURR_PATH}""/install_common_parser.sh"
_VERSION_INFO_FILE="${_CURR_PATH}""/../version.info"
_INSTALL_SHELL_FILE="${_CURR_PATH}""/opp_install.sh"
SCENE_FILE="${_CURR_PATH}""/../scene.info"
platform_data=$(grep -e "arch" "$SCENE_FILE" | cut --only-delimited -d"=" -f2-)
ops_nn_platform_old_dir=ops_nn_$platform_data-linux
ops_nn_platform_dir=ops_nn
upper_ops_nn_platform=$(echo "${ops_nn_platform_dir}" | tr 'a-z' 'A-Z')

_INSTALL_INFO_SUFFIX="share/info/${ops_nn_platform_dir}/ascend_install.info"
common_func_path="${_CURR_PATH}/common_func.inc"
version_cfg="${_CURR_PATH}/version_cfg.inc"
common_fuc_v2="${_CURR_PATH}/common_func_v2.inc"
_OPP_COMMON_FILE="${_CURR_PATH}/opp_common.sh"
. "${common_func_path}"
. "${common_fuc_v2}"
. "${version_cfg}"
. "${_OPP_COMMON_FILE}"

FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
UPGRADE_FAILED="0x0000"
UPGRADE_FAILED_DES="Update successfully."

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

#check ascend_install.info for the change in code warning
checkascendinfo() {
    file_param="${install_version_dir}/${_INSTALL_INFO_SUFFIX}"
    if [ -f "${file_param}" ]; then
        inst_type=$(cat ${file_param} | grep "Opp_Install_Type" | awk -F = '{print $2}')
        uname_param=$(cat ${file_param} | grep "UserName" | awk -F = '{print $2}')
        ugroup_param=$(cat ${file_param} | grep "UserGroup" | awk -F = '{print $2}')
        path_param=$(cat ${file_param} | grep "Opp_Install_path_Param" | awk -F = '{print $2}')
        path_params=$(cat ${file_param} | grep "Opp_Install_Path_Param" | awk -F = '{print $2}')
        version_param=$(cat ${file_param} | grep "Opp_Version" | awk -F = '{print $2}')
        if [ "$inst_type" != "" ]; then
            echo "${upper_ops_nn_platform}_INSTALL_TYPE=${inst_type}" >> ${file_param}
        fi
        if [ "$uname_param" != "" ]; then
            echo "USERNAME=${uname_param}" >> ${file_param}
        fi
        if [ "$ugroup_param" != "" ]; then
            echo "USERGROUP=${ugroup_param}" >> ${file_param}
        fi
        if [ "$path_param" != "" ]; then
                echo "${upper_ops_nn_platform}_INSTALL_PATH_VAL=${path_param}" >> ${file_param}
        fi
        if [ "$path_params" != "" ]; then
            echo "${upper_ops_nn_platform}_INSTALL_PATH_PARAM=${path_params}" >> ${file_param}
        fi

        if [ "$version_param" != "" ]; then
            echo "${upper_ops_nn_platform}_VERSION=${version_param}" >> ${file_param}
        fi
    fi
}

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${upper_ops_nn_platform}_INSTALL_TYPE"
KEY_INSTALLED_FEATURE="${upper_ops_nn_platform}_Install_Feature"
KEY_INSTALLED_CHIP="${upper_ops_nn_platform}_INSTALL_CHIP"
KEY_INSTALLED_PATH="${upper_ops_nn_platform}_INSTALL_PATH_VAL"
KEY_INSTALLED_VERSION="${upper_ops_nn_platform}_VERSION"

getinstalledinfo() {
    _key="$1"
    _res=""
    if [ -f "${_INSTALL_INFO_FILE}" ]; then
        chmod 644 "${_INSTALL_INFO_FILE}"> /dev/null 2>&1
        checkascendinfo
        case "${_key}" in
        USERNAME)
            res=$(cat ${_INSTALL_INFO_FILE} | grep "USERNAME" | awk -F = '{print $2}')
            ;;
        USERGROUP)
            res=$(cat ${_INSTALL_INFO_FILE} | grep "USERGROUP" | awk -F = '{print $2}')
            ;;
        ${upper_ops_nn_platform}_INSTALL_TYPE)
            type="INSTALL_TYPE"
            res=$(cat ${_INSTALL_INFO_FILE} | grep "${type}" | awk -F = '{print $2}')
            ;;
        ${upper_ops_nn_platform}_INSTALL_PATH_VAL)
            val="INSTALL_PATH_VAL"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${val} | awk -F = '{print $2}')
            ;;
        ${upper_ops_nn_platform}_VERSION)
            version="VERSION"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${version} | awk -F = '{print $2}')
            ;;
        ${upper_ops_nn_platform}_INSTALL_PATH_PARAM)
            param="INSTALL_PATH_PARAM"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${param} | awk -F = '{print $2}')
            ;;
        esac
    fi
    echo "${res}"
}

# keys of infos in run package
KEY_RUNPKG_VERSION="Version"
getrunpkginfo() {
    _key_param="$1"
    if [ -f "${_VERSION_INFO_FILE}" ]; then
        . "${_VERSION_INFO_FILE}" 2> /dev/null
        case "${_key_param}" in
        Version)
            echo ${Version}
            ;;
        esac
    fi
}

updateinstallinfo() {
    _key_val="$1"
    _val="$2"
    _is_new_gen="$3"
    _old_val=$(getinstalledinfo "${_key_val}")
    _target_install_dir="${install_version_dir}/$4"
    if [ -f "${_target_install_dir}" ]; then
        chmod 644 "${_target_install_dir}"
        if [ "${_old_val}"x = ""x ] || [ "${_is_new_gen}" = "true" ]; then
            echo "${_key_val}=${_val}" >> "${_target_install_dir}"
        else
            sed -i "/${_key_val}/c ${_key_val}=${_val}" "${_target_install_dir}"
        fi
    else
        echo "${_key_val}=${_val}" > "${_target_install_dir}"
    fi

    chmod 644 "${_target_install_dir}" 2> /dev/null
    if [ "$(id -u)" != "0" ]; then
        chmod 600 "${_target_install_dir}" 2> /dev/null
    fi
}

updatefeatureandchipinfo() {
    _key_val="$1"
    _val="$2"
    # _old_val=$(getinstalledinfo "${_key_val}")
    _is_new_gen="$3"
    _target_install_dir="${install_version_dir}/$4"
    if [ -f "${_target_install_dir}" ]; then
        chmod 644 "${_target_install_dir}"
        if [ "${_is_new_gen}" = "true" ]; then
            echo "${_key_val}=${_val}" >> "${_target_install_dir}"
        else
            grep_res=$(cat ${_target_install_info} | grep "$_key_val")
            if [ "${grep_res}x" = "x" ]; then
                echo "${_key_val}=${_val}" >> "${_target_install_info}"
            else
                sed -i "/${_key_val}/c ${_key_val}=${_val}" "${_target_install_info}"
            fi
        fi
    else
        echo "${_key_val}=${_val}" > "${_target_install_dir}"
    fi

    chmod 644 "${_target_install_dir}" 2> /dev/null
    if [ "$(id -u)" != "0" ]; then
        chmod 600 "${_target_install_dir}" 2> /dev/null
    fi
}

updateinstallinfos() {
    _uname="$1"
    _ugroup="$2"
    _type="$3"
    _path="$4"
    in_feature_new="$5"
    chip_type_new="$6"
    _version=$(getrunpkginfo "${KEY_RUNPKG_VERSION}")
    _target_install_info="${install_version_dir}/${_INSTALL_INFO_SUFFIX}"
    _is_new_gen_param="false"
    if [ ! -f "${_target_install_info}" ]; then
        _is_new_gen_param="true"
    fi
    updateinstallinfo "${KEY_INSTALLED_UNAME}" "${_uname}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
    updateinstallinfo "${KEY_INSTALLED_UGROUP}" "${_ugroup}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
    updateinstallinfo "${KEY_INSTALLED_TYPE}" "${_type}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
    updatefeatureandchipinfo "${KEY_INSTALLED_FEATURE}" "${in_feature_new}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
    updateinstallinfo "${KEY_INSTALLED_PATH}" "${_path}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
    updateinstallinfo "${KEY_INSTALLED_VERSION}" "${_version}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
    updatefeatureandchipinfo "${KEY_INSTALLED_CHIP}" "${chip_type_new}" "${_is_new_gen_param}" "${_INSTALL_INFO_SUFFIX}"
}

aicpuupdateinstallinfo(){
    _uname_param="$1"
    _ugroup_param="$2"
    _type_val="$3"
    _path_param="$4"
    _pre_path="$5"
    _target_install_path="${install_version_dir}/${_pre_path}"
    if [ ! -f "${_target_install_path}" ]; then
        _is_new_gen_res="true"
    fi
    updateinstallinfo "USERNAME" "${_uname_param}" "${_is_new_gen_res}" "${_pre_path}"
    updateinstallinfo "USERGROUP" "${_ugroup_param}" "${_is_new_gen_res}" "${_pre_path}"
    updateinstallinfo "Aicpu_Kernels_Install_Type" "${_type_val}" "${_is_new_gen_res}" "${_pre_path}"
    updateinstallinfo "Aicpu_Kernels_Install_Path_Param" "${_path_param}" "${_is_new_gen_res}" "${_pre_path}"
}

checkfolderexist() {
    _path_val="${1}"
    if [ ! -d "${_path_val}" ]; then
        logandprint "[ERROR]: ERR_NO:${FILE_READ_FAILED};ERR_DES:Installation directroy \
[${_path_val}] does not exist, upgrade failed."
        exit 1
    fi
}

checkfileexist() {
    _path_value="${1}"
    if [ ! -f "${_path_value}" ];then
        logandprint "[ERROR]: ERR_NO:${FILE_READ_FAILED};ERR_DES:The file (${_path_value}) \
does not existed, upgrade failed."
        exit 1
    fi
}

check_group(){
    _ugroup_val="$1"
    _uname_value="$2"
    if [ $(groups "${_uname_value}" | grep "${_uname_value} :" -c) -eq 1 ]; then
        group_user_related=$(groups "${_uname_value}"|awk -F":" '{print $2}'|grep -w "${_ugroup_val}")
    else
        group_user_related=$(groups "${_uname_value}"|grep -w "${_ugroup_val}")
    fi
    if [ "${group_user_related}x" != "x" ];then
        return 0
    else
        return 1
    fi
}

# check user name and user group is valid or not
checkinstallusergroupconditon() {
    _uname_info="$1"
    _ugroup_value="$2"
    check_group "${_ugroup_value}" "${_uname_info}"
    if [ $? -ne 0 ];then
        logandprint "[ERROR]: ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Usergroup ${_ugroup_value} \
not right! Please check the relatianship of user ${_uname_info} and the group ${_ugroup_value}."
        exit 1
    fi
}

checkinstalledtype() {
    _type_param="$1"
    if [ "${_type_param}" != "run" ] &&
    [ "${_type_param}" != "full" ] &&
    [ "${_type_param}" != "devel" ]; then
        logandprint "[ERROR]: ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Install type \
[${_ugroup}] of ops_nn module is not right!"
        exit 1
    fi
}

createsoftlink() {
    _src_path="$1"
    _dst_path="$2"
    if [ -L "$2" ]; then
        logandprint "[WARNING]: Soft link for [ops_nn/ops] is existed. Cannot create new soft link."
        return 0
    fi
    ln -s "${_src_path}" "${_dst_path}" 2> /dev/null
    if [ "$?" != "0" ]; then
        return 1
    else
        return 0
    fi
}

getinstallpath() {
    docker_root_tmp="$(echo "${docker_root}" | sed "s#/\+\$##g")"
    docker_root_regex="$(echo "${docker_root_tmp}" | sed "s#\/#\\\/#g")"
    relative_path=$(echo "${install_version_dir}" | sed "s/^${docker_root_regex}//g" | sed "s/\/\+\$//g")
    return
}

setenv() {
    logandprint "[INFO]: Set the environment path [ export ASCEND_OPS_NN_PATH=${relative_path_val}/${ops_nn_platform_dir} ]."
    if [ "${is_docker_install}" = y ] ; then
        upgrade_option="--docker-root=${docker_root}"
    else
        upgrade_option=""
    fi
    if [ "${is_setenv}" = "y" ];then
        upgrade_option="${upgrade_option} --setenv"
    fi
}

# init input paremeters
_TARGET_INSTALL_PATH="$1"
_TARGET_USERNAME="$2"
_TARGET_USERGROUP="$3"
_CHIP_TYPE="$4"
is_quiet="$5"
is_for_all="$6"
is_setenv="$7"
is_docker_install="$8"
docker_root="$9"
is_install_path="${10}"
is_upgrade="${11}"
in_feature_new="${12}"
chip_type_new="${13}"
pkg_version_dir="${14}"
in_install_for_all=""

logandprint "[INFO]: Command opp_upgrade"

if [ "${is_for_all}" = y ]; then
    in_install_for_all="--install_for_all"
fi

get_version "pkg_version" "$_VERSION_INFO_FILE"
is_multi_version_pkg "pkg_is_multi_version" "$_VERSION_INFO_FILE"

get_package_upgrade_version_dir "upgrade_version_dir" "$_TARGET_INSTALL_PATH" "${ops_nn_platform_dir}"
upgrade_version_dir="$pkg_version_dir"
upgrade_old_version_dir="$pkg_version_dir"
upgrade_install_info="$_TARGET_INSTALL_PATH/$pkg_version_dir/share/info/$ops_base_platform_dir/ascend_install.info"
upgrade_old_install_info="$upgrade_install_info"

if [ "$pkg_is_multi_version" = "true" ]; then
    install_version_dir=${_TARGET_INSTALL_PATH}/${pkg_version_dir}
else
    install_version_dir=${_TARGET_INSTALL_PATH}
fi

getinstallpath
relative_path_val=${relative_path}

# check input parameters is valid
if [ "${_TARGET_INSTALL_PATH}" = "" ] || [ "${_TARGET_USERNAME}" = "" ] ||
[ "${_TARGET_USERGROUP}" = "" ] || [ "${is_quiet}" = "" ]; then
    logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Empty paramters is invalid for upgrade."
    exit 1
fi

# init log file path
_UNINSTALL_SHELL_FILE="${install_version_dir}""/share/info/${ops_nn_platform_dir}/script/opp_uninstall.sh"
# adpter for old version's path
if [ ! -f "${_UNINSTALL_SHELL_FILE}" ]; then
    _UNINSTALL_SHELL_FILE="${install_version_dir}""/share/info/${ops_nn_platform_dir}/scripts/opp_uninstall.sh"
fi

_INSTALL_INFO_FILE="${install_version_dir}/${_INSTALL_INFO_SUFFIX}"
_IS_ADAPTER_MODE="false"
if [ ! -f "${_INSTALL_INFO_FILE}" ]; then
    _INSTALL_INFO_FILE="/etc/ascend_install.info"
    _IS_ADAPTER_MODE="true"
fi

_TARGET_USERNAME=$(getinstalledinfo "${KEY_INSTALLED_UNAME}")
_TARGET_USERGROUP=$(getinstalledinfo "${KEY_INSTALLED_UGROUP}")

# check install conditons by specific install path
install_type=$(getinstalledinfo "${KEY_INSTALLED_TYPE}")
if [ "${install_type}" = "" ]; then
    logwitherrorlevel "1" "error" "[ERROR]: ERR_NO:${UPGRADE_FAILED};ERR_DES:OPS_NN module\
 is not installed or directory is wrong."
fi
checkinstallusergroupconditon "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
checkinstalledtype "${install_type}"
checkfileexist "${_FILELIST_FILE}"
checkfileexist "${_COMMON_PARSER_FILE}"
# check the ops_nn module sub directory exist or not
opp_sub_dir="${_TARGET_INSTALL_PATH}/${upgrade_version_dir}""/share/info/${ops_nn_platform_dir}/"
old_opp_sub_dir="${_TARGET_INSTALL_PATH}/${upgrade_old_version_dir}""/share/info/${ops_nn_platform_old_dir}/"
if [ -d ${old_opp_sub_dir}/built-in ]; then
    opp_sub_dir=${old_opp_sub_dir}
fi
checkfolderexist "${opp_sub_dir}"

logandprint "[INFO]: upgradePercentage:10%"

logandprint "[INFO]: Begin upgrade ops_nn module."
logandprint "[INFO]: Uninstall the Existed ops_nn module before upgrade."
# call uninstall functions
if [ "${_IS_ADAPTER_MODE}" = "true" ]; then
    if [ ! -f "${_UNINSTALL_SHELL_FILE}" ]; then
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_UNINSTALL_SHELL_FILE}) not exists. Please make sure that the ops_nn module\
 installed in (${install_version_dir}) and then set the correct install path."
        exit 1
    fi
    #sh "${_UNINSTALL_SHELL_FILE}" "${install_path}" "uninstall"
    sh "${_UNINSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "uninstall"
    _IS_FRESH_ISNTALL_DIR="1"
    sh "${_INSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}" "${install_type}" "${is_quiet}" "${_IS_FRESH_ISNTALL_DIR}"
    if [ "$?" = 0 ]; then
        exit 0
    else
        exit 1
    fi
else
    if [ "$pkg_is_multi_version" = "true" ] && [ "${is_upgrade}" = "y" ]; then
        if [ -f "${_TARGET_INSTALL_PATH}/${upgrade_old_version_dir}/${ops_nn_platform_old_dir}/script/uninstall.sh" ]; then
            ${_TARGET_INSTALL_PATH}/${upgrade_old_version_dir}/${ops_nn_platform_old_dir}/script/uninstall.sh
            if [ "$?" = 0 ]; then
                logandprint "[INFO]: Uninstall last ops_nn version module successfully."
            else
                logandprint "[ERROR]: Uninstall last ops_nn version module fail."
                exit 1
            fi
        elif [ -f "${_TARGET_INSTALL_PATH}/${upgrade_version_dir}/ops_nn/script/uninstall.sh" ]; then
            ${_TARGET_INSTALL_PATH}/${upgrade_version_dir}/ops_nn/script/uninstall.sh
            if [ "$?" = 0 ]; then
                logandprint "[INFO]: Uninstall last ops_nn version module successfully."
            else
                logandprint "[ERROR]: Uninstall last ops_nn version module fail."
                exit 1
            fi
        fi
    else
        if [ -f "${_TARGET_INSTALL_PATH}/${upgrade_old_version_dir}/${ops_nn_platform_old_dir}/script/uninstall.sh" ]; then
            ${_TARGET_INSTALL_PATH}/${upgrade_old_version_dir}/${ops_nn_platform_old_dir}/script/uninstall.sh
        else
            sh "${_UNINSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "upgrade" "${is_quiet}" "All" "${is_docker_install}" "${docker_root}"
        fi
    fi
fi
_BUILTIN_PERM="550"
_CUSTOM_PERM="750"
_ONLYREAD_PERM="440"
if [ "${is_for_all}" = y ]; then
    _BUILTIN_PERM="555"
    _CUSTOM_PERM="755"
    _ONLYREAD_PERM="444"
fi
if [ "$(id -u)" != "0" ]; then
    _INSTALL_INFO_PERM="600"
else
    _INSTALL_INFO_PERM="644"
fi
# change permission for install folders
is_change_dir_mode="false"
if [ "$(id -u)" != 0 ] && [ ! -w "${_TARGET_INSTALL_PATH}" ]; then
    chmod u+w "${_TARGET_INSTALL_PATH}" 2> /dev/null
    is_change_dir_mode="true"
fi
logandprint "[INFO]: Update the ops_nn install info."

if [ "${in_feature_new}" = "" ]; then
    in_feature_1="--feature=all"
else
    in_feature_1="--feature=${in_feature_new}"
fi

if [ "${chip_type_new}" = "" ]; then
    chip_type_1="--chip=all"
else
    chip_type_1="--chip=${chip_type_new}"
fi

updateinstallinfos "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}" "${install_type}" "${relative_path_val}"  "${in_feature_new}" "${chip_type_new}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Update ops_nn install info failed."

logandprint "[INFO]: upgradePercentage:30%"
# create copy and chmod ops_nn module path
logandprint "[INFO]: Install ops_nn module path in the install folder."
setenv

sh "${_COMMON_PARSER_FILE}" --package="${ops_nn_platform_dir}" --install --username="${_TARGET_USERNAME}" --usergroup="${_TARGET_USERGROUP}" --set-cann-uninstall \
    --use-share-info --version=${pkg_version} --version-dir=$pkg_version_dir $upgrade_option ${in_install_for_all} ${in_feature_1} ${chip_type_1} "${install_type}" "${_TARGET_INSTALL_PATH}" "${_FILELIST_FILE}"
logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Install ops_nn module files failed."
logandprint "[INFO]: upgradePercentage:50%"

# change installed folder's permission except aicpu
if [ -f "${install_version_dir}/${ops_nn_platform_dir}" ]; then
    subdirs_info=$(ls "${install_version_dir}/${ops_nn_platform_dir}" 2> /dev/null)
    for dir in ${subdirs_info}; do
        if [ "${dir}" != "Ascend310" ] && [ "${dir}" != "Ascend310RC" ] && [ "${dir}" != "Ascend910" ] && [ "${dir}" != "Ascend310P" ] && [ "${dir}" != "Ascend" ] &&  [ "${dir}" != "aicpu" ] && [ "${dir}" != "script" ]; then
            chmod -R "${_CUSTOM_PERM}" "${install_version_dir}/ops_nn/${dir}" 2> /dev/null
        fi
    done
    chmod "${_CUSTOM_PERM}" "${install_version_dir}/${ops_nn_platform_dir}" 2> /dev/null
    logwitherrorlevel "$?" "error" "[ERROR]: ERR_NO:${UPGRADE_FAILED};ERR_DES:Uninstall the \
    installed directory (${install_version_dir}) failed."
fi

logandprint "[INFO]: Update the ops_nn install info."

# change installed folder's permission except aicpu
subdirs=$(ls "${install_version_dir}/${ops_nn_platform_dir}" 2> /dev/null)
for dir in ${subdirs}; do
    if [ "${dir}" != "vendors" ] && [ "${dir}" != "Ascend310" ] && [ "${dir}" != "Ascend310RC" ] && [ "${dir}" != "Ascend910" ] && [ "${dir}" != "Ascend310P" ] && [ "${dir}" != "Ascend" ] &&  [ "${dir}" != "aicpu" ] && [ "${dir}" != "script" ] && [ "${dir}" != "static_kernel" ]; then
        chmod -R "${_BUILTIN_PERM}" "${install_version_dir}/${ops_nn_platform_dir}/${dir}" 2> /dev/null
    fi
done

if [ "$(id -u)" = "0" ]; then
    chmod "755" "${install_version_dir}/${ops_nn_platform_dir}" 2> /dev/null
else
    chmod "${_BUILTIN_PERM}" "${install_version_dir}/${ops_nn_platform_dir}" 2> /dev/null
fi

vendor_dir=$(ls "${install_version_dir}/${ops_nn_platform_dir}/vendors" 2> /dev/null)
if [ -d "$vendor_dir" ]; then
    chmod -R "${_CUSTOM_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/vendors/$vendor_dir/framework/" 2> /dev/null
    chmod -R "${_CUSTOM_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/vendors/$vendor_dir/fusion_pass/" 2> /dev/null
    chmod -R "${_CUSTOM_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/vendors/$vendor_dir/fusion_rules/" 2> /dev/null
    chmod -R "${_CUSTOM_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/vendors/$vendor_dir/op_impl/" 2> /dev/null
    chmod -R "${_CUSTOM_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/vendors/$vendor_dir/op_proto/" 2> /dev/null
fi
chmod "${_ONLYREAD_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/scene.info" 2> /dev/null
chmod "${_ONLYREAD_PERM}" "${install_version_dir}""/${ops_nn_platform_dir}/version.info" 2> /dev/null
chmod 600 "${install_version_dir}""/${ops_nn_platform_dir}/ascend_install.info" 2> /dev/null

if [ "${is_change_dir_mode}" = "true" ]; then
    chmod u-w "${_TARGET_INSTALL_PATH}" 2> /dev/null
fi

# change installed folder's owner and group except aicpu
subdirs=$(ls "${install_version_dir}/${ops_nn_platform_dir}" 2> /dev/null)
for dir in ${subdirs}; do
    if [ "${dir}" != "Ascend310" ] && [ "${dir}" != "Ascend310RC" ] && [ "${dir}" != "Ascend910" ] && [ "${dir}" != "Ascend310P" ] && [ "${dir}" != "Ascend" ] && [ "${dir}" != "aicpu" ] && [ "${dir}" != "script" ]; then
        chown -R "${_TARGET_USERNAME}":"${_TARGET_USERGROUP}" "${install_version_dir}/ops_nn/${dir}" 2> /dev/null
    fi
done

#chmod to support copy
if [ -d "${install_version_dir}/${ops_nn_platform_dir}/vendors" ] && [ "$(id -u)" != "0" ]; then
    chmod -R "${_CUSTOM_PERM}" ${install_version_dir}/${ops_nn_platform_dir}/vendors
fi

logandprint "[INFO]: upgradePercentage:100%"
logandprint "[INFO]: Installation information listed below:"
logandprint "[INFO]: Install log file path: (${_INSTALL_LOG_FILE})"
logandprint "[INFO]: Operation log file path: (${_OPERATE_LOG_FILE})"
logandprint "[INFO]: OpsNN package upgraded successfully! The new version takes effect immediately."
exit 0

