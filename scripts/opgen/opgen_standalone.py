# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import os
import shutil
import sys
import re
import logging


class OpGenerator:
    """算子工程生成器"""

    SOC_ARCH_MAP = {
        "ascend950": "arch35",
    }

    SOC_UT_TARGET_MAP = {
        "ascend950": "ascend950pr_9599",
    }

    def __init__(self, op_type, op_name, output_path, template_variant, soc=None):
        self.op_type = op_type
        self.op_name = op_name
        self.output_path = output_path
        self.soc = soc
        self.template_name = "add_example"

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        if template_variant == "aicpu":
            self.template_dir = os.path.abspath(
                os.path.join(self.script_dir, "template", "add_example_aicpu")
            )
        else:
            self.template_dir = os.path.abspath(
                os.path.join(self.script_dir, "template", "add_example")
            )

        self.dest_dir = os.path.abspath(
            os.path.join(self.output_path, self.op_type, self.op_name)
        )

    def run(self):
        """执行生成流程"""
        self._validate_inputs()
        self._copy_template()
        self._rename_files()
        self._replace_content()
        if self.soc and self.soc in self.SOC_ARCH_MAP:
            self._apply_arch35_structure()
        logging.info(f"成功为 {self.op_type}/{self.op_name} 创建算子工程！")
        logging.info(f"工程路径: {self.dest_dir}")

    def _validate_inputs(self):
        """校验输入参数的有效性和安全性"""
        if not self.op_type or not self.op_name:
            raise ValueError("算子类型和算子名称均不能为空。")

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_type):
            raise ValueError(
                f"算子类型 '{self.op_type}' 包含无效字符。只允许字母、数字和下划线。"
            )

        if not re.match(r"^[a-zA-Z0-9_]+$", self.op_name):
            raise ValueError(
                f"算子名称 '{self.op_name}' 包含无效字符。只允许字母、数字和下划线。"
            )

        if os.path.exists(self.dest_dir):
            raise FileExistsError(f"目标目录 '{self.dest_dir}' 已存在。")

    def _copy_template(self):
        """复制模板文件到目标目录"""
        logging.info(f"使用模板在 '{self.dest_dir}' 创建算子工程...")
        if not os.path.exists(self.template_dir):
            raise FileNotFoundError(
                f"找不到模板目录 '{self.template_dir}'。请确保模板目录存在。"
            )

        try:
            shutil.copytree(self.template_dir, self.dest_dir)
        except OSError as e:
            raise OSError(f"复制模板文件失败: {e}") from e

    def _rename_files(self):
        """重命名文件和目录中的占位符"""
        for root, dirs, files in os.walk(self.dest_dir, topdown=False):
            for name in files + dirs:
                if self.template_name not in name:
                    continue

                old_path = os.path.join(root, name)
                new_name = name.replace(self.template_name, self.op_name)
                new_path = os.path.join(root, new_name)
                try:
                    os.rename(old_path, new_path)
                except OSError as e:
                    raise OSError(
                        f"重命名 '{old_path}' 到 '{new_path}' 失败: {e}"
                    ) from e

    def _replace_content_in_file(self, file_path, replacements):
        """Helper to replace content in a single file."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (IOError, OSError) as e:
            logging.warning(f"读取文件 '{file_path}' 失败: {e}")
            return

        original_content = content
        for old, new in replacements.items():
            content = content.replace(old, new)

        if content == original_content:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except (IOError, OSError) as e:
            logging.warning(f"写入文件 '{file_path}' 失败: {e}")

    def _replace_content(self):
        """替换文件内容中的占位符"""
        op_name_capitalized = "".join(
            word.capitalize() for word in self.op_name.split("_")
        )
        template_name_capitalized = "".join(
            word.capitalize() for word in self.template_name.split("_")
        )

        replacements = {
            self.template_name: self.op_name,
            self.template_name.upper(): self.op_name.upper(),
            template_name_capitalized: op_name_capitalized,
            "add_example": self.op_name,
        }
        for root, _, files in os.walk(self.dest_dir):
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    continue

                file_path = os.path.join(root, file)
                self._replace_content_in_file(file_path, replacements)

    def _apply_arch35_structure(self):
        """对 --soc=ascend950 进行 arch35 目录结构调整"""
        arch_dir = self.SOC_ARCH_MAP[self.soc]
        op_host_arch = os.path.join(self.dest_dir, "op_host", arch_dir)
        op_kernel_arch = os.path.join(self.dest_dir, "op_kernel", arch_dir)
        os.makedirs(op_host_arch, exist_ok=True)
        os.makedirs(op_kernel_arch, exist_ok=True)

        tiling_src = os.path.join(
            self.dest_dir, "op_host", f"{self.op_name}_tiling.cpp"
        )
        tiling_dst = os.path.join(op_host_arch, f"{self.op_name}_tiling.cpp")
        if os.path.exists(tiling_src):
            shutil.move(tiling_src, tiling_dst)
            self._modify_tiling_includes(tiling_dst, arch_dir)

        for fname in [
            f"{self.op_name}.h",
            f"{self.op_name}_tiling_data.h",
            f"{self.op_name}_tiling_key.h",
            f"{self.op_name}.cpp",
        ]:
            src = os.path.join(self.dest_dir, "op_kernel", fname)
            dst = os.path.join(op_kernel_arch, fname)
            if os.path.exists(src):
                shutil.move(src, dst)

        kernel_cmake = os.path.join(self.dest_dir, "op_kernel", "CMakeLists.txt")
        self._write_kernel_arch_cmake(kernel_cmake, arch_dir)

        def_file = os.path.join(self.dest_dir, "op_host", f"{self.op_name}_def.cpp")
        self._modify_def_file(def_file)

        cmake_file = os.path.join(self.dest_dir, "CMakeLists.txt")
        self._modify_cmake_arch35(cmake_file, arch_dir)

        test_file = os.path.join(
            self.dest_dir, "tests", "ut", "op_kernel", f"test_{self.op_name}.cpp"
        )
        self._modify_test_includes(test_file, arch_dir)

        test_cmake = os.path.join(
            self.dest_dir, "tests", "ut", "op_kernel", "CMakeLists.txt"
        )
        self._modify_test_cmake(test_cmake)

    def _write_kernel_arch_cmake(self, file_path, arch_dir):
        """生成 op_kernel/arch35/CMakeLists.txt"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(
                    f"add_kernel_sources(\n"
                    f"    KERNEL_SRC {arch_dir}/{self.op_name}.cpp\n"
                    f"    COMPUTE_UNITS {self.soc}\n"
                    f")\n"
                )
        except (IOError, OSError):
            pass

    def _modify_def_file(self, file_path):
        """修改 def.cpp: 只保留指定 soc 的 AddConfig，移除 ExtendCfgInfo"""
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return

        addconfig_pattern = (
            r'this->AICore\(\)\.AddConfig\("[^"]+",\s*aicoreConfig\);\s*'
        )
        extendcfg_pattern = r'\s*\.ExtendCfgInfo\("[^"]+",\s*"[^"]+"\)'

        lines = content.split("\n")
        new_lines = []
        found_first = False
        for line in lines:
            if re.search(addconfig_pattern, line):
                if not found_first:
                    new_lines.append(
                        f'        this->AICore().AddConfig("{self.soc}", aicoreConfig);'
                    )
                    found_first = True
                continue
            line = re.sub(extendcfg_pattern, "", line)
            new_lines.append(line)
        new_content = "\n".join(new_lines)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except (IOError, OSError):
            pass

    def _modify_cmake_arch35(self, file_path, arch_dir):
        """修改 CMakeLists.txt: 添加 SUPPORT_COMPUTE_UNIT 和 SUPPORT_TILING_DIR，以及 add_subdirectory 逻辑"""
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return

        old_line = (
            f"add_modules_sources(HOSTNAME ${{OPHOST_NAME}} MODE PRIVATE DIR "
            f"${{CMAKE_CURRENT_SOURCE_DIR}} OPTYPE {self.op_name} ACLNNTYPE aclnn)"
        )

        new_block = (
            f'set(SUPPORT_COMPUTE_UNIT "{self.soc}")\n'
            f'set(SUPPORT_TILING_DIR "{arch_dir}")\n'
            f"add_modules_sources(HOSTNAME ${{OPHOST_NAME}} MODE PRIVATE DIR "
            f"${{CMAKE_CURRENT_SOURCE_DIR}} OPTYPE {self.op_name} ACLNNTYPE aclnn "
            f"COMPUTE_UNIT ${{SUPPORT_COMPUTE_UNIT}} TILING_DIR ${{SUPPORT_TILING_DIR}})\n"
        )

        if old_line in content:
            content = content.replace(old_line, new_block)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except (IOError, OSError):
            pass

    def _modify_test_includes(self, file_path, arch_dir):
        """修改测试文件中的 include 路径，指向 op_kernel/arch35/"""
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return

        content = content.replace(
            f"../../../op_kernel/{self.op_name}.cpp",
            f"../../../op_kernel/{arch_dir}/{self.op_name}.cpp",
        )
        content = content.replace(
            f"../../../op_kernel/{self.op_name}_tiling_data.h",
            f"../../../op_kernel/{arch_dir}/{self.op_name}_tiling_data.h",
        )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except (IOError, OSError):
            pass

    def _modify_tiling_includes(self, file_path, arch_dir):
        """修改 tiling 文件的 include 路径，指向 op_kernel/arch35/"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return

        content = content.replace(
            f"{self.op_name}/op_kernel/{self.op_name}_tiling_data.h",
            f"{self.op_name}/op_kernel/{arch_dir}/{self.op_name}_tiling_data.h",
        )
        content = content.replace(
            f"{self.op_name}/op_kernel/{self.op_name}_tiling_key.h",
            f"{self.op_name}/op_kernel/{arch_dir}/{self.op_name}_tiling_key.h",
        )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except (IOError, OSError):
            pass

    def _modify_test_cmake(self, file_path):
        """修改测试 CMakeLists.txt: 替换 soc 版本目标"""
        if self.soc not in self.SOC_UT_TARGET_MAP:
            return
        ut_target = self.SOC_UT_TARGET_MAP[self.soc]
        if not os.path.exists(file_path):
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (IOError, OSError):
            return

        content = re.sub(
            rf'AddOpTestCase\({self.op_name}\s+"[^"]*"',
            f'AddOpTestCase({self.op_name} "{ut_target}"',
            content,
        )

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
        except (IOError, OSError):
            pass


def execute(args):
    """根据命令行参数执行算子生成"""
    generator = OpGenerator(
        op_type=args.op_type,
        op_name=args.op_name,
        output_path=args.output_path,
        template_variant=args.template_variant,
        soc=getattr(args, "soc", None),
    )
    generator.run()


def register_parser(subparsers):
    """为 opgen 命令注册解析器。"""
    parser_opgen = subparsers.add_parser("opgen", help="生成项目骨架")
    parser_opgen.add_argument(
        "--op_type", "-t", required=True, help="算子分类，例如 math"
    )
    parser_opgen.add_argument(
        "--op_name", "-n", required=True, help="新算子的名称，例如 asinh"
    )
    parser_opgen.add_argument(
        "--output_path", "-p", default=".", help="生成工程的根路径"
    )
    parser_opgen.add_argument(
        "--template_variant",
        "-v",
        choices=["default", "aicpu"],
        default="default",
        help="选择模板变种",
    )
    parser_opgen.add_argument("--soc", default=None, help="目标芯片类型，如 ascend950")
    parser_opgen.set_defaults(func=execute)


def main():
    """主函数，用于独立执行"""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout
    )
    parser = argparse.ArgumentParser(description="生成项目骨架")

    parser.add_argument("--op_type", "-t", required=True, help="算子分类，例如 math")
    parser.add_argument(
        "--op_name", "-n", required=True, help="新算子的名称，例如 asinh"
    )
    parser.add_argument("--output_path", "-p", default=".", help="生成工程的根路径")
    parser.add_argument(
        "--template_variant",
        "-v",
        choices=["default", "aicpu"],
        default="default",
        help="选择模板变种",
    )
    parser.add_argument("--soc", default=None, help="目标芯片类型，如 ascend950")

    args = parser.parse_args()

    try:
        execute(args)
    except Exception as e:
        logging.error(f"发生非预期的错误，退出。错误信息: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
