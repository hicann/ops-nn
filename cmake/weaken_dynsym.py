#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""weaken_dynsym.py — 将 .so 的 .dynsym 表中指定未定义符号从 GLOBAL 改为 WEAK

用法: python3 weaken_dynsym.py <so_file> <symbol_name> [<symbol_name> ...]

原理: dlopen(RTLD_NOW) 使用 .dynsym 表解析符号。
      GLOBAL 未定义符号找不到时 -> dlopen 失败。
      WEAK 未定义符号找不到时 -> 解析为 NULL, dlopen 成功。
      objcopy --weaken-symbol 只修改 .symtab, 不修改 .dynsym, 对 .so 无效。
      本脚本直接修改 .dynsym 表中目标符号的 st_info 字节, 将 BIND 从 GLOBAL(1) 改为 WEAK(2)。

兼容周期至 CANN 9.2.0（2027-06-30），到期后随旧版本运行时退服下线，
与 weaken_compat_symbols.sh、symbol.cmake 的 POST_BUILD 步骤及 pass 内版本守卫一并移除。
"""

import sys

from elftools.elf.elffile import ELFFile

STB_GLOBAL = 1
STB_WEAK = 2


def weaken_dynsym(so_path, symbol_names):
    with open(so_path, "r+b") as f:
        elf = ELFFile(f)
        dynsym = elf.get_section_by_name(".dynsym")
        if dynsym is None:
            print(f"ERROR: .dynsym section not found in {so_path}")
            return False

        dynsym_offset = dynsym.header["sh_offset"]
        dynsym_entsize = dynsym.header["sh_entsize"]
        dynsym_num = dynsym.num_symbols()

        # st_info 在 Elf32_Sym 和 Elf64_Sym 中的偏移不同
        # Elf64_Sym: st_name(4), st_info(1), st_other(1), st_shndx(2), st_value(8), st_size(8) = 24 bytes
        #   st_info at offset 4
        # Elf32_Sym: st_name(4), st_value(4), st_size(4), st_info(1), st_other(1), st_shndx(2) = 16 bytes
        #   st_info at offset 12
        if elf.elfclass == 64:
            st_info_offset_in_entry = 4
        else:
            st_info_offset_in_entry = 12

        modified = 0
        for i in range(dynsym_num):
            sym = dynsym.get_symbol(i)
            name = sym.name
            if name in symbol_names:
                entry_offset = dynsym_offset + i * dynsym_entsize
                info_offset = entry_offset + st_info_offset_in_entry
                # 读取当前 st_info 字节
                f.seek(info_offset)
                old_info_byte = f.read(1)[0]
                old_bind = old_info_byte >> 4
                old_type = old_info_byte & 0xF
                new_bind = STB_WEAK
                new_info_byte = (new_bind << 4) | old_type

                # 写入新的 st_info
                f.seek(info_offset)
                f.write(bytes([new_info_byte]))

                bind_names = {0: "LOCAL", 1: "GLOBAL", 2: "WEAK"}
                print(f"  [{i}] {name}")
                print(f"      st_info: 0x{old_info_byte:02x} -> 0x{new_info_byte:02x}")
                print(
                    f"      bind: {bind_names.get(old_bind, '?')} -> {bind_names.get(new_bind, '?')}"
                )
                modified += 1

        if modified == 0:
            print(f"WARNING: no matching symbols found in .dynsym of {so_path}")
            print("  All undefined symbols:")
            for i in range(dynsym_num):
                sym = dynsym.get_symbol(i)
                if sym["st_shndx"] == "SHN_UNDEF" and sym.name:
                    print(f"    [{i}] {sym.name}")
            return False

        print(f"Modified {modified} symbol(s) in .dynsym")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <so_file> <symbol_name> [<symbol_name> ...]")
        sys.exit(1)

    so_path = sys.argv[1]
    symbol_names = set(sys.argv[2:])
    print(f"Weakening symbols in {so_path}:")
    for name in symbol_names:
        print(f"  target: {name}")

    ok = weaken_dynsym(so_path, symbol_names)
    sys.exit(0 if ok else 1)
