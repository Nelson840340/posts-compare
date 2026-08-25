#!/usr/bin/env bash
# macOS arm64：faiss-cpu 与 torch 各自捆绑一份 libomp.dylib，且 install name 不同
# （faiss: /DLC/faiss/.dylibs/libomp.dylib，torch: /opt/llvm-openmp/lib/libomp.dylib），
# dyld 会加载两份 OpenMP 运行时，faiss 检索时 worker 线程 SIGSEGV（exit 139）。
# ctypes 预载 torch libomp 无法去重（install name 不同）。
#
# 修复：把 faiss 两个二进制对 @loader_path/.dylibs/libomp.dylib 的依赖
# 直接改指向 torch 的 libomp 绝对路径，全程仅一份 OpenMP 运行时。
#
# 幂等：已打过补丁时直接跳过。venv 中 faiss-cpu 被重装后需重新执行本脚本。
# 用法：bash scripts/fix_faiss_libomp.sh
set -euo pipefail

SITE="$(python -c 'import faiss, os; print(os.path.join(os.path.dirname(faiss.__file__)))')"
TORCH_OMP="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))')"

OLD="@loader_path/.dylibs/libomp.dylib"
for BIN in "$SITE/_swigfaiss.abi3.so" "$SITE/libfaiss.dylib"; do
    [ -f "$BIN" ] || { echo "跳过（不存在）: $BIN"; continue; }
    if otool -L "$BIN" | grep -q "$OLD"; then
        install_name_tool -change "$OLD" "$TORCH_OMP" "$BIN"
        codesign --force --sign - "$BIN"
        echo "已修补: $BIN -> $TORCH_OMP"
    else
        echo "无需修补（已指向其他 libomp）: $BIN"
    fi
done
echo "完成。"
