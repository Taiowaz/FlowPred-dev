#!/bin/bash

# 获取当前目录名
CURRENT_DIR=$(basename "$(pwd)")
# 定义压缩包存储目录
RES_DIR="build/res"
# 创建存储目录，如果目录不存在
mkdir -p "$RES_DIR"
# 定义压缩包名称，格式为 目录名_时间戳.tar.gz
ARCHIVE_NAME="${CURRENT_DIR}_$(date +%Y%m%d%H%M%S).tar.gz"
# 拼接完整的压缩包路径
FULL_ARCHIVE_PATH="${RES_DIR}/${ARCHIVE_NAME}"

# 执行压缩操作，将 --exclude 选项移到要压缩的目录之前
tar -czvf "$FULL_ARCHIVE_PATH" \
--exclude='./log' \
--exclude='./evaluation' \
--exclude='./data' \
--exclude='./build' \
--exclude='./analyze' \
.

# 检查压缩操作是否成功
if [ $? -eq 0 ]; then
    echo "压缩成功，压缩包路径为 $FULL_ARCHIVE_PATH"
else
    echo "压缩失败，请检查错误信息"
fi
