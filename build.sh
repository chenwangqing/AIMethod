#! /bin/bash
################################################################################
# 名称：编译脚本
# 作者：chenxiangshu@outlook.com
# 日期：2024年1月11日
################################################################################

# 编译文件
FILES=(Tools.cpp
    Tools.CV.cpp
    Tensor.cpp
    Ratiocinate.cpp
    TargetDetection.cpp
    TargetSegmention.cpp
    main.cpp)

# 编译标志
CXX_FLAGS="-g -std=c++14 \
    -I/usr/local/include/onnxruntime \
    -I/usr/local/include/opencv4 \
    -I/usr/include/eigen3"

# 宏定义
DEF_FLAGS="-DCFG_INFER_ENGINE=1"

# 连接标志
LD_FLAGS="-L/usr/local/lib \
    -lonnxruntime -pthread \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn"

# -------------------------------------- 开始编译 --------------------------------------

mkdir -p build
cd build

n=${#FILES[@]}
i=1
lds=""
for file in ${FILES[@]}; do
    echo "编译[$i/$n]: $file"
    bash -c "g++ -c ../$file $CXX_FLAGS $DEF_FLAGS" || exit 1
    i=$((i + 1))
    lds+="${file%.*}.o "
done
echo "生成 main"
bash -c "g++ $lds -o main $LD_FLAGS" || exit 1

du -sh main