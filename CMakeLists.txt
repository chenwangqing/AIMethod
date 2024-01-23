
# CMake 最低版本号要求
cmake_minimum_required(VERSION 3.5)

# 项目信息
project(main VERSION 1.0.0)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
SET(CMAKE_SYSTEM_NAME Linux)

SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fno-exceptions ") # 设置c++的编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 ") # 设置c的编译选项

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/build)

# -U_FORTIFY_SOURCE     禁用运输时栈检查
# -Wno-unused-result    忽略返回值
# -fPIC                 连接生成的动态库
# -fvisibility=hidden   默认不公开函数
# -fno-exceptions       不使用异常
# -fno-rtti             不需要RTTI
add_definitions(" -Wall -lc -lgcc -D_GNU_SOURCE -Wno-unused-result -Wformat=0 -U_FORTIFY_SOURCE -fPIC -fvisibility=hidden -fexceptions ")

# 允许GDB 调试
set(CMAKE_BUILD_TYPE "Debug")

# 添加调试选项
add_definitions("-g -ggdb")

add_definitions("-DCFG_INFER_ENGINE=1")

set(EXTRA_LIBS 
    libpthread.so
    libonnxruntime.so
    libopencv_core.so
    libopencv_highgui.so
    libopencv_imgproc.so
    libopencv_imgcodecs.so
    libopencv_dnn.so
    libopencv_flann.so
    libopencv_stitching.so
    libopencv_video.so
    libopencv_photo.so
    libopencv_calib3d.so
    libopencv_features2d.so
    libopencv_ml.so
    libopencv_objdetect.so)

set(EXTRA_LIBS_DIR /usr/local/lib)

set(INCLUDE_DIR 
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    /usr/local/include/onnxruntime
    /usr/local/include/opencv4)

set(CODE_SRCS 
    Tools.cpp
    Tools.CV.cpp
    Tensor.cpp
    Ratiocinate.cpp
    TargetDetection.cpp
    TargetSegmention.cpp
    PoseEstimation.cpp
    main.cpp)

# 头文件
include_directories(${INCLUDE_DIR})

# lib
link_directories(${EXTRA_LIBS_DIR})

# 输出执行程序
add_executable(${PROJECT_NAME} ${CODE_SRCS})
target_link_libraries(${PROJECT_NAME} -static-libgcc -static-libstdc++) # 静态链接到glibc
target_link_libraries(${PROJECT_NAME} ${EXTRA_LIBS})