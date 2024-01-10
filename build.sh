g++ Tools.cpp main.cpp Tools.CV.cpp Ratiocinate.cpp TargetDetection.cpp \
    -g -o main -std=c++14 \
    -I/usr/local/include/onnxruntime \
    -I/usr/local/include/opencv4 \
    -L/usr/local/lib \
    -lonnxruntime \
    -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

du -sh main