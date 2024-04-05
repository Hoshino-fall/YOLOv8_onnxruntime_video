cmake_minimum_required(VERSION 3.27)
project(SimpleOnnx_yolo)

set(CMAKE_CXX_STANDARD 14)

set(ONNX_DIR "/home/hoshino/Desktop/2_SimpleOnnx_yolo/onnxruntime-linux-x64-1.15.1")
set(ONNX_INCLUDE ${ONNX_DIR}/include)
set(ONNX_LIB ${ONNX_DIR}/lib/libonnxruntime.so)


find_package(OpenCV)

add_executable(SimpleOnnx_yolo main_run.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNX_INCLUDE} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${ONNX_LIB} ${OpenCV_LIBS})

# copy
file(COPY ${CMAKE_SOURCE_DIR}/yolov8n.onnx DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/2_test_yolo.mp4 DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/coconames.txt DESTINATION ${CMAKE_BINARY_DIR})