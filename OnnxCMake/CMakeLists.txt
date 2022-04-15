# CMakeList.txt : CMake project for OnnxCMake, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)
set(TARGET_NAME OnnxCMake)

# Add source to this project's executable.
add_executable (OnnxCMake "OnnxCMake.cpp" "OnnxCMake.h")
target_include_directories(${TARGET_NAME} PRIVATE "${ONNXRUNTIME_ROOTDIR}/include" "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session")
target_link_libraries(${TARGET_NAME} PRIVATE "${ONNXRUNTIME_ROOTDIR}/lib/onnxruntime.lib")

# TODO: Add tests and install targets if needed.
