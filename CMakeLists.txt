# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)
include(FetchContent)
project ("OnnxCMake")

FetchContent_Declare(
    onnxruntime
    #URL                  https://github.com/microsoft/onnxruntime/releases/download/v1.7.0/onnxruntime-win-gpu-x64-1.7.0.zip
    #URL_HASH             MD5=75b1c87dbefad6efc78321c8d31312cd
    #URL                  https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-win-gpu-x64-1.8.1.zip
    #URL_HASH             MD5=495442f6f124597635af3cfa96213122  # from PowerShell: CertUtil -hashfile json.hpp MD5
    #URL                  https://github.com/microsoft/onnxruntime/releases/download/v1.11.0/onnxruntime-win-x64-1.11.0.zip
    #URL_HASH             MD5=eaeafd9bd301851c6c3ee8d5fbed2f66  # from PowerShell: CertUtil -hashfile json.hpp MD5
    URL                  https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-win-x64-gpu-1.10.0.zip
    URL_HASH             MD5=4ad7e04899f3a5cc4aa4c8b1c0dbad7a  # from PowerShell: CertUtil -hashfile json.hpp MD5
    DOWNLOAD_NO_PROGRESS True
    DOWNLOAD_DIR         ${CMAKE_BINARY_DIR}/_deps/onnxruntime-download
)
FetchContent_GetProperties(onnxruntime)

if(NOT onnxruntime_POPULATED)
    message("Checking if [onnxruntime] dependency needs to be fetched...")
    FetchContent_Populate(onnxruntime)
    set(ONNXRUNTIME_ROOTDIR ${onnxruntime_SOURCE_DIR})
    set(ONNX_INCLUDE_DIR ${onnxruntime_SOURCE_DIR}/include)
    set(ONNX_LIB_DIR  ${onnxruntime_SOURCE_DIR}/lib/)
    message("ONNX_INCLUDE_DIR set to ${ONNX_INCLUDE_DIR}")
    #add_definitions(-DUSE_CUDA)
    #add_definitions(-DUSE_DML)
endif()

# Include sub-projects.
add_subdirectory ("OnnxCMake")
