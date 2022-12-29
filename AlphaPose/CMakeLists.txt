cmake_minimum_required(VERSION 3.23)
project(AlphaPose)

set(CMAKE_CXX_STANDARD 17)

#set(ONNXRUNTIME_ROOTDIR "VALUE" CACHE STRING "ONNX Lib Dir")

set(ONNXRUNTIME_ROOTDIR "D:/cv/project/onnx/onnxruntime-win-x64-1.13.1")
include_directories("${ONNXRUNTIME_ROOTDIR}/include")
link_directories("${ONNXRUNTIME_ROOTDIR}/lib")

# 配置OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(WARNING "Not open openmp!")
endif()

# 配置OpenCV
if(WIN32)
    set(OpenCV_DIR "D:/lib/opencv/build/x64/vc15/lib")
    find_package(OpenCV REQUIRED)
elseif(UNIX AND NOT APPLE)
    find_package(OpenCV REQUIRED)
else()
    message(FATAL_ERROR "Not support for ${CMAKE_SYSTEM_NAME}!")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(AlphaPose)
file(GLOB sources CONFIGURE_DEPENDS src/*.cpp src/*.h)
target_sources(AlphaPose PUBLIC main.cpp ${sources})
