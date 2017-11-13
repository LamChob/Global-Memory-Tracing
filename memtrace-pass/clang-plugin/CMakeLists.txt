#find_package(LLVM 4.0 CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(LLVM_ENABLE_PLUGINS ON)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS} .)

add_library(Memtrace-AA MODULE AugmentApp.cpp)
set_target_properties(Memtrace-AA PROPERTIES COMPILE_FLAGS "-std=c++11 -fno-rtti")

# cuda path
set(CUDA_PATH /opt/cuda-8.0)

#add_test(Clang-Plugin ${CMAKE_SOURCE_DIR}/testScripts/clangtest.sh ${CMAKE_SOURCE_DIR}/testApps ${CMAKE_BINARY_DIR}/lib/libMekong-AK.so )

enable_testing()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# installation
install(TARGETS Memtrace-AA LIBRARY DESTINATION lib)
