project(sepass)
cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0057 NEW)

# Load LLVM Package from path specified in -DLLVM_PATH= exclusively
find_package(LLVM REQUIRED CONFIG)
# Add LLVM cmake directory to CMake search path, allows including
# LLVM CMake modules
list(APPEND CMAKE_MODULE_PATH "/opt/mekong/lib/cmake/llvm")
# The next one one is SO important, sets CXX+LD flags according
# to the build configuration of the LLVM installation you're building
# against
include(HandleLLVMOptions)
# This modules contains the macro add_llvm_loadable_module among others
include(AddLLVM)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Weffc++ -pedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-DDEBUG -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Echo selected LLVM installation, different installations are usually
# NOT compatible
message(STATUS "Using LLVM installation '${LLVM_DIR}'")

set(LLVM_ENABLE_PLUGINS ON)
set(LLVM_PLUGIN_EXT ".so")

# Add a target for our custom pass, very similar to add_library
#add_llvm_loadable_module(memtrace-pass src/memtrace-pass.cpp)

# Configures a file by creating a copy and replacing CMake variables
# surrounded by '@' with their value. Example:
# @LLVM_TOOLS_BINARY_DIR@/clang++ -> /opt/llvm-3.9/bin/clang++ (path may vary)
#configure_file(run-memtrace.sh run.sh @ONLY)

add_subdirectory(clang-plugin)
add_subdirectory(llvm-pass)

# Variable for 755 permssions
set(PERMS_EXECUTABLE OWNER_WRITE OWNER_READ OWNER_EXECUTE
		     GROUP_READ GROUP_EXECUTE
		     WORLD_READ WORLD_EXECUTE)

# Install run.sh into /bin directory of CMAKE_INSTALL_PREFIX and add executable
# permissions on 'make install' or 'ninja install'. Only required for targets
# other than add_library, add_executable, add_llvm_loadable_module etc.
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/run-memtrace.sh 
	DESTINATION build
	PERMISSIONS ${PERMS_EXECUTABLE})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/utils/TraceUtils.h
	DESTINATION build
	PERMISSIONS ${PERMS_EXECUTABLE})
