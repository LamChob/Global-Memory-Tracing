#!/bin/bash

set -e

CLANGPLUGIN="clang-plugin/libMemtrace-AA.so"
LLVMPLUGIN="llvm-pass/libmemtrace-pass.so"
APPSRCPATH="../application/hotspot"
APPSRC="hotspot.cu"

CUDASYS=/opt/cuda-7.0

# hardwire llvm tool paths into the script because our pass is build against a
# very specific LLVM installation and they are usually not compatible with other
# ones (not even the same version, if the build configuration differs).
cxx=/opt/mekong/bin/clang++
opt=/opt/mekong/bin/opt
opt_opts=

# figure out whether the LLVM installation has assertions enabled
if ( "$opt" -help | grep -q debug-only ); then
  opt_opts=-debug-only=sepass
fi

"$cxx" -Xclang -load -Xclang $CLANGPLUGIN -Xclang -plugin \
    -Xclang cuda-aug -Xclang -plugin-arg-cuda-aug -Xclang -f -Xclang -plugin-arg-cuda-aug -Xclang ./augmented-$APPSRC \
    --cuda-path=$CUDASYS -Wno-writable-strings -std=c++11 -E $APPSRCPATH/$APPSRC -I$CUDASYS/samples/common/inc -I. -I../utils


# -stdlib=libstdc++ is used because a source compiled clang++ does not find the regular
# c++ standard library on macOSX for some reason.
#echo "Building Main"
#"$cxx" -c -L$CUDASYS/lib64 --cuda-path=$CUDASYS -I$CUDASYS/samples/common/inc -O1 -I$CUDASYS/include \
#-o main.o --std=c++11 -I. -I../utils -I../application/histogram/ \
#../application/ 
echo "Building Host Utils"
"$cxx" -c -L$CUDASYS/lib64 --cuda-path=$CUDASYS -I$CUDASYS/samples/common/inc -O1 --cuda-gpu-arch=sm_30 -I$CUDASYS/include \
-o hutils.o --std=c++11 -I. -I../utils \
../utils/TraceUtils.cpp

echo "Building Device Utils"
"$cxx" -c -L$CUDASYS/lib64 --cuda-path=$CUDASYS -I$CUDASYS/samples/common/inc -O1 --cuda-gpu-arch=sm_30 -I$CUDASYS/include \
-o dutils.o --std=c++11 -I. -I../utils \
../utils/DeviceUtils.cu

echo "Building Kernel"
# (only) compile simple.cc to llvm bitcode object
#"$cxx" -c --cuda-path=$CUDASYS -I$CUDASYS/samples/common/inc --cuda-gpu-arch=sm_30 -L$CUDASYS/lib64  -O1 \
#   -lcudart_static -m64 --std=c++11 -I. -I../utils -I../application/histogram/ -emit-llvm -S  \
# ./augmented-$APPSRC 
"$cxx" -c -Xclang -load -Xclang $LLVMPLUGIN  --cuda-path=$CUDASYS -I$CUDASYS/samples/common/inc --cuda-gpu-arch=sm_30 -L$CUDASYS/lib64  -O1 \
-lcudart_static -m64  --std=c++11 -I. -I../utils -I$APPSRCPATH  \
 ./augmented-$APPSRC -Wno-writable-strings

echo "Linking"

# (only) compile simple.cc to llvm bitcode object
"$cxx" -Xclang -load -Xclang $LLVMPLUGIN --cuda-path=$CUDASYS -I$CUDASYS/samples/common/inc --cuda-gpu-arch=sm_30 -L$CUDASYS/lib64  -O1 \
  -lcudart -ldl -lrt -L. -m64 --std=c++11 -I. -I../utils -I$APPSRCPATH  \
 -o hotspot dutils.o hutils.o augmented-hotspot.o 

# (only) compile external.cc to llvm bitcode object
#"$cxx" -c -S -emit-llvm /share/drieber/Master/se_pass/application/external.cpp -o external.bc
# apply embeddata pass to the simple bitcode object, embedding the data used by external.bc
#"$opt" -S $opt_opts -load=./sepass.so -sepass simple.bc > simple.embedded.bc
# generate native code + link to executable
#"$cxx" simple.embedded.bc external.bc -o simple -L/opt/cuda-7.0/lib64 -lcudart 
