#!/bin/bash


if [ -z "$MKLROOT" ]
then
	echo "NO MKLROOT defined"
	exit 1
fi

echo "Patching eigen to support Tensor FFT with MKL"
combinediff ./third_party/eigen3/gpu_packet_math.patch ./third_party/eigen3/tensor_fft.patch > ./third_party/eigen3/gpu_packet_math.patch

MKLML_PREFACE="mklml_lnx_2019.0.5.20190502"
echo "######   Preparing MKL Library for compilation   ######"
mkdir -p ./mklml
wget -nc https://github.com/intel/mkl-dnn/releases/download/v0.21/${MKLML_PREFACE}.tgz -O mklml.tgz
tar -C ./mklml -xzvf ./mklml.tgz
ln -s $MKLROOT/include/* mklml/${MKLML_PREFACE}/include/ 2>/dev/null
ln -s $MKLROOT/lib/intel64/* mklml/${MKLML_PREFACE}/lib/ 2>/dev/null 

export TF_MKL_ROOT="${PWD}/mklml/${MKLML_PREFACE}"


JOBS=$(($(grep -c processor /proc/cpuinfo)-2))
#DBG="--compilation_mode=dbg -c dbg --copt=-g --copt=-O0 --cxxopt=-O0 --cxxopt=-g --strip=never -s"
DBG=""
#--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"

bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl  --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" ${DBG} //tensorflow/tools/pip_package:build_pip_package
. ./bazel-bin/tensorflow/tools/pip_package/build_pip_package ./pip_package/
