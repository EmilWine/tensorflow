#!/bin/bash


if [ -z "$MKLROOT" ]
then
	echo "NO MKLROOT defined"
	exit 1
fi


MKLML_PREFACE="mklml_lnx_2019.0.5.20190502"
echo "######   Preparing MKL Library for compilation   ######"
mkdir -p ./mklml
wget -N -nc https://github.com/intel/mkl-dnn/releases/download/v0.21/${MKLML_PREFACE}.tgz -O mklml.tgz
tar -C ./mklml -xzvf ./mklml.tgz
ln -s $MKLROOT/include/* mklml/${MKLML_PREFACE}/include/ 2>/dev/null
ln -s $MKLROOT/lib/intel64/* mklml/${MKLML_PREFACE}/lib/ 2>/dev/null 

export TF_MKL_ROOT="${PWD}/mklml/${MKLML_PREFACE}"


JOBS=$(($(grep -c processor /proc/cpuinfo)-2))
#--cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"


DBG="--compilation_mode=dbg -c dbg --copt=-g --copt=-O0 --cxxopt=-O0 --cxxopt=-g --strip=never -s"
bazel build --jobs ${JOBS} --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_USE_BLAS" --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops/insoundz_ops/tf_insoundz/rankoneupdate:python/ops/_rankoneupdate_ops.so

#bazel build --jobs ${JOBS} --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_USE_BLAS" --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops/insoundz_ops/tf_insoundz/matrix_scale:python/ops/_matrix_scale_ops.so

#bazel build --jobs ${JOBS} --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_USE_BLAS" --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops/insoundz_ops/tf_insoundz/hermitian_matmul:python/ops/_hermitian_matmul_ops.so
#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops:build_pip_pkg
#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops/rankoneupdate:rankoneupdate_op.so
#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops:rankoneupdate_op.so

#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops:time_two_kernels.so &&
#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" -s //tensorflow/core/user_ops:emilsky_op.so
#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops:assign_variable_test2_op.so &&
#bazel build --jobs ${JOBS} -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --config=mkl --cxxopt="-DEIGEN_MKL_DEFAULT" --cxxopt="-DEIGEN_USE_MKL_ALL" $DBG //tensorflow/core/user_ops:assign_variable_test_op.so 
