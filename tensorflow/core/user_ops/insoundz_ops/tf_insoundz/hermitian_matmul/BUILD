package(default_visibility = ["//visibility:public"])
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
		name = "python/ops/_hermitian_matmul_ops.so",
		srcs = ["cc/kernels/hermitian_matmul_kernel.cc",
			"cc/ops/hermitian_matmul_ops.cc", 
	  	],
		deps = 
		["//tensorflow/core:framework",
		  "//tensorflow/core:lib",
		  "//tensorflow/core/kernels:linalg_ops_common"
	  	])

py_library(
    name = "hermitian_matmul_ops_py",
    srcs = ([
        "python/ops/hermitian_matmul_ops.py",
    ]),
    data = [
        ":python/ops/_hermitian_matmul_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "hermitian_matmul_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":hermitian_matmul_ops_py"
    ],
    srcs_version = "PY2AND3",
)

