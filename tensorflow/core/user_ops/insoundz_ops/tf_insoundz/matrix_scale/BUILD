package(default_visibility = ["//visibility:public"])
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
		name = "python/ops/_matrix_scale_ops.so",
		srcs = ["cc/kernels/matrix_scale_kernel.cc",
			"cc/ops/matrix_scale_ops.cc", 
	  	],
		deps = 
		["//tensorflow/core:framework",
		  "//tensorflow/core:lib",
		  "//tensorflow/core/kernels:linalg_ops_common"
	  	])

py_library(
    name = "matrix_scale_ops_py",
    srcs = ([
        "python/ops/matrix_scale_ops.py",
    ]),
    data = [
        ":python/ops/_matrix_scale_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "matrix_scale_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":matrix_scale_ops_py"
    ],
    srcs_version = "PY2AND3",
)

