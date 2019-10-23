load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

LINALG_DEPS = [
    "//tensorflow/core/kernels:linalg_ops_common",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
]

tf_custom_op_library(
		name = "emilsky_op.so",
		srcs = ["emilsky_op.cc"],
		deps = LINALG_DEPS
		)
