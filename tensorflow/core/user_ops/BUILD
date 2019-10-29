load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")


GEN_DEPS = [
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    ]

LINALG_DEPS = [
    "//tensorflow/core/kernels:linalg_ops_common",
]

tf_custom_op_library(
		name = "emilsky_op.so",
		srcs = ["emilsky_op.cc"],
		deps = LINALG_DEPS + GEN_DEPS
		)

tf_custom_op_library(
		name = "assign_variable_test_op.so",
		srcs = ["assign_variable_test_op.cc"],
		deps = 
		["//tensorflow/core/kernels:dense_update_functor",
		"//tensorflow/core/kernels:variable_ops",
		"//tensorflow/core:framework",
		"//tensorflow/core:lib",
		   ],
		)

tf_custom_op_library(
		name = "assign_variable_test2_op.so",
		srcs = ["assign_variable_test2_op.cc"],
		deps = LINALG_DEPS +  GEN_DEPS +
		["//tensorflow/core/kernels:dense_update_functor",
		"//tensorflow/core/kernels:variable_ops",
		   "//tensorflow/core:core_cpu_lib",
		   ],
		)
tf_custom_op_library(
		name = "rankoneupdate_op.so",
		srcs = ["rankoneupdate_op.cc"],
		deps = LINALG_DEPS + GEN_DEPS
		)

tf_custom_op_library(
		name = "time_two_kernels.so",
		srcs = ["time_two_kernels.cc", "time_two.h"],
		deps = LINALG_DEPS + GEN_DEPS +
		["//tensorflow/core/kernels:dense_update_functor"]
		)
