package(default_visibility = ["//visibility:public"])
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

#cc_binary(
#    name = 'python/ops/_time_two_ops.so',
#    srcs = [
#        "cc/kernels/time_two.h",
#        "cc/kernels/time_two_kernels.cc",
#        "cc/ops/time_two_ops.cc",
#    ],
#    linkshared = 1,
#    deps = [
#        "@local_config_tf//:libtensorflow_framework",
#        "@local_config_tf//:tf_header_lib",
#    ] + if_cuda_is_configured([":time_two_ops_gpu"]),
#    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX11_ABI=0"] + if_cuda(["-DGOOGLE_CUDA=1"])
#)

tf_custom_op_library(
		name = "python/ops/_rankoneupdate_ops.so",
		srcs = ["cc/rankoneupdate_op.cc"],
		deps = 
		["//tensorflow/core:framework",
		  "//tensorflow/core:lib",
		  "//tensorflow/core/kernels:linalg_ops_common"
	  	])

#cc_binary(
#    name = 'python/ops/_rankoneupdate_ops.so',
#    srcs = ["cc/rankoneupdate_op.cc"],
#    linkshared = 1,
#    deps = 
#    ["//tensorflow/core:framework",
#      "//tensorflow/core:lib",
#      "//tensorflow/core/kernels:linalg_ops_common"
#    ],
#    copts = ["-pthread", "-std=c++11", "-D_GLIBCXX_USE_CXX10_ABI=0"])


py_library(
    name = "rankoneupdate_ops_py",
    srcs = ([
        "python/ops/rankoneupdate_ops.py",
    ]),
    data = [
        ":python/ops/_rankoneupdate_ops.so"
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "rankoneupdate_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/ops/__init__.py",
    ]),
    deps = [
        ":rankoneupdate_ops_py"
    ],
    srcs_version = "PY2AND3",
)

