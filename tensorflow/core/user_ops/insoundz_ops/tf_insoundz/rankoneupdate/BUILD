package(default_visibility = ["//visibility:public"])
load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
		name = "python/ops/_rankoneupdate_ops.so",
		srcs = ["cc/kernels/rankoneupdate_kernel.cc",
			"cc/ops/rankoneupdate_ops.cc",
	  	],
		deps = [
			"//tensorflow/core:framework",
			"//tensorflow/core:lib",
			"//tensorflow/core/user_ops/insoundz_ops/tf_insoundz/common_inz:linear_algebra_inplace",
	  	])

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

py_test(
    name = "rankoneupdate_ops_py_test",
    srcs = [
        "python/ops/rankoneupdate_op_test.py"
    ],
    main = "python/ops/rankoneupdate_op_test.py",
    deps = [
        ":rankoneupdate_ops_py",
    ],
    srcs_version = "PY2AND3",
)


