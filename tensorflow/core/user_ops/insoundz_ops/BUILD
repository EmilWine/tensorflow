sh_binary(
    name = "build_pip_pkg",
    srcs = ["build_pip_pkg.sh"],
    data = [
        "setup.py",
	"tf_insoundz/__init__.py",
	"MANIFEST.in",
        "//tensorflow/core/user_ops/insoundz_ops/tf_insoundz/rankoneupdate:rankoneupdate_py",
        "//tensorflow/core/user_ops/insoundz_ops/tf_insoundz/hermitian_matmul:hermitian_matmul_py",
        "//tensorflow/core/user_ops/insoundz_ops/tf_insoundz/matrix_scale:matrix_scale_py",
    ],
)
