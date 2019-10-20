licenses(["notice"])  # 3-Clause BSD

exports_files(["license.txt"])

filegroup(
    name = "LICENSE",
    srcs = [
        "license.txt",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_headers",
    srcs = glob(["include/*(.cc|.cpp|.cxx|.c++|.C|.c|.h|.hh|.hpp|.ipp|.hxx|.inc|.S|.s|.asm|.a|.lib|.pic.a|.lo|.lo.lib|.pic.lo|.so|.dylib|.dll|.o|.obj|.pic.o)"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
)

#"lib/libmklml_intel.so",
#"lib/libiomp5.so",
cc_library(
    name = "mkl_libs_linux",
    srcs = [
        "lib/libiomp5.so",
        "lib/libmkl_intel_lp64.so",
        "lib/libmkl_core.so",
        "lib/libmkl_sequential.so",
        "lib/libmkl_rt.so",
	"lib/libmkl_def.so",
	"lib/libmkl_avx2.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_darwin",
    srcs = [
        "lib/libiomp5.dylib",
        "lib/libmklml.dylib",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_libs_windows",
    srcs = [
        "lib/libiomp5md.lib",
        "lib/mklml.lib",
    ],
    linkopts = ["/FORCE:MULTIPLE"],
    visibility = ["//visibility:public"],
)
