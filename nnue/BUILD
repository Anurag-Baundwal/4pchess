package(default_visibility=[
    "//visibility:public",
])

cc_binary(
    name = "gen_data",
    srcs = ["gen_data.cc"],
    deps = [
        "//:board",
        "//:player",
        "//:types",
        "//:utils",
        ":nnue",
    ],
    linkopts = select({
        "@bazel_tools//tools/cpp:msvc": ["/STACK:134217728"], # 128MB stack size (128 * 1024 * 1024 bytes)
        "//conditions:default": [], # No special stack options for other platforms
    }),
)

cc_library(
    name = "nnue",
    hdrs = ["nnue.h"],
    srcs = ["nnue.cc"],
    deps = [
        "//:types",
    ],
    copts = select({
        "@bazel_tools//tools/cpp:msvc": ["/arch:AVX2"],
        "@bazel_tools//tools/cpp:clang": ["-mavx2"],
        "@bazel_tools//tools/cpp:gcc": ["-mavx2"],
        "//conditions:default": [],
    }),
)


cc_test(
    name = "nnue_test",
    srcs = ["nnue_test.cc"],
    deps = [
        ":nnue",
        "//:board",
        "//:utils",
        "@com_google_googletest//:gtest_main",
    ],
)


