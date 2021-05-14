"""BUILD file for internal mip-NeRF libraries."""

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "internal",
    srcs = [
        "datasets.py",
        "models.py",
        "utils.py",
        "vis.py",
    ],
    deps = [
        ":mip",
        "//learning/brain/research/jax:gpu_support",
        "//pyglib:gfile",
        "//third_party/py/PIL:pil",
        "//third_party/py/cvx2",
        "//third_party/py/flax",
        "//third_party/py/jax",
        "//third_party/py/matplotlib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "math_test",
    srcs = [
        "math_test.py",
    ],
    python_version = "PY3",
    deps = [
        ":math",
        "//testing/pybase",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "math",
    srcs = [
        "math.py",
    ],
    deps = [
        "//learning/brain/research/jax:gpu_support",
        "//third_party/py/jax",
    ],
)

py_test(
    name = "mip_test",
    srcs = [
        "mip_test.py",
    ],
    python_version = "PY3",
    tags = [
        "requires-gpu-nvidia",
    ],
    deps = [
        ":math",
        ":mip",
        "//testing/pybase",
    ],
)

py_library(
    name = "mip",
    srcs = [
        "mip.py",
    ],
    deps = [
        ":math",
        "//learning/brain/research/jax:gpu_support",
        "//third_party/py/jax",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    deps = [
        "//learning/brain/research/jax:gpu_support",
        "//third_party/py/cvx2",
        "//third_party/py/dataclasses",
        "//third_party/py/flax",
        "//third_party/py/gin",
        "//third_party/py/jax",
    ],
)
