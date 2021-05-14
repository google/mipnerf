"""BUILD file for mip-NeRF unit tests."""

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_test(
    name = "math_test",
    srcs = [
        "math_test.py",
    ],
    python_version = "PY3",
    deps = [
        "//experimental/users/barron/mipnerf_rc/internal:math",
        "//testing/pybase",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
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
        "//experimental/users/barron/mipnerf_rc/internal:math",
        "//experimental/users/barron/mipnerf_rc/internal:mip",
        "//testing/pybase",
        "//third_party/py/numpy",
    ],
)
