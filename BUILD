"BUILD file for mipnerf."

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "gin_files",
    srcs = glob(["configs/*.gin"]),
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    data = [":gin_files"],
    python_version = "PY3",
    visibility = ["//third_party/google_research/google_research/mipnerf_rc:__pkg__"],
    deps = [
        "//experimental/users/barron/mipnerf_rc/internal",
        "//learning/brain/research/jax:gpu_support",
        "//learning/brain/research/jax:tpu_support",
        "//learning/deepmind/xmanager2/client/google",
        "//third_party/py/flax",
        "//third_party/py/flax/metrics:tensorboard",
        "//third_party/py/flax/training",
        "//third_party/py/gin",
        "//third_party/py/jax",
    ],
)

py_binary(
    name = "eval",
    srcs = [
        "eval.py",
    ],
    data = [":gin_files"],
    python_version = "PY3",
    visibility = ["//experimental/users/barron/mipnerf_rc/jaxnerf:__pkg__"],
    deps = [
        "//experimental/users/barron/mipnerf_rc/internal",
        "//learning/brain/research/jax:gpu_support",
        "//learning/brain/research/jax:tpu_support",
        "//learning/deepmind/xmanager2/client/google",
        "//tech/env:envelope_loader",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/flax",
        "//third_party/py/flax/training",
        "//third_party/py/gin",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_hub",
    ],
)
