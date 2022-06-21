import os


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "version")
    with open(version_file, "r") as f:
        version = f.readlines()
        version = version[0]
        version = version.strip()
    return version


__version__ = get_version()

print(
    f"WARNING: {__file__} is an empty package. Do a git clone an install it with pip locally."
)
