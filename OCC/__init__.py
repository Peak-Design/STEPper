import os
from pathlib import Path
import platform

# Version number
PYTHONOCC_VERSION_MAJOR = 7
PYTHONOCC_VERSION_MINOR = 9
PYTHONOCC_VERSION_PATCH = 3

# Empty for official releases, set to -dev, -rc1, etc for development releases
PYTHONOCC_VERSION_DEVEL = ""

VERSION = f"{PYTHONOCC_VERSION_MAJOR}.{PYTHONOCC_VERSION_MINOR}.{PYTHONOCC_VERSION_PATCH}{PYTHONOCC_VERSION_DEVEL}"


def initialize_occt_libraries(occt_essentials_path) -> None:
    """
    Initializes the OCCT libraries by adding all DLL directories to the DLL search path.

    Raises:
        AssertionError: If the OCCT_ESSENTIALS_ROOT environment variable is not set.
    """
    if not os.path.exists(occt_essentials_path):
        raise AssertionError(
            f"OCCT_ESSENTIALS_ROOT({occt_essentials_path}) is not set correctly."
        )

    for root, dirs, files in os.walk(occt_essentials_path):
        if "debug" in root.lower():
            continue
        for file in files:
            if Path(file).suffix.lower() == ".dll":
                os.add_dll_directory(root)
                break


_system = platform.system()
_core_dir = os.path.join(os.path.dirname(__file__), "Core")

if _system == "Windows":
    # For bundled deployment (e.g. Blender addon), DLLs are in OCC/Core/
    if os.path.isdir(_core_dir):
        os.add_dll_directory(_core_dir)
    else:
        try:
            from .config import OCCT_ESSENTIALS_ROOT
            initialize_occt_libraries(occt_essentials_path=OCCT_ESSENTIALS_ROOT)
        except ImportError:
            if "OCCT_ESSENTIALS_ROOT" in os.environ:
                initialize_occt_libraries(
                    occt_essentials_path=os.environ["OCCT_ESSENTIALS_ROOT"]
                )

elif _system == "Linux":
    # On Linux, .so files use $ORIGIN rpath set during packaging.
    # As a fallback, pre-load libs via ctypes if rpath resolution fails.
    if os.path.isdir(_core_dir):
        import ctypes
        for _lib in sorted(os.listdir(_core_dir)):
            if _lib.startswith("libTK") and _lib.endswith(".so"):
                try:
                    ctypes.cdll.LoadLibrary(os.path.join(_core_dir, _lib))
                except OSError:
                    pass

elif _system == "Darwin":
    # On macOS, dylibs use @loader_path set by install_name_tool during
    # packaging.  As a fallback, pre-load them via ctypes.
    if os.path.isdir(_core_dir):
        import ctypes
        for _lib in sorted(os.listdir(_core_dir)):
            if _lib.startswith("libTK") and _lib.endswith(".dylib"):
                try:
                    ctypes.cdll.LoadLibrary(os.path.join(_core_dir, _lib))
                except OSError:
                    pass
