cmake_minimum_required(VERSION 3.20)
project(stepper_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find Python (must match Blender's embedded Python 3.13) ──────────
find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Development NumPy)

# ── Find OpenCascade ─────────────────────────────────────────────────
# On conda environments, OCCT is typically found automatically.
# Set OpenCASCADE_DIR if not in a standard location.
find_package(OpenCASCADE REQUIRED)

# ── Build the extension module ───────────────────────────────────────
Python3_add_library(stepper_native MODULE stepper_native.cpp)

target_include_directories(stepper_native PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${OpenCASCADE_INCLUDE_DIR}
)

target_link_libraries(stepper_native PRIVATE
    ${OpenCASCADE_LIBRARIES}
)

# Remove "lib" prefix on all platforms (Python expects bare module name)
set_target_properties(stepper_native PROPERTIES
    PREFIX ""
)

# On Windows, set the output suffix to .pyd
if(WIN32)
    set_target_properties(stepper_native PROPERTIES
        SUFFIX ".pyd"
    )
endif()
