cmake_minimum_required(VERSION 3.20)
project(stepper_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Find Python (must match Blender's embedded Python 3.13) ──────────
find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Development NumPy)

# ── Find OpenCascade ─────────────────────────────────────────────────
# Only request the components we actually use (FoundationClasses, ModelingData,
# ModelingAlgorithms, DataExchange).  Loading ALL components pulls in
# Visualization which requires VTK — we don't need that.
find_package(OpenCASCADE REQUIRED COMPONENTS FoundationClasses ModelingData ModelingAlgorithms DataExchange)

# ── Build the extension module ───────────────────────────────────────
Python3_add_library(stepper_native MODULE stepper_native.cpp)

target_include_directories(stepper_native PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
    ${OpenCASCADE_INCLUDE_DIR}
)

target_link_libraries(stepper_native PRIVATE
    TKernel TKMath TKBRep TKTopAlgo TKG3d
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
