# C++ library + pybind11 module

# Core static library (algorithm code, no Python)
add_library(pypolca_core STATIC
    core/math_ops.cpp
    core/em_engine.cpp
)

set_target_properties(pypolca_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(pypolca_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(pypolca_core PUBLIC Eigen3::Eigen)

# Pybind11 extension module (thin wrapper)
pybind11_add_module(_core bindings.cpp)

# Override pybind11's hidden visibility — GCC 13 on GHA has RTTI issues with it
set_target_properties(_core PROPERTIES CXX_VISIBILITY_PRESET default
                                          VISIBILITY_INLINES_HIDDEN OFF)

target_link_libraries(_core PRIVATE pypolca_core)

# Install the .so into the Python package directory
install(TARGETS _core DESTINATION pypolca)
