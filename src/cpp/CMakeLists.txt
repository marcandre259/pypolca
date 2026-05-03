# C++ library + pybind11 module

# Core static library (algorithm code, no Python)
add_library(polca_core STATIC
    core/math_ops.cpp
    core/em_engine.cpp
)

target_include_directories(polca_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(polca_core PUBLIC Eigen3::Eigen)

# Pybind11 extension module (thin wrapper)
pybind11_add_module(_core bindings.cpp)

target_link_libraries(_core PRIVATE polca_core)

# Install the .so into the Python package directory
install(TARGETS _core DESTINATION polca)
