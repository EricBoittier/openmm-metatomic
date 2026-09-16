cmake_minimum_required(VERSION 3.22)
project(openmm-metatomic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OPENMM_METATOMIC_TORCH "Build the TorchScript (.pt) backend" ON)
set(OPENMM_DIR "" CACHE PATH "OpenMM install prefix (headers + libOpenMM)")

find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_EXECUTABLE OR Python3_VERSION VERSION_GREATER_EQUAL 3.15)
    find_program(_py314 python3.14)
    if(_py314)
        set(Python3_EXECUTABLE ${_py314})
        message(STATUS "Using Python3_EXECUTABLE=${Python3_EXECUTABLE}")
    endif()
endif()

function(openmm_metatomic_detect_prefixes)
    set(_prefixes "")
    if(Python3_EXECUTABLE)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c
                "from pathlib import Path
import sys
out = []
try:
    import metatensor
    out.append(str(Path(metatensor.__file__).resolve().parent / 'lib' / 'cmake'))
except Exception:
    pass
try:
    import torch
    out.append(torch.utils.cmake_prefix_path)
    mm = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
    import metatensor_torch, metatomic_torch
    out.append(str(Path(metatensor_torch.__file__).resolve().parent / f'torch-{mm}' / 'lib' / 'cmake'))
    out.append(str(Path(metatomic_torch.__file__).resolve().parent / f'torch-{mm}' / 'lib' / 'cmake'))
except Exception:
    pass
print(';'.join(out))"
            OUTPUT_VARIABLE _py_prefixes
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_py_prefixes)
            list(APPEND _prefixes ${_py_prefixes})
        endif()
    endif()
    file(GLOB _local_mta_314
        "${CMAKE_CURRENT_SOURCE_DIR}/../metatomic/python/metatomic_core/build/lib.*314*/metatomic/lib/cmake"
    )
    file(GLOB _local_mta
        "${CMAKE_CURRENT_SOURCE_DIR}/../metatomic/python/metatomic_core/build/lib.*/metatomic/lib/cmake"
    )
    if(_local_mta_314)
        list(APPEND _prefixes ${_local_mta_314})
    else()
        list(APPEND _prefixes ${_local_mta})
    endif()
    if(_prefixes)
        list(PREPEND CMAKE_PREFIX_PATH ${_prefixes})
        set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
        message(STATUS "CMAKE_PREFIX_PATH += ${_prefixes}")
    endif()
endfunction()

openmm_metatomic_detect_prefixes()

find_package(metatensor 0.2.4 CONFIG REQUIRED)
find_package(metatomic CONFIG REQUIRED)

if(OPENMM_METATOMIC_TORCH)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "")
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "")
    # nvcc compiler-id links -lcudart_static; Fedora's CUDA package often
    # ships only the shared runtime.
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
    find_package(metatomic_torch CONFIG REQUIRED)
    unset(CMAKE_TRY_COMPILE_TARGET_TYPE)
endif()

set(SPIKE_SOURCES spike/evaluate.cpp)
add_executable(openmm-metatomic-spike ${SPIKE_SOURCES})
target_include_directories(openmm-metatomic-spike PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
)
target_link_libraries(openmm-metatomic-spike PRIVATE metatomic::shared)
if(OPENMM_METATOMIC_TORCH)
    target_compile_definitions(openmm-metatomic-spike PRIVATE OPENMM_METATOMIC_TORCH)
    target_link_libraries(openmm-metatomic-spike PRIVATE metatomic_torch)
endif()

get_target_property(_mta_loc metatomic::shared IMPORTED_LOCATION)
if(_mta_loc)
    get_filename_component(_mta_libdir "${_mta_loc}" DIRECTORY)
    set_property(TARGET openmm-metatomic-spike APPEND PROPERTY BUILD_RPATH "${_mta_libdir}")
endif()
if(OPENMM_METATOMIC_TORCH AND DEFINED TORCH_INSTALL_PREFIX)
    set_property(TARGET openmm-metatomic-spike APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
endif()

set(HARMONIC_PT "${CMAKE_CURRENT_BINARY_DIR}/harmonic.pt")
if(OPENMM_METATOMIC_TORCH AND Python3_EXECUTABLE)
    add_custom_command(
        OUTPUT ${HARMONIC_PT}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/spike/export_harmonic.py ${HARMONIC_PT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/spike/export_harmonic.py
        COMMENT "Export TorchScript harmonic model"
    )
    add_custom_target(harmonic-pt ALL DEPENDS ${HARMONIC_PT})
endif()

enable_testing()
add_test(NAME spike-core COMMAND openmm-metatomic-spike)
if(OPENMM_METATOMIC_TORCH AND Python3_EXECUTABLE)
    add_dependencies(openmm-metatomic-spike harmonic-pt)
    add_test(NAME spike-torch COMMAND openmm-metatomic-spike ${HARMONIC_PT})
endif()

if(OPENMM_DIR)
    if(NOT EXISTS "${OPENMM_DIR}/include/openmm/internal/CustomCPPForceImpl.h")
        message(FATAL_ERROR "OPENMM_DIR=${OPENMM_DIR} does not look like an OpenMM install")
    endif()
    add_library(OpenMMMetatomic SHARED
        openmmapi/src/MetatomicForce.cpp
        openmmapi/src/MetatomicForceImpl.cpp
        openmmapi/src/MetatomicEvaluator.cpp
        serialization/src/MetatomicForceProxy.cpp
        serialization/src/SerializationProxyRegistration.cpp
    )
    target_compile_definitions(OpenMMMetatomic PRIVATE OPENMM_METATOMIC_BUILDING_SHARED_LIBRARY)
    target_include_directories(OpenMMMetatomic PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
        ${CMAKE_CURRENT_SOURCE_DIR}/serialization/include
        ${OPENMM_DIR}/include
    )
    target_link_libraries(OpenMMMetatomic PUBLIC metatomic::shared)
    target_link_libraries(OpenMMMetatomic PRIVATE "${OPENMM_DIR}/lib/libOpenMM.so")
    if(OPENMM_METATOMIC_TORCH)
        target_compile_definitions(OpenMMMetatomic PRIVATE OPENMM_METATOMIC_TORCH)
        target_link_libraries(OpenMMMetatomic PRIVATE metatomic_torch)
    endif()
    set_target_properties(OpenMMMetatomic PROPERTIES
        BUILD_RPATH "${_mta_libdir};${OPENMM_DIR}/lib"
    )
    install(TARGETS OpenMMMetatomic DESTINATION lib/plugins)
endif()
