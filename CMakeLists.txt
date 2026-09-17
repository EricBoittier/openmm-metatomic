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
    import metatensor.torch as metatensor_torch
    import metatomic.torch as metatomic_torch
    out.append(str(Path(metatensor_torch.__file__).resolve().parent / f'torch-{mm}' / 'lib' / 'cmake'))
    out.append(str(Path(metatomic_torch.__file__).resolve().parent / f'torch-{mm}' / 'lib' / 'cmake'))
except Exception:
    pass
try:
    import sphericart
    out.append(str(Path(sphericart.__file__).resolve().parent / 'lib' / 'cmake'))
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

find_package(metatensor 0.2.5 CONFIG REQUIRED)
find_package(metatomic CONFIG REQUIRED)

function(openmm_metatomic_import_torch)
    if(NOT Python3_EXECUTABLE)
        message(FATAL_ERROR "Python3 is required to locate the TorchScript backend")
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "from pathlib import Path
import torch
import metatensor.torch as metatensor_torch
import metatomic.torch as metatomic_torch
mm = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
print(Path(torch.__file__).resolve().parent)
print(Path(metatensor_torch.__file__).resolve().parent / f'torch-{mm}')
print(Path(metatomic_torch.__file__).resolve().parent / f'torch-{mm}')"
        OUTPUT_VARIABLE _torch_paths
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    string(REPLACE "\n" ";" _torch_paths "${_torch_paths}")
    list(GET _torch_paths 0 _torch_root)
    list(GET _torch_paths 1 _mts_torch_root)
    list(GET _torch_paths 2 _mta_torch_root)

    add_library(c10 SHARED IMPORTED)
    set_target_properties(c10 PROPERTIES
        IMPORTED_LOCATION "${_torch_root}/lib/libc10.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_torch_root}/include"
    )
    add_library(torch_cpu SHARED IMPORTED)
    set_target_properties(torch_cpu PROPERTIES
        IMPORTED_LOCATION "${_torch_root}/lib/libtorch_cpu.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_torch_root}/include;${_torch_root}/include/torch/csrc/api/include"
        INTERFACE_LINK_LIBRARIES c10
    )
    add_library(torch SHARED IMPORTED)
    set_target_properties(torch PROPERTIES
        IMPORTED_LOCATION "${_torch_root}/lib/libtorch.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_torch_root}/include;${_torch_root}/include/torch/csrc/api/include"
        INTERFACE_LINK_LIBRARIES "torch_cpu;c10"
    )
    add_library(metatensor_torch SHARED IMPORTED)
    set_target_properties(metatensor_torch PROPERTIES
        IMPORTED_LOCATION "${_mts_torch_root}/lib/libmetatensor_torch.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_mts_torch_root}/include"
        INTERFACE_LINK_LIBRARIES "torch;metatensor"
    )
    add_library(metatomic_torch SHARED IMPORTED)
    set_target_properties(metatomic_torch PROPERTIES
        IMPORTED_LOCATION "${_mta_torch_root}/lib/libmetatomic_torch.so"
        INTERFACE_INCLUDE_DIRECTORIES "${_mta_torch_root}/include"
        INTERFACE_LINK_LIBRARIES "metatensor_torch;torch"
    )
    set(TORCH_INSTALL_PREFIX "${_torch_root}" PARENT_SCOPE)
    message(STATUS "Imported TorchScript backend from ${_mta_torch_root}")
endfunction()

if(OPENMM_METATOMIC_TORCH)
    openmm_metatomic_import_torch()
endif()

option(OPENMM_METATOMIC_VESIN "Use vesin for neighbor lists (both backends) instead of the O(N^2) fallback" ON)

function(openmm_metatomic_import_vesin)
    if(NOT Python3_EXECUTABLE)
        return()
    endif()
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "from pathlib import Path
import vesin
root = Path(vesin.__file__).resolve().parent
print(root / 'include')
print(root / 'lib' / 'libvesin.so')"
        OUTPUT_VARIABLE _vesin_paths
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE _vesin_result
    )
    if(NOT _vesin_result EQUAL 0 OR NOT _vesin_paths)
        message(STATUS "vesin not found (pip install vesin); using the O(N^2) neighbor list fallback")
        return()
    endif()
    string(REPLACE "\n" ";" _vesin_paths "${_vesin_paths}")
    list(GET _vesin_paths 0 _vesin_include)
    list(GET _vesin_paths 1 _vesin_lib)
    if(NOT EXISTS "${_vesin_lib}")
        message(STATUS "vesin Python package found but ${_vesin_lib} is missing; using the O(N^2) fallback")
        return()
    endif()
    add_library(vesin SHARED IMPORTED GLOBAL)
    set_target_properties(vesin PROPERTIES
        IMPORTED_LOCATION "${_vesin_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_vesin_include}"
    )
    set(OPENMM_METATOMIC_VESIN_FOUND TRUE PARENT_SCOPE)
    get_filename_component(_vesin_libdir "${_vesin_lib}" DIRECTORY)
    set(OPENMM_METATOMIC_VESIN_LIBDIR "${_vesin_libdir}" PARENT_SCOPE)
    message(STATUS "Imported vesin from ${_vesin_lib}")
endfunction()

if(OPENMM_METATOMIC_VESIN)
    openmm_metatomic_import_vesin()
endif()

function(openmm_metatomic_link_vesin target)
    if(TARGET vesin)
        target_compile_definitions(${target} PRIVATE OPENMM_METATOMIC_USE_VESIN)
        target_link_libraries(${target} PRIVATE vesin)
        set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "${OPENMM_METATOMIC_VESIN_LIBDIR}")
    endif()
endfunction()

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
openmm_metatomic_link_vesin(openmm-metatomic-spike)

get_target_property(_mta_loc metatomic::shared IMPORTED_LOCATION)
if(_mta_loc)
    get_filename_component(_mta_libdir "${_mta_loc}" DIRECTORY)
    set_property(TARGET openmm-metatomic-spike APPEND PROPERTY BUILD_RPATH "${_mta_libdir}")
endif()
if(OPENMM_METATOMIC_TORCH AND DEFINED TORCH_INSTALL_PREFIX)
    set_property(TARGET openmm-metatomic-spike APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
    get_target_property(_mta_torch_loc metatomic_torch IMPORTED_LOCATION)
    if(_mta_torch_loc)
        get_filename_component(_mta_torch_libdir "${_mta_torch_loc}" DIRECTORY)
        set_property(TARGET openmm-metatomic-spike APPEND PROPERTY BUILD_RPATH "${_mta_torch_libdir}")
    endif()
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

if(Python3_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "from pathlib import Path
import sphericart
root = Path(sphericart.__file__).resolve().parent
print(root / 'include')
print(root / 'lib' / 'libsphericart.so')"
        OUTPUT_VARIABLE _sphericart_paths
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if(_sphericart_paths)
    string(REPLACE "\n" ";" _sphericart_paths "${_sphericart_paths}")
    list(GET _sphericart_paths 0 SPHERICART_INCLUDE_DIR)
    list(GET _sphericart_paths 1 SPHERICART_LIBRARY)
    add_library(sphericart SHARED IMPORTED)
    set_target_properties(sphericart PROPERTIES
        IMPORTED_LOCATION "${SPHERICART_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SPHERICART_INCLUDE_DIR}"
    )
    get_filename_component(SPHERICART_LIBDIR "${SPHERICART_LIBRARY}" DIRECTORY)
    message(STATUS "Imported sphericart from ${SPHERICART_LIBRARY}")
endif()

add_executable(openmm-metatomic-bpnn spike/evaluate_bpnn.cpp)
target_include_directories(openmm-metatomic-bpnn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
)
target_link_libraries(openmm-metatomic-bpnn PRIVATE metatomic::shared)
if(TARGET sphericart)
    target_link_libraries(openmm-metatomic-bpnn PRIVATE sphericart)
    set_property(TARGET openmm-metatomic-bpnn APPEND PROPERTY BUILD_RPATH "${SPHERICART_LIBDIR}")
else()
    message(FATAL_ERROR "sphericart C++ library not found (pip install sphericart)")
endif()
if(_mta_loc)
    set_property(TARGET openmm-metatomic-bpnn APPEND PROPERTY BUILD_RPATH "${_mta_libdir}")
endif()
if(OPENMM_METATOMIC_TORCH)
    target_compile_definitions(openmm-metatomic-bpnn PRIVATE OPENMM_METATOMIC_TORCH)
    target_link_libraries(openmm-metatomic-bpnn PRIVATE metatomic_torch)
    if(DEFINED TORCH_INSTALL_PREFIX)
        set_property(TARGET openmm-metatomic-bpnn APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
        if(_mta_torch_libdir)
            set_property(TARGET openmm-metatomic-bpnn APPEND PROPERTY BUILD_RPATH "${_mta_torch_libdir}")
        endif()
    endif()
endif()

set(BPNN_PT "${CMAKE_CURRENT_BINARY_DIR}/soap-bpnn.pt")
if(OPENMM_METATOMIC_TORCH AND Python3_EXECUTABLE)
    add_custom_command(
        OUTPUT ${BPNN_PT}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/spike/export_bpnn.py ${BPNN_PT}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/spike/export_bpnn.py
            ${CMAKE_CURRENT_SOURCE_DIR}/examples/_bpnn.py
        COMMENT "Export TorchScript SOAP-BPNN model"
    )
    add_custom_target(bpnn-pt ALL DEPENDS ${BPNN_PT})
endif()

enable_testing()
add_test(NAME spike-core COMMAND openmm-metatomic-spike)
add_test(NAME spike-bpnn COMMAND openmm-metatomic-bpnn)
if(OPENMM_METATOMIC_TORCH AND Python3_EXECUTABLE)
    add_test(NAME spike-torch COMMAND openmm-metatomic-spike ${HARMONIC_PT})
    add_test(NAME spike-bpnn-torch COMMAND openmm-metatomic-bpnn ${BPNN_PT})
    add_test(
        NAME bpnn-python
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/examples/_bpnn.py
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples
    )
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
    set_target_properties(OpenMMMetatomic PROPERTIES
        BUILD_RPATH "${_mta_libdir};${OPENMM_DIR}/lib"
    )
    if(OPENMM_METATOMIC_TORCH)
        target_compile_definitions(OpenMMMetatomic PRIVATE OPENMM_METATOMIC_TORCH)
        target_link_libraries(OpenMMMetatomic PRIVATE metatomic_torch)
        set_property(TARGET OpenMMMetatomic APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
        if(_mta_torch_libdir)
            set_property(TARGET OpenMMMetatomic APPEND PROPERTY BUILD_RPATH "${_mta_torch_libdir}")
        endif()
    endif()
    openmm_metatomic_link_vesin(OpenMMMetatomic)
    install(TARGETS OpenMMMetatomic DESTINATION lib/plugins)

    add_executable(openmm-metatomic-bench-scaling spike/bench_scaling.cpp)
    target_include_directories(openmm-metatomic-bench-scaling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
        ${OPENMM_DIR}/include
    )
    target_link_libraries(openmm-metatomic-bench-scaling PRIVATE
        OpenMMMetatomic metatomic::shared "${OPENMM_DIR}/lib/libOpenMM.so"
    )
    set_property(TARGET openmm-metatomic-bench-scaling APPEND PROPERTY
        BUILD_RPATH "${_mta_libdir};${OPENMM_DIR}/lib"
    )
    if(OPENMM_METATOMIC_TORCH)
        target_compile_definitions(openmm-metatomic-bench-scaling PRIVATE OPENMM_METATOMIC_TORCH)
        target_link_libraries(openmm-metatomic-bench-scaling PRIVATE metatomic_torch)
        set_property(TARGET openmm-metatomic-bench-scaling APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
        if(_mta_torch_libdir)
            set_property(TARGET openmm-metatomic-bench-scaling APPEND PROPERTY BUILD_RPATH "${_mta_torch_libdir}")
        endif()
    endif()
    add_test(
        NAME bench-scaling-smoke
        COMMAND openmm-metatomic-bench-scaling --sizes 3,30 --repeats 3 --warmup 1
    )

    add_executable(openmm-metatomic-test-pairlist spike/test_pairlist.cpp)
    target_include_directories(openmm-metatomic-test-pairlist PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
        ${OPENMM_DIR}/include
    )
    target_link_libraries(openmm-metatomic-test-pairlist PRIVATE
        OpenMMMetatomic metatomic::shared "${OPENMM_DIR}/lib/libOpenMM.so"
    )
    set_property(TARGET openmm-metatomic-test-pairlist APPEND PROPERTY
        BUILD_RPATH "${_mta_libdir};${OPENMM_DIR}/lib"
    )
    if(OPENMM_METATOMIC_TORCH)
        target_compile_definitions(openmm-metatomic-test-pairlist PRIVATE OPENMM_METATOMIC_TORCH)
        target_link_libraries(openmm-metatomic-test-pairlist PRIVATE metatomic_torch)
        set_property(TARGET openmm-metatomic-test-pairlist APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
        if(_mta_torch_libdir)
            set_property(TARGET openmm-metatomic-test-pairlist APPEND PROPERTY BUILD_RPATH "${_mta_torch_libdir}")
        endif()

        set(PAIRLIST_PT "${CMAKE_CURRENT_BINARY_DIR}/pairlist.pt")
        if(Python3_EXECUTABLE)
            add_custom_command(
                OUTPUT ${PAIRLIST_PT}
                COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/spike/export_pairlist_torch.py ${PAIRLIST_PT} 30
                DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/spike/export_pairlist_torch.py
                COMMENT "Export TorchScript neighbor-list harmonic model"
            )
            add_custom_target(pairlist-pt ALL DEPENDS ${PAIRLIST_PT})
            add_test(NAME test-pairlist COMMAND openmm-metatomic-test-pairlist ${PAIRLIST_PT})
        else()
            add_test(NAME test-pairlist COMMAND openmm-metatomic-test-pairlist)
        endif()
    else()
        add_test(NAME test-pairlist COMMAND openmm-metatomic-test-pairlist)
    endif()

    function(openmm_metatomic_plugin_exe name source)
        add_executable(${name} ${source})
        target_include_directories(${name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/openmmapi/include
            ${OPENMM_DIR}/include
        )
        target_link_libraries(${name} PRIVATE
            OpenMMMetatomic metatomic::shared "${OPENMM_DIR}/lib/libOpenMM.so"
        )
        set_property(TARGET ${name} APPEND PROPERTY
            BUILD_RPATH "${_mta_libdir};${OPENMM_DIR}/lib"
        )
        if(OPENMM_METATOMIC_TORCH)
            target_compile_definitions(${name} PRIVATE OPENMM_METATOMIC_TORCH)
            target_link_libraries(${name} PRIVATE metatomic_torch)
            set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
            if(_mta_torch_libdir)
                set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "${_mta_torch_libdir}")
            endif()
        endif()
        openmm_metatomic_link_vesin(${name})
    endfunction()

    openmm_metatomic_plugin_exe(openmm-metatomic-run-md spike/run_md.cpp)
    openmm_metatomic_plugin_exe(openmm-metatomic-bench-settings spike/bench_settings.cpp)

    add_test(
        NAME run-md-smoke
        COMMAND openmm-metatomic-run-md --model harmonic --backend core --nve 5 --nvt 5 --eval 2 --warmup-eval 1 --warmup-step 1 --platform Reference
    )
    add_test(
        NAME bench-settings-smoke
        COMMAND openmm-metatomic-bench-settings
            --models harmonic --backends core --platforms Reference
            --nl none --nve 2 --nvt 2 --eval 2 --warmup-eval 1 --warmup-step 1
            --harmonic-atoms 3
    )

    add_subdirectory(python)
endif()
