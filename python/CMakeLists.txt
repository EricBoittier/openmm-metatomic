find_package(SWIG QUIET)
if(NOT SWIG_EXECUTABLE)
    find_program(SWIG_EXECUTABLE swig)
endif()
if(NOT SWIG_EXECUTABLE)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import shutil,sys; p=shutil.which('swig'); print(p or '')"
        OUTPUT_VARIABLE SWIG_EXECUTABLE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
if(NOT SWIG_EXECUTABLE)
    message(STATUS "SWIG not found; skipping Python wrappers")
    return()
endif()
message(STATUS "Python wrappers via ${SWIG_EXECUTABLE}")

set(WRAP_FILE "${CMAKE_CURRENT_BINARY_DIR}/MetatomicForceWrapper.cpp")
set(SWIG_PY "${CMAKE_CURRENT_BINARY_DIR}/openmmmetatomic/openmmmetatomic.py")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/openmmmetatomic")

add_custom_command(
    OUTPUT "${WRAP_FILE}" "${SWIG_PY}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -interface _openmmmetatomic
        -o "${WRAP_FILE}"
        -outdir "${CMAKE_CURRENT_BINARY_DIR}/openmmmetatomic"
        "-I${OPENMM_DIR}/include"
        "-I${CMAKE_SOURCE_DIR}/openmmapi/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/swig/MetatomicForce.i"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmmetatomic/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmmetatomic/openmmml.py"
        "${CMAKE_CURRENT_BINARY_DIR}/openmmmetatomic/"
    DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/swig/MetatomicForce.i"
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmmetatomic/__init__.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/openmmmetatomic/openmmml.py"
        OpenMMMetatomic
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "SWIG wrap MetatomicForce"
)

set(PLUGIN_HEADER_DIR "${CMAKE_SOURCE_DIR}/openmmapi/include")
set(PLUGIN_LIBRARY_DIR "${CMAKE_BINARY_DIR}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/setup.py"
    "${CMAKE_CURRENT_BINARY_DIR}/setup.py"
    @ONLY
)

add_custom_target(python-wrapper ALL
    DEPENDS "${WRAP_FILE}" OpenMMMetatomic
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/setup.py" build_ext --inplace
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Build openmmmetatomic Python extension"
)

add_custom_target(PythonInstall
    DEPENDS python-wrapper
    COMMAND ${Python3_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

add_test(
    NAME python-metatomic-force
    COMMAND ${Python3_EXECUTABLE} -c
        "from openmmmetatomic import MetatomicForce; f=MetatomicForce('harmonic'); f.setBackend('core'); f.setAtomicTypes([1,1,8]); assert f.getModelPath()=='harmonic'"
)
set_tests_properties(python-metatomic-force PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};OPENMM_PLUGIN_DIR=${CMAKE_BINARY_DIR}"
)

add_test(
    NAME python-serialization
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
        "${CMAKE_SOURCE_DIR}/tests/python/test_serialization.py"
)
set_tests_properties(python-serialization PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};OPENMM_PLUGIN_DIR=${CMAKE_BINARY_DIR}"
)

add_test(
    NAME python-openmmml-native
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
        "${CMAKE_SOURCE_DIR}/tests/python/test_openmmml_native.py"
)
set_tests_properties(python-openmmml-native PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};OPENMM_PLUGIN_DIR=${CMAKE_BINARY_DIR}"
)

add_test(
    NAME python-complex-mixed
    COMMAND ${Python3_EXECUTABLE} -m pytest -q
        "${CMAKE_SOURCE_DIR}/tests/python/test_complex_mixed.py"
)
set_tests_properties(python-complex-mixed PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR};OPENMM_PLUGIN_DIR=${CMAKE_BINARY_DIR}"
)
