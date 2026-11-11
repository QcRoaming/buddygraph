cmake_minimum_required(VERSION 3.20)
project(BuddyGraphPythonBindings LANGUAGES C CXX)

# Build only the upstream core bindings and dialect registration against a
# precompiled MLIR SDK. MLIR_SOURCE_DIR must match the SDK version.
find_package(MLIR REQUIRED CONFIG)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind 2 CONFIG REQUIRED)
if(NOT EXISTS "${MLIR_SOURCE_DIR}/lib/Bindings/Python/MainModule.cpp")
  message(FATAL_ERROR "Set MLIR_SOURCE_DIR to the matching MLIR source package")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
set(binding_src "${MLIR_SOURCE_DIR}/lib/Bindings/Python")
set(package_dir "${CMAKE_BINARY_DIR}/python_packages/mlir_core/mlir")
file(MAKE_DIRECTORY "${package_dir}/_mlir_libs" "${package_dir}/dialects")
foreach(file ir.py passmanager.py rewrite.py _mlir_libs/__init__.py
    dialects/_ods_common.py dialects/func.py)
  configure_file("${MLIR_SOURCE_DIR}/python/mlir/${file}"
    "${package_dir}/${file}" COPYONLY)
endforeach()
add_custom_command(OUTPUT "${package_dir}/dialects/_func_ops_gen.py"
  COMMAND $<TARGET_FILE:mlir-tblgen> -gen-python-op-bindings -bind-dialect=func
    -I "${MLIR_SOURCE_DIR}/python" -I "${MLIR_SOURCE_DIR}/include"
    -I "${LLVM_INSTALL_PREFIX}/include"
    "${MLIR_SOURCE_DIR}/python/mlir/dialects/FuncOps.td"
    -o "${package_dir}/dialects/_func_ops_gen.py"
  DEPENDS "${MLIR_SOURCE_DIR}/python/mlir/dialects/FuncOps.td")
add_custom_target(func-python ALL
  DEPENDS "${package_dir}/dialects/_func_ops_gen.py")

file(WRITE "${CMAKE_BINARY_DIR}/empty.cpp" "// C API from the precompiled SDK.\n")
add_library(BuddyGraphPythonCAPI SHARED "${CMAKE_BINARY_DIR}/empty.cpp")
target_link_libraries(BuddyGraphPythonCAPI PRIVATE
  "-Wl,--whole-archive" MLIRCAPIDebug MLIRCAPIIR MLIRCAPIInterfaces
  MLIRCAPIFunc MLIRCAPIRegisterEverything MLIRCAPIConversion MLIRCAPITransforms
  "-Wl,--no-whole-archive" MLIR LLVM)
set_target_properties(BuddyGraphPythonCAPI PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${package_dir}/_mlir_libs")

set(core_sources MainModule.cpp IRAffine.cpp IRAttributes.cpp IRCore.cpp
  IRInterfaces.cpp IRModule.cpp IRTypes.cpp Pass.cpp Rewrite.cpp)
list(TRANSFORM core_sources PREPEND "${binding_src}/")
nanobind_add_module(_mlir NB_DOMAIN mlir ${core_sources})
nanobind_add_module(_mlirRegisterEverything NB_DOMAIN mlir
  "${binding_src}/RegisterEverything.cpp")
foreach(target _mlir _mlirRegisterEverything)
  target_compile_definitions(${target} PRIVATE MLIR_PYTHON_PACKAGE_PREFIX=mlir.)
  target_link_libraries(${target} PRIVATE BuddyGraphPythonCAPI LLVM)
  set_target_properties(${target} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${package_dir}/_mlir_libs"
    BUILD_RPATH "$ORIGIN")
endforeach()
