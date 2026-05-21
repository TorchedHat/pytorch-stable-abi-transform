cmake_minimum_required(VERSION 3.20)
project(stable-abi-transform VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLVM/Clang defines needed for client code
add_compile_definitions(
    _GNU_SOURCE
    __STDC_CONSTANT_MACROS
    __STDC_FORMAT_MACROS
    __STDC_LIMIT_MACROS
)

# Detect LLVM paths via llvm-config
find_program(LLVM_CONFIG_EXE NAMES llvm-config llvm-config-20 llvm-config-19)
if(NOT LLVM_CONFIG_EXE)
    message(FATAL_ERROR "llvm-config not found. Install llvm-devel / llvm-dev.")
endif()

execute_process(
    COMMAND ${LLVM_CONFIG_EXE} --prefix
    OUTPUT_VARIABLE LLVM_INSTALL_PREFIX_VAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "llvm-config --prefix failed")
endif()

execute_process(
    COMMAND ${LLVM_CONFIG_EXE} --version
    OUTPUT_VARIABLE LLVM_VERSION_VAL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "llvm-config --version failed")
endif()

execute_process(
    COMMAND ${LLVM_CONFIG_EXE} --includedir
    OUTPUT_VARIABLE LLVM_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "llvm-config --includedir failed")
endif()

execute_process(
    COMMAND ${LLVM_CONFIG_EXE} --libdir
    OUTPUT_VARIABLE LLVM_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "llvm-config --libdir failed")
endif()

string(REGEX REPLACE "([0-9]+)\\..*" "\\1" CLANG_MAJOR_VERSION "${LLVM_VERSION_VAL}")

add_executable(stable-abi-transform
    src/main.cpp
    src/Config.cpp
    src/StableAbiAction.cpp
    src/PreprocessorCallbacks.cpp
    src/TransformerRules.cpp
    src/Reporter.cpp
    src/Verifier.cpp
)

target_include_directories(stable-abi-transform PRIVATE ${LLVM_INCLUDE_DIR})
target_link_directories(stable-abi-transform PRIVATE ${LLVM_LIB_DIR})

target_link_libraries(stable-abi-transform PRIVATE clang-cpp LLVM)

target_compile_definitions(stable-abi-transform PRIVATE
    LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX_VAL}"
    CLANG_VERSION_MAJOR_STR="${CLANG_MAJOR_VERSION}"
    TOOL_VERSION="${PROJECT_VERSION}"
)
