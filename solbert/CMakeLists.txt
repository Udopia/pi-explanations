cmake_minimum_required(VERSION 3.18)

project(solbert VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

set(SOLBERT_CADICAL_TAG "rel-3.0.1" CACHE STRING "CaDiCaL git tag or commit")
set(SOLBERT_CADICAL_LIBRARY "" CACHE FILEPATH "Use an existing libcadical archive instead of building CaDiCaL")

if(SOLBERT_CADICAL_LIBRARY)
    if(NOT EXISTS "${SOLBERT_CADICAL_LIBRARY}")
        message(FATAL_ERROR "SOLBERT_CADICAL_LIBRARY does not exist: ${SOLBERT_CADICAL_LIBRARY}")
    endif()
    set(CADICAL_LIBRARY "${SOLBERT_CADICAL_LIBRARY}")
else()
    include(ExternalProject)
    find_program(MAKE_EXECUTABLE NAMES gmake make REQUIRED)

    set(CADICAL_ARCH_SUFFIX "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
    set(CADICAL_CXXFLAGS "-fPIC")
    if(APPLE)
        if(CMAKE_OSX_DEPLOYMENT_TARGET)
            string(APPEND CADICAL_CXXFLAGS " -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()
        if(CMAKE_OSX_ARCHITECTURES)
            string(REPLACE ";" "_" CADICAL_ARCH_SUFFIX "macos-${CMAKE_OSX_ARCHITECTURES}")
            foreach(CADICAL_ARCH IN LISTS CMAKE_OSX_ARCHITECTURES)
                string(APPEND CADICAL_CXXFLAGS " -arch ${CADICAL_ARCH}")
            endforeach()
        endif()
    endif()

    set(CADICAL_PREFIX "${CMAKE_BINARY_DIR}/cadical-${CADICAL_ARCH_SUFFIX}")
    set(CADICAL_SOURCE_DIR "${CADICAL_PREFIX}/src/cadical")
    set(CADICAL_LIBRARY "${CADICAL_SOURCE_DIR}/build/libcadical.a")

    ExternalProject_Add(cadical_external
        GIT_REPOSITORY https://github.com/arminbiere/cadical.git
        GIT_TAG "${SOLBERT_CADICAL_TAG}"
        GIT_SHALLOW TRUE
        UPDATE_DISCONNECTED TRUE
        PREFIX "${CADICAL_PREFIX}"
        CONFIGURE_COMMAND
            ${CMAKE_COMMAND} -E env "CXXFLAGS=${CADICAL_CXXFLAGS}" ./configure
        BUILD_COMMAND "${MAKE_EXECUTABLE}" -j1
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS "${CADICAL_LIBRARY}"
        BUILD_IN_SOURCE TRUE
    )
endif()

add_library(cadical STATIC IMPORTED GLOBAL)
set_target_properties(cadical PROPERTIES
    IMPORTED_LOCATION "${CADICAL_LIBRARY}"
    IMPORTED_LINK_INTERFACE_LANGUAGES CXX
)
if(TARGET cadical_external)
    add_dependencies(cadical cadical_external)
endif()

Python3_add_library(_native MODULE WITH_SOABI src/solbert.cc)
target_include_directories(_native PRIVATE "${PROJECT_SOURCE_DIR}")
target_link_libraries(_native PRIVATE cadical)
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/solbert"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/solbert"
)
add_custom_command(TARGET _native POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/solbert"
            "${CMAKE_BINARY_DIR}/solbert"
)

install(TARGETS _native
    LIBRARY DESTINATION solbert
    RUNTIME DESTINATION solbert
)

if(BUILD_TESTING)
    add_test(
        NAME solbert_api
        COMMAND ${CMAKE_COMMAND} -E env
            "PYTHONPATH=${CMAKE_BINARY_DIR}"
                "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/test/test_solbert.py"
    )
endif()
