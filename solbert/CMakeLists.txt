cmake_minimum_required(VERSION 3.10)

project(solbert VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CADICAL_CONFIGURE "./configure")
set(CADICAL_CONFIGURE_OPTION "-fPIC")
set(CADICAL_BUILD make -j)
if (SKIP_CADICAL) 
    message("Warning: SKIP_CADICAL suppresses rebuild of CaDiCaL")
    set(CADICAL_CONFIGURE "")
    set(CADICAL_CONFIGURE_OPTION "")
    set(CADICAL_BUILD "")
endif()

include(ExternalProject)
ExternalProject_Add(Cadical
    GIT_REPOSITORY https://github.com/arminbiere/cadical
    GIT_TAG "origin/master"
    PREFIX "${CMAKE_BINARY_DIR}/cadical"
    CONFIGURE_COMMAND "${CADICAL_CONFIGURE}" "${CADICAL_CONFIGURE_OPTION}"
    BUILD_COMMAND "${CADICAL_BUILD}"
    INSTALL_COMMAND ""
    #BUILD_BYPRODUCTS "${CMAKE_BINARY_DIR}/cadical/src/Cadical/build/libcadical.a"
    BUILD_IN_SOURCE 1 
)

add_library(solver STATIC IMPORTED)
add_dependencies(solver Cadical)
set_target_properties(solver PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/cadical/src/Cadical/build/libcadical.a")

include_directories(solbert PUBLIC "${PROJECT_SOURCE_DIR}")

add_subdirectory("src")

add_executable(solbert src/Main.cc)
add_dependencies(solbert solver)
target_link_libraries(solbert PUBLIC ${LIBS} solver $<TARGET_OBJECTS:apps> $<TARGET_OBJECTS:util>)

target_include_directories(solbert PUBLIC "${PROJECT_SOURCE_DIR}")
