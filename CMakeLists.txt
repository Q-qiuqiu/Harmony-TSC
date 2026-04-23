cmake_minimum_required(VERSION 3.17)
project(docker_scheduler)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip      
)
## For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG v0.18.0
        GIT_SHALLOW TRUE
)

FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        GIT_SHALLOW TRUE
)

fetchcontent_declare(
        boost-uuid
        GIT_REPOSITORY https://github.com/boostorg/uuid
        GIT_TAG boost-1.86.0
        GIT_SHALLOW TRUE
)
FetchContent_Declare(
        boost-assert
        GIT_REPOSITORY https://github.com/boostorg/assert
        GIT_TAG boost-1.86.0
        GIT_SHALLOW TRUE
)

FetchContent_Declare(
        boost-config
        GIT_REPOSITORY https://github.com/boostorg/config
        GIT_TAG boost-1.86.0
        GIT_SHALLOW TRUE
)
FetchContent_Declare(
        boost-throw_exception
        GIT_REPOSITORY https://github.com/boostorg/throw_exception
        GIT_TAG boost-1.86.0
        GIT_SHALLOW TRUE
)

FetchContent_Declare(
        boost-type_traits
        GIT_REPOSITORY https://github.com/boostorg/type_traits
        GIT_TAG boost-1.86.0
        GIT_SHALLOW TRUE
)

FetchContent_Declare(
        boost-static_assert
        GIT_REPOSITORY https://github.com/boostorg/static_assert
        GIT_TAG boost-1.86.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(boost-uuid boost-assert boost-config boost-throw_exception boost-type_traits boost-static_assert)

FetchContent_Declare(
        z3
        GIT_REPOSITORY https://github.com/Z3Prover/z3.git
        GIT_TAG z3-4.13.3
        GIT_SHALLOW TRUE
)

FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.0
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(googletest httplib json z3 spdlog)

# solve std namespace conflict on Windows
if (WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
endif ()

# config_file
get_filename_component(ABSOLUTE_CONFIG_PATH "${CMAKE_CURRENT_SOURCE_DIR}/config_files" ABSOLUTE)

add_subdirectory(src/time_tools)
add_subdirectory(src/custom_struct)

add_subdirectory(src/docker_client)
add_subdirectory(src/scheduler)
add_subdirectory(src/docker_scheduler_agent)
add_subdirectory(src/gateway)
add_subdirectory(src/z3_test)

target_compile_definitions(gateway PUBLIC ABSOLUTE_CONFIG_PATH="${ABSOLUTE_CONFIG_PATH}")
# 构建gtest begin
enable_testing()
include(GoogleTest)

add_subdirectory(tests/docker_client)
add_subdirectory(tests/z3_mock_test)
# 构建gtest end
