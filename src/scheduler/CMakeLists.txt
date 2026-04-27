add_library(scheduler
        scheduler.cpp
)

target_include_directories(scheduler
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC
        "${Z3_SOURCE_DIR}/src/api/c++"
)

target_link_libraries(scheduler
        PUBLIC
        httplib::httplib
        nlohmann_json::nlohmann_json
        device_struct
        libz3
        spdlog::spdlog
        PRIVATE
        docker_client
        time_tools
        Boost::uuid
        Boost::assert Boost::config Boost::throw_exception Boost::type_traits Boost::static_assert
)

target_compile_definitions(scheduler PUBLIC ABSOLUTE_CONFIG_PATH="${ABSOLUTE_CONFIG_PATH}")
