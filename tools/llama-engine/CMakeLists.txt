set(TARGET llama-engine-impl)

add_library(${TARGET} STATIC
    llama-engine.cpp
    llama-engine.h
    ../server/server-http.cpp
    ../server/server-http.h
)

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../server)
target_include_directories(${TARGET} PRIVATE ../ui ${CMAKE_CURRENT_BINARY_DIR}/../ui)
target_link_libraries(${TARGET} PUBLIC server-context llama-common cpp-httplib llama-ui ${CMAKE_THREAD_LIBS_INIT})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} LIBRARY)
endif()

set(TARGET llama-engine)

add_executable(${TARGET} main.cpp)
target_link_libraries(${TARGET} PRIVATE llama-engine-impl)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
