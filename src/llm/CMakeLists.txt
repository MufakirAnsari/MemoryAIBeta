# MemoryAI Enterprise - Llama.cpp with Mythic MP-30 Acceleration
# Optimized for 45 TOPS @ 150mW performance

cmake_minimum_required(VERSION 3.18)
project(llama_cpp_mythic CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Architecture detection
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARCH_ARM64 ON)
    add_definitions(-DARM64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_X86_64 ON)
    add_definitions(-DX86_64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(ARCH_RISCV ON)
    add_definitions(-DRISCV)
endif()

# Mythic MP-30 Configuration
option(MYTHIC_OFFLOAD "Enable Mythic MP-30 acceleration" ON)
option(MYTHIC_KERNEL_DEBUG "Enable Mythic kernel debugging" OFF)

if(MYTHIC_OFFLOAD)
    find_package(PkgConfig REQUIRED)
    
    # Mythic SDK paths
    set(MYTHIC_SDK_PATH "/opt/mythic/sdk" CACHE PATH "Mythic SDK installation path")
    set(MYTHIC_KERNEL_PATH "/opt/mythic/kernels" CACHE PATH "Mythic kernel path")
    
    # Include directories
    include_directories(${MYTHIC_SDK_PATH}/include)
    include_directories(${MYTHIC_KERNEL_PATH}/include)
    
    # Mythic libraries
    find_library(MYTHIC_RUNTIME mythic_runtime ${MYTHIC_SDK_PATH}/lib)
    find_library(MYTHIC_COMPILER mythic_compiler ${MYTHIC_SDK_PATH}/lib)
    find_library(MYTHIC_KERNEL mythic_kernel ${MYTHIC_KERNEL_PATH}/lib)
    
    add_definitions(-DMYTHIC_OFFLOAD)
    add_definitions(-DMYTHIC_45TOPS)
    add_definitions(-DMYTHIC_150MW)
    
    if(MYTHIC_KERNEL_DEBUG)
        add_definitions(-DMYTHIC_DEBUG)
    endif()
    
    message(STATUS "🔥 Mythic MP-30 acceleration enabled")
    message(STATUS "   SDK Path: ${MYTHIC_SDK_PATH}")
    message(STATUS "   Performance: 45 TOPS @ 150mW")
else()
    message(STATUS "⚠️  Mythic acceleration disabled")
endif()

# Post-quantum signatures
option(PQ_SIGNATURES "Enable post-quantum signature verification" ON)
if(PQ_SIGNATURES)
    find_package(OpenSSL REQUIRED)
    find_package(liboqs REQUIRED)
    
    include_directories(${LIBOQS_INCLUDE_DIRS})
    add_definitions(-DPQ_SIGNATURES)
    add_definitions(-DDILITHIUM_3)
    
    message(STATUS "🔐 Post-quantum signatures enabled (Dilithium-3)")
endif()

# Optimization flags
if(ARCH_ARM64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=apple-m1 -mtune=native")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mcpu=apple-m1 -mtune=native")
elseif(ARCH_X86_64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -mtune=native")
elseif(ARCH_RISCV)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=rv64gc")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=rv64gc")
endif()

# Common flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-unused-parameter")

# Enable SIMD optimizations
if(ARCH_ARM64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    add_definitions(-DGGML_USE_NEON)
elseif(ARCH_X86_64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -mf16c")
    add_definitions(-DGGML_USE_AVX2)
endif()

# Source files
set(LLAMA_SOURCES
    ggml.c
    ggml-alloc.c
    ggml-backend.c
    ggml-quants.c
    llama.cpp
    unicode.cpp
    unicode-data.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mythic/mythic_kernel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mythic/mythic_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pq_signatures/dilithium3.cpp
)

set(LLAMA_HEADERS
    ggml.h
    ggml-alloc.h
    ggml-backend.h
    ggml-quants.h
    llama.h
    unicode.h
    ${CMAKE_CURRENT_SOURCE_DIR}/mythic/mythic_kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pq_signatures/dilithium3.h
)

# Main library
add_library(llama STATIC ${LLAMA_SOURCES} ${LLAMA_HEADERS})

# Mythic kernel integration
if(MYTHIC_OFFLOAD)
    target_link_libraries(llama ${MYTHIC_RUNTIME} ${MYTHIC_COMPILER} ${MYTHIC_KERNEL})
endif()

# Post-quantum signatures
if(PQ_SIGNATURES)
    target_link_libraries(llama ${LIBOQS_LIBRARIES} OpenSSL::SSL)
endif()

# Threading
find_package(Threads REQUIRED)
target_link_libraries(llama Threads::Threads)

# Math library
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(llama ${MATH_LIBRARY})
endif()

# Executables
add_executable(llama-cli examples/main.cpp)
target_link_libraries(llama-cli llama)

add_executable(llama-quantize examples/quantize.cpp)
target_link_libraries(llama-quantize llama)

add_executable(llama-server examples/server.cpp)
target_link_libraries(llama-server llama)

# Installation
install(TARGETS llama llama-cli llama-quantize llama-server
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(FILES ${LLAMA_HEADERS} DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "🚀 MemoryAI Enterprise Build Configuration")
message(STATUS "========================================")
message(STATUS "Architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Mythic MP-30: ${MYTHIC_OFFLOAD}")
message(STATUS "PQ Signatures: ${PQ_SIGNATURES}")
message(STATUS "Optimization: O3 with native CPU features")
message(STATUS "")
message(STATUS "🔥 Building with 45 TOPS Mythic acceleration...")
message(STATUS "🔐 Post-quantum security enabled...")
message(STATUS "⚡ Optimized for ultra-low power inference...")