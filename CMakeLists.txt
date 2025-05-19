cmake_minimum_required(VERSION 3.31)
project(parallel_attention C)

set(CMAKE_C_STANDARD 11)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_C_FLAGS "-march=native -g ${CMAKE_CXX_FLAGS}")

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

add_executable(attention attention.c)
add_executable(attention_thread attention_thread.c)
add_executable(attention_mpi attention_mpi.c)

target_link_libraries(attention PRIVATE m)

target_link_libraries(attention_thread PRIVATE Threads::Threads m pthread)

target_include_directories(attention_mpi PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(attention_mpi PRIVATE ${MPI_C_LIBRARIES} m)
