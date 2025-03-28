cmake_minimum_required(VERSION 3.16)
project(lgmath)

option(USE_AMENT "Use ament_cmake to build lgmath for ROS2." OFF)

# Compiler setup
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# add_compile_options(-O3 -Wall -pedantic)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Og")

## cmake flow (default)
if(NOT USE_AMENT)

  set(PROJECT_VERSION 1.1.0)

  # Find dependencies
  find_package(Eigen3 3.3.7 REQUIRED)

  # Build library
  file(GLOB_RECURSE SOURCE_FILES "src/*.cpp")
  add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
  target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )

  if(USE_AUTODIFF)
    message(STATUS "Building with automatic differentiation support.")
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AUTODIFF)
    if(USE_AUTODIFF_BACKWARD)
      target_compile_definitions(${PROJECT_NAME} PRIVATE USE_AUTODIFF_BACKWARD)
      message(FATAL_ERROR "Backward mode is not yet supported.")
    else()
      message(STATUS "Using forward mode autodiff.")
    endif()
    find_package(autodiff REQUIRED)
    target_link_libraries(${PROJECT_NAME} INTERFACE autodiff::autodiff)
  endif()

  # Install
  install(
    DIRECTORY include/
    DESTINATION include
  )

  install(
    TARGETS ${PROJECT_NAME}
    DESTINATION lib
    EXPORT ${PROJECT_NAME}Targets
  )

  # Export
  set(PROJECT_LIBRARY ${PROJECT_NAME})

  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
  )
  # export to build directory so no need to install in order to use find_package
  export(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake
  )

  # install export cmake files
  install(
    EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
  )

  install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake/${PROJECT_NAME}
  )

  if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/e88cb95b92acbdce9b058dd894a68e1281b38495.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    # Unit-tests
    add_executable(so3_tests tests/SO3Tests.cpp)
    target_link_libraries(so3_tests ${PROJECT_NAME} GTest::gtest_main)
    add_executable(se3_tests tests/SE3Tests.cpp)
    target_link_libraries(se3_tests ${PROJECT_NAME} GTest::gtest_main)
    add_executable(covariance_tests tests/CovarianceTests.cpp)
    target_link_libraries(covariance_tests ${PROJECT_NAME} GTest::gtest_main)
    add_executable(rotation_tests tests/RotationTests.cpp)
    target_link_libraries(rotation_tests ${PROJECT_NAME} GTest::gtest_main)
    add_executable(transform_tests tests/TransformTests.cpp)
    target_link_libraries(transform_tests ${PROJECT_NAME} GTest::gtest_main)
    add_executable(transform_with_covariance_tests tests/TransformWithCovarianceTests.cpp)
    target_link_libraries(transform_with_covariance_tests ${PROJECT_NAME} GTest::gtest_main)

    gtest_discover_tests(so3_tests)
    gtest_discover_tests(se3_tests)
    gtest_discover_tests(covariance_tests)
    gtest_discover_tests(rotation_tests)
    gtest_discover_tests(transform_tests)
    gtest_discover_tests(transform_with_covariance_tests)

    if(USE_AUTODIFF)
      if(USE_AUTODIFF_BACKWARD)
        add_executable(so3_tests_autodiff_backward tests/SO3AutodiffTestsBackward.cpp)
        target_link_libraries(so3_tests_autodiff_backward ${PROJECT_NAME} GTest::gtest_main)
        add_executable(se3_tests_autodiff_backward tests/SE3AutodiffTestsBackward.cpp)
        target_link_libraries(se3_tests_autodiff_backward ${PROJECT_NAME} GTest::gtest_main)
        gtest_discover_tests(so3_tests_autodiff_backward)
        gtest_discover_tests(se3_tests_autodiff_backward)
      else()
        add_executable(so3_tests_autodiff_forward tests/SO3AutodiffTestsForward.cpp)
        target_link_libraries(so3_tests_autodiff_forward ${PROJECT_NAME} GTest::gtest_main)
        add_executable(se3_tests_autodiff_forward tests/SE3AutodiffTestsForward.cpp)
        target_link_libraries(se3_tests_autodiff_forward ${PROJECT_NAME} GTest::gtest_main)
        gtest_discover_tests(so3_tests_autodiff_forward)
        gtest_discover_tests(se3_tests_autodiff_forward)
      endif()
    endif()
  endif()

  ## ROS2 ament_cmake flow
else()

  find_package(ament_cmake REQUIRED)
  find_package(eigen3_cmake_module REQUIRED)
  find_package(Eigen3 3.3.7 REQUIRED)

  file(GLOB_RECURSE SOURCE src/*.cpp)
  add_library(${PROJECT_NAME} ${SOURCE})
  ament_target_dependencies(${PROJECT_NAME} Eigen3)
  target_include_directories(${PROJECT_NAME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  install(
    DIRECTORY include/
    DESTINATION include
  )

  install(
    TARGETS ${PROJECT_NAME}
    EXPORT export_${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION lib/${PROJECT_NAME}
    INCLUDES DESTINATION include
  )

  ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

  if(BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)

    # Unit-tests
    ament_add_gtest(so3_tests tests/SO3Tests.cpp)
    target_link_libraries(so3_tests ${PROJECT_NAME})
    ament_add_gtest(se3_tests tests/SE3Tests.cpp)
    target_link_libraries(se3_tests ${PROJECT_NAME})
    ament_add_gtest(covariance_tests tests/CovarianceTests.cpp)
    target_link_libraries(covariance_tests ${PROJECT_NAME})
    ament_add_gtest(rotation_tests tests/RotationTests.cpp)
    target_link_libraries(rotation_tests ${PROJECT_NAME})
    ament_add_gtest(transform_tests tests/TransformTests.cpp)
    target_link_libraries(transform_tests ${PROJECT_NAME})
    ament_add_gtest(transform_with_covariance_tests tests/TransformWithCovarianceTests.cpp)
    target_link_libraries(transform_with_covariance_tests ${PROJECT_NAME})

    # Benchmarks
    ament_add_gtest(so3_benchmarks benchmarks/SO3SpeedTest.cpp)
    target_link_libraries(so3_benchmarks ${PROJECT_NAME})
    ament_add_gtest(se3_benchmarks benchmarks/SE3SpeedTest.cpp)
    target_link_libraries(se3_benchmarks ${PROJECT_NAME})
    ament_add_gtest(rotation_benchmarks benchmarks/RotationSpeedTest.cpp)
    target_link_libraries(rotation_benchmarks ${PROJECT_NAME})
    ament_add_gtest(transform_benchmarks benchmarks/TransformSpeedTest.cpp)
    target_link_libraries(transform_benchmarks ${PROJECT_NAME})
    ament_add_gtest(transform_with_covariance_benchmarks benchmarks/TransformWithCovarianceSpeedTest.cpp)
    target_link_libraries(transform_with_covariance_benchmarks ${PROJECT_NAME})

    # Linting
    find_package(ament_lint_auto REQUIRED)
    ament_lint_auto_find_test_dependencies() # Lint based on linter test_depend in package.xml
  endif()

endif()

# Documentation
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
else(DOXYGEN_FOUND)
  message("Doxygen not found.")
  add_custom_target(doc
    COMMENT "Doxygen was not found on this system so this target does not exist, please install it and re-run CMake." VERBATIM)
endif(DOXYGEN_FOUND)

if(USE_AMENT)
  ament_package()
endif()
