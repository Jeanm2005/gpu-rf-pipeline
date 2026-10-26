# add_golden_test(<name> COMMAND <command-and-args...> EXPECTED <golden-file>)
#
# Registers a CTest case comparing COMMAND's stdout, byte for byte, against EXPECTED
# (compiler/AGENTS.md: "Golden files compare exact text"). Also registers a target
# update_golden_<name> that regenerates EXPECTED from the current output; run it only when the
# owner asks, then review the diff (compiler/AGENTS.md).
#
# Captured at include() time: inside the function below, CMAKE_CURRENT_LIST_DIR would instead
# resolve to the caller's directory (tests/), not this file's.
set(SIGFLOW_GOLDEN_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(add_golden_test name)
  cmake_parse_arguments(GOLDEN "" "EXPECTED" "COMMAND" ${ARGN})
  if(NOT GOLDEN_COMMAND OR NOT GOLDEN_EXPECTED)
    message(FATAL_ERROR "add_golden_test(${name}): COMMAND and EXPECTED are both required")
  endif()
  set(runner "${SIGFLOW_GOLDEN_CMAKE_DIR}/RunGoldenTest.cmake")

  # Encoded as a single "::"-joined string rather than passed as a ";"-separated CMake list:
  # add_custom_target's generated shell command splits an unescaped ";" into separate shell
  # statements, silently truncating the command (verified by hand - it ran "cmake" with no
  # useful args, then "-E" and "hello" as their own commands).
  string(REPLACE ";" "::" encoded_command "${GOLDEN_COMMAND}")

  add_test(
    NAME golden_${name}
    COMMAND ${CMAKE_COMMAND}
      "-DACTUAL_COMMAND=${encoded_command}"
      "-DEXPECTED_FILE=${GOLDEN_EXPECTED}"
      -P "${runner}"
  )
  add_custom_target(update_golden_${name}
    COMMAND ${CMAKE_COMMAND}
      "-DACTUAL_COMMAND=${encoded_command}"
      "-DEXPECTED_FILE=${GOLDEN_EXPECTED}"
      -DGOLDEN_UPDATE=ON
      -P "${runner}"
  )
endfunction()
