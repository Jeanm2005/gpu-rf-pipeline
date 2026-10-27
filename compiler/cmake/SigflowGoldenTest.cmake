# add_golden_test(<name> COMMAND <command-and-args...> EXPECTED <golden-file>
#                 [WORKING_DIRECTORY <dir>])
#
# Registers a CTest case comparing COMMAND's combined stdout and stderr, byte for byte, against
# EXPECTED (compiler/AGENTS.md: "Golden files compare exact text"). A non-zero exit code is
# appended as a final "[exit N]" line, so negative tests pin both the diagnostics and the
# failure. Also registers a target update_golden_<name> that regenerates EXPECTED from the
# current output; run it only when the owner asks, then review the diff (compiler/AGENTS.md).
#
# add_sigflowc_golden_tests(<dir>)
#
# Registers one golden test per <dir>/<name>.sf. The first line of each program may give the
# sigflowc flags as "# args: --emit=tokens"; the case runs `sigflowc <flags> <name>.sf` from
# <dir> and compares against <dir>/<name>.expected. `sigflowc --update-golden <dir>` writes the
# same text in-process (src/driver.cpp, golden_text()).
#
# Captured at include() time: inside the functions below, CMAKE_CURRENT_LIST_DIR would instead
# resolve to the caller's directory (tests/), not this file's.
set(SIGFLOW_GOLDEN_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(add_golden_test name)
  cmake_parse_arguments(GOLDEN "" "EXPECTED;WORKING_DIRECTORY" "COMMAND" ${ARGN})
  if(NOT GOLDEN_COMMAND OR NOT GOLDEN_EXPECTED)
    message(FATAL_ERROR "add_golden_test(${name}): COMMAND and EXPECTED are both required")
  endif()
  if(NOT GOLDEN_WORKING_DIRECTORY)
    set(GOLDEN_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
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
      "-DWORKING_DIR=${GOLDEN_WORKING_DIRECTORY}"
      -P "${runner}"
  )
  add_custom_target(update_golden_${name}
    COMMAND ${CMAKE_COMMAND}
      "-DACTUAL_COMMAND=${encoded_command}"
      "-DEXPECTED_FILE=${GOLDEN_EXPECTED}"
      "-DWORKING_DIR=${GOLDEN_WORKING_DIRECTORY}"
      -DGOLDEN_UPDATE=ON
      -P "${runner}"
  )
endfunction()

function(add_sigflowc_golden_tests dir)
  file(GLOB programs CONFIGURE_DEPENDS "${dir}/*.sf")
  list(SORT programs)
  foreach(program IN LISTS programs)
    get_filename_component(name "${program}" NAME_WE)
    get_filename_component(file_name "${program}" NAME)
    file(STRINGS "${program}" first_line LIMIT_COUNT 1)
    set(flags "")
    if(first_line MATCHES "^# args:(.*)$")
      string(STRIP "${CMAKE_MATCH_1}" flags)
      separate_arguments(flags UNIX_COMMAND "${flags}")
    endif()
    add_golden_test(${name}
      COMMAND $<TARGET_FILE:sigflowc> ${flags} ${file_name}
      EXPECTED "${dir}/${name}.expected"
      WORKING_DIRECTORY "${dir}"
    )
  endforeach()
endfunction()
