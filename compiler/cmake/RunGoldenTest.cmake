# Script-mode helper invoked by add_golden_test() (see SigflowGoldenTest.cmake). Runs
# ACTUAL_COMMAND from WORKING_DIR, captures its stdout and stderr together plus its exit code,
# and either compares that to EXPECTED_FILE or, with GOLDEN_UPDATE set, overwrites EXPECTED_FILE
# with it.
if(NOT ACTUAL_COMMAND OR NOT EXPECTED_FILE)
  message(FATAL_ERROR "RunGoldenTest.cmake needs ACTUAL_COMMAND and EXPECTED_FILE")
endif()
if(NOT WORKING_DIR)
  set(WORKING_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# ACTUAL_COMMAND arrives "::"-joined (see SigflowGoldenTest.cmake); turn it back into a list.
string(REPLACE "::" ";" ACTUAL_COMMAND "${ACTUAL_COMMAND}")

# Naming the same variable for both pipes merges them in the order they were written.
execute_process(
  COMMAND ${ACTUAL_COMMAND}
  WORKING_DIRECTORY "${WORKING_DIR}"
  OUTPUT_VARIABLE actual_output
  ERROR_VARIABLE actual_output
  RESULT_VARIABLE actual_result
)
# Same format as sigflow::golden_text() in src/driver.cpp.
if(NOT actual_result EQUAL 0)
  string(APPEND actual_output "[exit ${actual_result}]\n")
endif()

if(GOLDEN_UPDATE)
  file(WRITE "${EXPECTED_FILE}" "${actual_output}")
  message(STATUS "Updated golden file: ${EXPECTED_FILE}")
  return()
endif()

if(NOT EXISTS "${EXPECTED_FILE}")
  message(FATAL_ERROR
    "Missing golden file ${EXPECTED_FILE}; regenerate it with the update_golden_<name> target")
endif()

file(READ "${EXPECTED_FILE}" expected_output)
if(NOT actual_output STREQUAL expected_output)
  message(FATAL_ERROR
    "Golden mismatch for ${EXPECTED_FILE}\n"
    "--- expected ---\n${expected_output}\n"
    "--- actual ---\n${actual_output}\n")
endif()
