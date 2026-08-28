if(NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED work)
  message(FATAL_ERROR "p8_restart_consistency requires exe, input, and work")
endif()
string(TIMESTAMP stamp "%s")
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef nonce)
set(run "${work}_${stamp}_${nonce}")
file(MAKE_DIRECTORY "${run}")
execute_process(
  COMMAND "${exe}" "${input}" "vwis.checkpoint_file=${run}/checkpoint"
  WORKING_DIRECTORY "${run}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "P8 restart consistency failed (${result})\n${output}\n${error}")
endif()
string(FIND "${output}${error}" "VWiS AMReX P8-001/P8-002: PASS" found)
if(found EQUAL -1)
  message(FATAL_ERROR "P8 PASS marker missing\n${output}\n${error}")
endif()
