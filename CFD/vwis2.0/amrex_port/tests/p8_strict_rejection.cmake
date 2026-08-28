if(NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED restart_input OR NOT DEFINED work)
  message(FATAL_ERROR "p8_strict_rejection requires exe, input, restart_input, and work")
endif()
string(TIMESTAMP stamp "%s")
string(RANDOM LENGTH 8 ALPHABET 0123456789abcdef nonce)
set(run "${work}_${stamp}_${nonce}")
file(MAKE_DIRECTORY "${run}")
execute_process(
  COMMAND "${exe}" "${input}" "vwis.checkpoint_file=${run}/checkpoint"
  WORKING_DIRECTORY "${run}"
  RESULT_VARIABLE write_result
  OUTPUT_VARIABLE write_output
  ERROR_VARIABLE write_error)
if(NOT write_result EQUAL 0)
  message(FATAL_ERROR "P8 checkpoint producer failed (${write_result})\n${write_output}\n${write_error}")
endif()
set(header "${run}/checkpoint/Header")
file(READ "${header}" contents)
string(REPLACE "VWIS_AMREX_CARTESIAN_CHECKPOINT" "CORRUPTED_CHECKPOINT_HEADER" corrupted "${contents}")
if(corrupted STREQUAL contents)
  message(FATAL_ERROR "P8 test could not corrupt Header magic")
endif()
file(WRITE "${header}" "${corrupted}")
execute_process(
  COMMAND "${exe}" "${restart_input}" "vwis.restart_file=${run}/checkpoint"
  WORKING_DIRECTORY "${run}"
  RESULT_VARIABLE restart_result
  OUTPUT_VARIABLE restart_output
  ERROR_VARIABLE restart_error)
if(restart_result EQUAL 0)
  message(FATAL_ERROR "corrupted P8 Header was accepted\n${restart_output}\n${restart_error}")
endif()
string(FIND "${restart_output}${restart_error}" "P8 checkpoint rejected:" found)
if(found EQUAL -1)
  message(FATAL_ERROR "strict rejection diagnostic missing\n${restart_output}\n${restart_error}")
endif()
