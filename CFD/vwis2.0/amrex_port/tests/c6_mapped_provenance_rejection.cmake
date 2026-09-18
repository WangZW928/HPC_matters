if (NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED mismatch_input OR NOT DEFINED work)
    message(FATAL_ERROR "c6_mapped_provenance_rejection requires exe, input, mismatch_input, and work")
endif()
file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
set(checkpoint "${work}/checkpoint")
execute_process(
    COMMAND "${exe}" "${input}" "avwis.checkpoint_file=${checkpoint}"
    RESULT_VARIABLE write_status
    OUTPUT_VARIABLE write_output
    ERROR_VARIABLE write_error)
if (NOT write_status EQUAL 0)
    message(FATAL_ERROR "C6 mapped checkpoint producer failed:\n${write_output}${write_error}")
endif()
file(READ "${checkpoint}/Header" header)
foreach (required
         "schema_version 3"
         "coordinate_system mapped"
         "mapping_type analytic_orthogonal"
         "projection_operator orthogonal_mlmg"
         "metric_discretization avwis-shared-triangle-metric-v1"
         "mapping_provenance_checksum ")
    string(FIND "${header}" "${required}" found)
    if (found EQUAL -1)
        message(FATAL_ERROR "C6 Header is missing '${required}'")
    endif()
endforeach()
execute_process(
    COMMAND "${exe}" "${mismatch_input}" "avwis.restart_file=${checkpoint}"
    RESULT_VARIABLE restart_status
    OUTPUT_VARIABLE restart_output
    ERROR_VARIABLE restart_error)
if (restart_status EQUAL 0)
    message(FATAL_ERROR "C6 accepted mismatched mapping provenance")
endif()
string(FIND "${restart_output}${restart_error}" "mapping_stretch mismatch" found)
if (found EQUAL -1)
    message(FATAL_ERROR
        "C6 failed for the wrong reason; expected mapping_stretch mismatch:\n"
        "${restart_output}${restart_error}")
endif()
