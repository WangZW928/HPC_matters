if (NOT DEFINED exe OR NOT DEFINED producer_input OR NOT DEFINED consumer_input OR
    NOT DEFINED work)
    message(FATAL_ERROR
        "c6_mapped_cross_layout requires exe, producer_input, consumer_input, and work")
endif()

file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
set(checkpoint "${work}/checkpoint")
execute_process(
    COMMAND "${exe}" "${producer_input}" "avwis.checkpoint_file=${checkpoint}"
    RESULT_VARIABLE write_status
    OUTPUT_VARIABLE write_output
    ERROR_VARIABLE write_error)
if (NOT write_status EQUAL 0)
    message(FATAL_ERROR "C6 cross-layout producer failed:\n${write_output}${write_error}")
endif()

execute_process(
    COMMAND "${exe}" "${consumer_input}" "avwis.restart_file=${checkpoint}"
    RESULT_VARIABLE read_status
    OUTPUT_VARIABLE read_output
    ERROR_VARIABLE read_error)
if (NOT read_status EQUAL 0)
    message(FATAL_ERROR "C6 cross-layout restart failed:\n${read_output}${read_error}")
endif()
string(FIND "${read_output}${read_error}" "AVWiS P8 checkpoint restored" found)
if (found EQUAL -1)
    message(FATAL_ERROR "C6 cross-layout restart marker missing:\n${read_output}${read_error}")
endif()
message(STATUS "C6 mapped cross-layout checkpoint/restart PASS")
