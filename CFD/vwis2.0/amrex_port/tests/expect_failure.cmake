if (NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED expected)
    message(FATAL_ERROR "expect_failure.cmake requires exe, input, and expected")
endif()

execute_process(
    COMMAND "${exe}" "${input}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)

if (status EQUAL 0)
    message(FATAL_ERROR
        "Expected command to fail, but it succeeded:\n${exe} ${input}\n${stdout}${stderr}")
endif()

set(combined "${stdout}${stderr}")
string(FIND "${combined}" "${expected}" match_at)
if (match_at EQUAL -1)
    message(FATAL_ERROR
        "Command failed for the wrong reason. Expected text:\n${expected}\n"
        "Actual output:\n${combined}")
endif()
