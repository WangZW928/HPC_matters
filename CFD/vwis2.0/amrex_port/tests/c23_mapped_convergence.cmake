if (NOT DEFINED exe OR NOT DEFINED inputs OR NOT DEFINED work)
    message(FATAL_ERROR "c23_mapped_convergence.cmake requires exe, inputs, and work")
endif()

file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
set(resolutions 16 32 64)
foreach (n IN LISTS resolutions)
    set(report "${work}/mapped_${n}.json")
    execute_process(
        COMMAND "${exe}" "${inputs}/p5_c23_mapped_time_${n}.in"
                "avwis.metadata_file=${report}"
        RESULT_VARIABLE status
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if (NOT status EQUAL 0)
        message(FATAL_ERROR "C2.3 n=${n} failed:\n${stdout}${stderr}")
    endif()
    file(READ "${report}" json)
    string(JSON adv_${n} GET "${json}" advection_l2)
    string(JSON visc_${n} GET "${json}" viscous_l2)
    string(JSON step_${n} GET "${json}" manufactured_step_error)
    string(JSON free_${n} GET "${json}" free_stream_step_error)
endforeach()

execute_process(
    COMMAND awk -v a16=${adv_16} -v a32=${adv_32} -v a64=${adv_64}
                -v v16=${visc_16} -v v32=${visc_32} -v v64=${visc_64}
                -v s16=${step_16} -v s32=${step_32} -v s64=${step_64}
                -v f16=${free_16} -v f32=${free_32} -v f64=${free_64}
                "BEGIN { if (!(a16/a32 > 3.2 && a32/a64 > 3.2 && v16/v32 > 3.2 && v32/v64 > 3.2 && s16 < 1e-9 && s32 < 1e-9 && s64 < 1e-9 && f16 < 1e-10 && f32 < 1e-10 && f64 < 1e-10)) exit 1; printf \"advection ratios %.8g %.8g; viscous ratios %.8g %.8g\\n\", a16/a32, a32/a64, v16/v32, v32/v64 }"
    RESULT_VARIABLE convergence_status
    OUTPUT_VARIABLE convergence_output
    ERROR_VARIABLE convergence_error)
if (NOT convergence_status EQUAL 0)
    message(FATAL_ERROR
        "C2.3 convergence/free-stream gate failed:\n"
        "adv=${adv_16},${adv_32},${adv_64}\n"
        "visc=${visc_16},${visc_32},${visc_64}\n"
        "step=${step_16},${step_32},${step_64}\n"
        "free=${free_16},${free_32},${free_64}\n${convergence_error}")
endif()
message(STATUS "C2.3 mapped convergence PASS: ${convergence_output}")
