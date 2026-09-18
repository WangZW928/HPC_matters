if (NOT DEFINED exe OR NOT DEFINED inputs OR NOT DEFINED work)
    message(FATAL_ERROR "p10_taylor_green_convergence.cmake requires exe, inputs, and work")
endif()

file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
set(resolutions 16 32 64)
foreach (n IN LISTS resolutions)
    set(report "${work}/taylor_green_${n}.json")
    execute_process(
        COMMAND "${exe}" "${inputs}/p10_taylor_green_mms_${n}.in"
                "avwis.metadata_file=${report}"
        RESULT_VARIABLE status
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if (NOT status EQUAL 0)
        message(FATAL_ERROR "P10-001 n=${n} failed:\n${stdout}${stderr}")
    endif()
    file(READ "${report}" json)
    string(JSON adv_${n} GET "${json}" advection_l2)
    string(JSON visc_${n} GET "${json}" viscous_l2)
    string(JSON legacy_${n} GET "${json}" legacy_reference max_abs_error)
    string(JSON time_ratio_${n} GET "${json}" time_refinement_ratio)
endforeach()

execute_process(
    COMMAND awk -v a16=${adv_16} -v a32=${adv_32} -v a64=${adv_64}
                -v v16=${visc_16} -v v32=${visc_32} -v v64=${visc_64}
                -v l16=${legacy_16} -v l32=${legacy_32} -v l64=${legacy_64}
                -v t16=${time_ratio_16} -v t32=${time_ratio_32} -v t64=${time_ratio_64}
                "BEGIN { if (!(a16/a32 > 3.2 && a32/a64 > 3.2 && v16/v32 > 3.2 && v32/v64 > 3.2 && l16 < 1e-12 && l32 < 1e-12 && l64 < 1e-12 && t16 > 1.5 && t32 > 1.5 && t64 > 1.5)) exit 1; printf \"advection ratios %.8g %.8g; viscous ratios %.8g %.8g; time ratios %.8g %.8g %.8g\\n\", a16/a32, a32/a64, v16/v32, v32/v64, t16, t32, t64 }"
    RESULT_VARIABLE convergence_status
    OUTPUT_VARIABLE convergence_output
    ERROR_VARIABLE convergence_error)
if (NOT convergence_status EQUAL 0)
    message(FATAL_ERROR
        "P10-001 convergence/legacy gate failed:\n"
        "adv=${adv_16},${adv_32},${adv_64}\n"
        "visc=${visc_16},${visc_32},${visc_64}\n"
        "legacy=${legacy_16},${legacy_32},${legacy_64}\n"
        "time ratios=${time_ratio_16},${time_ratio_32},${time_ratio_64}\n"
        "${convergence_error}")
endif()
message(STATUS "P10-001 Taylor-Green MMS PASS: ${convergence_output}")
