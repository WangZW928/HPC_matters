if (NOT DEFINED exe OR NOT DEFINED inputs OR NOT DEFINED work)
    message(FATAL_ERROR
        "c8_mapped_channel_convergence.cmake requires exe, inputs, and work")
endif()

file(REMOVE_RECURSE "${work}")
file(MAKE_DIRECTORY "${work}")
set(resolutions 16 32 64)
foreach (n IN LISTS resolutions)
    set(report "${work}/mapped_channel_${n}.json")
    set(profile "${work}/mapped_channel_${n}.csv")
    execute_process(
        COMMAND "${exe}" "${inputs}/c8_mapped_channel_${n}.in"
                "avwis.metadata_file=${report}"
                "avwis.plane_file=${profile}"
        RESULT_VARIABLE status
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if (NOT status EQUAL 0)
        message(FATAL_ERROR "C8-G1 n=${n} failed:\n${stdout}${stderr}")
    endif()
    if (NOT EXISTS "${profile}")
        message(FATAL_ERROR "C8-G1 n=${n} did not create profile CSV")
    endif()
    file(READ "${report}" json)
    string(JSON velocity_${n} GET "${json}" velocity_l2)
    string(JSON residual_${n} GET "${json}" steady_residual_l2)
    string(JSON flow_${n} GET "${json}" flow_rate_relative_error)
    string(JSON shear_${n} GET "${json}" wall_shear_max_abs_error)
    string(JSON pressure_${n} GET "${json}" pressure_datum_free_linf)
    string(JSON divergence_${n} GET "${json}" divergence_linf)
    string(JSON legacy_available_${n} GET "${json}" legacy_reference executable_reference_available)
endforeach()

execute_process(
    COMMAND awk
        -v u16=${velocity_16} -v u32=${velocity_32} -v u64=${velocity_64}
        -v r16=${residual_16} -v r32=${residual_32} -v r64=${residual_64}
        -v f16=${flow_16} -v f32=${flow_32} -v f64=${flow_64}
        -v s16=${shear_16} -v s32=${shear_32} -v s64=${shear_64}
        -v p16=${pressure_16} -v p32=${pressure_32} -v p64=${pressure_64}
        -v d16=${divergence_16} -v d32=${divergence_32} -v d64=${divergence_64}
        "BEGIN {
          if (!(u16/u32 > 2.5 && u32/u64 > 2.5 &&
                r16/r32 > 2.5 && r32/r64 > 2.5 &&
                f16/f32 > 2.5 && f32/f64 > 2.5 &&
                s16/s32 > 1.8 && s32/s64 > 1.8 &&
                u64 < 1e-6 && r64 < 1e-4 && f64 < 2e-4 && s64 < 1e-6 &&
                p16 < 1e-11 && p32 < 1e-11 && p64 < 1e-11 &&
                d16 < 1e-11 && d32 < 1e-11 && d64 < 1e-11)) exit 1;
          printf \"velocity ratios %.8g %.8g; residual %.8g %.8g; flow %.8g %.8g; shear %.8g %.8g\\n\",
                 u16/u32,u32/u64,r16/r32,r32/r64,f16/f32,f32/f64,s16/s32,s32/s64
        }"
    RESULT_VARIABLE convergence_status
    OUTPUT_VARIABLE convergence_output
    ERROR_VARIABLE convergence_error)
if (NOT convergence_status EQUAL 0)
    message(FATAL_ERROR
        "C8-G1 convergence gate failed:\n"
        "velocity=${velocity_16},${velocity_32},${velocity_64}\n"
        "residual=${residual_16},${residual_32},${residual_64}\n"
        "flow=${flow_16},${flow_32},${flow_64}\n"
        "shear=${shear_16},${shear_32},${shear_64}\n"
        "pressure=${pressure_16},${pressure_32},${pressure_64}\n"
        "divergence=${divergence_16},${divergence_32},${divergence_64}\n"
        "${convergence_error}")
endif()
if (legacy_available_16 OR legacy_available_32 OR legacy_available_64)
    message(FATAL_ERROR "C8-G1 must not claim an unavailable legacy executable run")
endif()
message(STATUS "C8-G1 mapped channel reference PASS: ${convergence_output}")
