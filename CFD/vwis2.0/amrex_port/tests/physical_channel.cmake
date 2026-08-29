if (NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED report)
  message(FATAL_ERROR "exe, input, and report are required")
endif()
file(REMOVE "${report}")
execute_process(
  COMMAND "${exe}" "${input}" "vwis.metadata_file=${report}"
  RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "Physical channel run failed (${result})\n${stdout}\n${stderr}")
endif()
if (NOT EXISTS "${report}")
  message(FATAL_ERROR "Physical channel did not create ${report}")
endif()
file(READ "${report}" json)
foreach (required
    "\"status\": \"physical run / not yet validated\""
    "\"reference_available\": false"
    "\"case_type\": \"physical Cartesian plane channel\""
    "\"boundary_conditions\""
    "\"post_projection_max_abs_divergence\""
    "\"integrated_divergence\""
    "\"net_mass_flux\""
    "\"outlet_flow\""
    "\"section_mean_u_out\""
    "\"centerline_u\""
    "\"pressure_drop\""
    "\"pressure_mean\""
    "\"pressure_min\""
    "\"pressure_max\""
    "\"momentum\""
    "\"kinetic_energy\""
    "\"step_seconds\""
    "\"time_average_method\""
    "\"time_averages\""
    "\"sample_count\": 40"
    "\"total_step_seconds\"")
  string(FIND "${json}" "${required}" found)
  if (found EQUAL -1)
    message(FATAL_ERROR "Physical channel report is missing ${required}\n${json}")
  endif()
endforeach()
message(STATUS "Physical channel run complete (not yet validated): ${report}")
