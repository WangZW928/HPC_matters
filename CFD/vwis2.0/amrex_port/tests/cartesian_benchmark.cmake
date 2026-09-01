if (NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED report)
  message(FATAL_ERROR "exe, input, and report are required")
endif()
file(REMOVE "${report}")
execute_process(
  COMMAND "${exe}" "${input}" "avwis.metadata_file=${report}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE stdout
  ERROR_VARIABLE stderr)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "Cartesian benchmark failed (${result})\n${stdout}\n${stderr}")
endif()
if (NOT EXISTS "${report}")
  message(FATAL_ERROR "Cartesian benchmark did not create ${report}")
endif()
file(READ "${report}" json)
foreach (required
    "\"status\": \"PASS\""
    "\"case_type\": \"manufactured/contract baseline\""
    "\"legacy_reference\": false"
    "\"steps\": 8"
    "\"post_projection_max_abs_divergence\""
    "\"net_flux\""
    "\"momentum\""
    "\"kinetic_energy\""
    "\"step_seconds\""
    "\"total_step_seconds\""
    "\"compiler\""
    "\"amrex_version\"")
  string(FIND "${json}" "${required}" found)
  if (found EQUAL -1)
    message(FATAL_ERROR "Cartesian benchmark report is missing ${required}\n${json}")
  endif()
endforeach()
message(STATUS "Cartesian benchmark PASS: ${report}")
