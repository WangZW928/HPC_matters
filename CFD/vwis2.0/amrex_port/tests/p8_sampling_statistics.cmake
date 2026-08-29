if (NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED report OR NOT DEFINED plane)
  message(FATAL_ERROR "exe, input, report, and plane are required")
endif()
file(REMOVE "${report}" "${plane}")
execute_process(
  COMMAND "${exe}" "${input}" "vwis.metadata_file=${report}" "vwis.plane_file=${plane}"
  RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "P8 sampling/statistics contract failed (${result})\n${stdout}\n${stderr}")
endif()
if (NOT EXISTS "${report}" OR NOT EXISTS "${plane}")
  message(FATAL_ERROR "P8 sampling/statistics outputs were not created")
endif()
file(READ "${report}" json)
foreach (required
    "\"schema\": \"vwis-uniform-diagnostics-v1\""
    "\"status\": \"PASS\""
    "\"case_type\": \"manufactured contract test\""
    "\"plotfile_compatible\": false"
    "\"cell\": [2,1,1]"
    "\"pressure\": 211"
    "\"cell_count\": 6"
    "\"mean_pressure\": 110.5"
    "\"normal_flow\": 1"
    "\"momentum\""
    "\"kinetic_energy\""
    "\"pressure_mean\": 160.5"
    "\"pressure_min\": 0"
    "\"pressure_max\": 321")
  string(FIND "${json}" "${required}" found)
  if (found EQUAL -1)
    message(FATAL_ERROR "P8 diagnostics JSON is missing ${required}\n${json}")
  endif()
endforeach()
file(STRINGS "${plane}" rows)
list(LENGTH rows row_count)
if (NOT row_count EQUAL 7)
  message(FATAL_ERROR "P8 plane CSV must contain one header plus 6 rows, got ${row_count}")
endif()
list(GET rows 0 header)
if (NOT header STREQUAL "i,j,k,x,y,z,u,v,w,pressure,divergence")
  message(FATAL_ERROR "Unexpected P8 plane CSV header: ${header}")
endif()
message(STATUS "P8-003 sampling/statistics contract PASS: ${report}; ${plane}")
