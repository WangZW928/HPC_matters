if (NOT DEFINED exe OR NOT DEFINED input OR NOT DEFINED work)
  message(FATAL_ERROR "exe, input, and work are required")
endif()
file(MAKE_DIRECTORY "${work}")
set(report "${work}/summary.json")
set(field "${work}/field.csv")
set(centerline "${work}/centerlines.csv")
set(history "${work}/history.csv")
file(REMOVE "${report}" "${field}" "${centerline}" "${history}")
execute_process(
  COMMAND "${exe}" "${input}"
    "vwis.metadata_file=${report}"
    "vwis.field_file=${field}"
    "vwis.centerline_file=${centerline}"
    "vwis.history_file=${history}"
  RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
if (NOT result EQUAL 0)
  message(FATAL_ERROR "Lid cavity sanity run failed (${result})\n${stdout}\n${stderr}")
endif()
foreach (artifact "${report}" "${field}" "${centerline}" "${history}")
  if (NOT EXISTS "${artifact}")
    message(FATAL_ERROR "Lid cavity did not create ${artifact}")
  endif()
endforeach()
file(READ "${report}" json)
foreach (required
    "\"schema\": \"vwis-lid-driven-cavity-v1\""
    "\"status\": \"demonstration / engineering result; not CFD validation\""
    "\"grid\": [8, 8, 1]"
    "\"reynolds_number\": 100"
    "\"all_finite\": true"
    "\"no_nan_or_inf\": true"
    "\"moving_wall_reconstruction_max_error\""
    "\"max_post_projection_abs_divergence\""
    "\"max_abs_net_mass_flux\""
    "\"centerline_method\""
    "\"limitations\"")
  string(FIND "${json}" "${required}" found)
  if (found EQUAL -1)
    message(FATAL_ERROR "Lid cavity report is missing ${required}\n${json}")
  endif()
endforeach()
file(STRINGS "${field}" field_lines)
file(STRINGS "${centerline}" centerline_lines)
file(STRINGS "${history}" history_lines)
list(LENGTH field_lines field_count)
list(LENGTH centerline_lines centerline_count)
list(LENGTH history_lines history_count)
if (NOT field_count EQUAL 65 OR NOT centerline_count EQUAL 17 OR NOT history_count EQUAL 9)
  message(FATAL_ERROR "Unexpected cavity CSV row counts: field=${field_count}, centerline=${centerline_count}, history=${history_count}")
endif()
message(STATUS "Lid cavity deterministic schema/basic sanity: PASS (finite/divergence/mass/wall guards; not validation)")
