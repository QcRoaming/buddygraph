configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(BUDDYGRAPH_TEST_DEPENDS
  FileCheck
  count
  not
  buddygraph-opt
)

add_lit_testsuite(check-buddygraph "Running the BuddyGraph regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${BUDDYGRAPH_TEST_DEPENDS}
)
set_target_properties(check-buddygraph PROPERTIES FOLDER "Tests")

add_lit_testsuites(BUDDYGRAPH ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${BUDDYGRAPH_TEST_DEPENDS})
