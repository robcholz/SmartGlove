idf_build_get_property(component_targets __COMPONENT_TARGETS)
if ("___idf_espressif__esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir espressif__esp-dl COMPONENT_DIR)
elseif("___idf_esp-dl" IN_LIST component_targets)
    idf_component_get_property(espdl_dir esp-dl COMPONENT_DIR)
endif()

set(embed_file "model.espdl")
set(utilities_cmake "${espdl_dir}/fbs_loader/cmake/utilities.cmake")
set(cmake_dir "${espdl_dir}/fbs_loader/cmake")
set(component_embed_args)

if(NOT EXISTS "${utilities_cmake}")
    set(component_embed_args EMBED_FILES ${embed_file})
endif()

idf_component_register(
    SRCS "espdl_experiment.cpp"
    INCLUDE_DIRS "."
    REQUIRES espressif__esp-dl
    ${component_embed_args}
)

if(EXISTS "${utilities_cmake}")
    include("${utilities_cmake}")
    target_add_aligned_binary_data(${COMPONENT_LIB} "${embed_file}" BINARY)
endif()
