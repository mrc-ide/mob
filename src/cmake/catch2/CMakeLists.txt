cmake_minimum_required(VERSION 3.30)
project(Catch2)

include(FetchContent)

FetchContent_Declare(
	Catch2
	GIT_SHALLOW    TRUE
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        914aeecfe23b1e16af6ea675a4fb5dbd5a5b8d0a)

FetchContent_MakeAvailable(Catch2)

FetchContent_GetProperties(Catch2
	SOURCE_DIR CATCH2_SOURCE_DIR
	BINARY_DIR CATCH2_BINARY_DIR)

# Catch2 intentionally skips installation of its targets when included from
# another project. In our case though, the project is a shim that just serves
# to download and build Catch2 - installing the library really is what we're
# after.
#
# Those lines come from Catch2's own CMakeLists.txt:
# https://github.com/catchorg/Catch2/blob/76f70b1403dbc0781216f49e20e45b71f7eccdd8/src/CMakeLists.txt#L400-L435
#
install(
	TARGETS Catch2
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(
	DIRECTORY
		"${CATCH2_SOURCE_DIR}/src/"
		"${CATCH2_BINARY_DIR}/generated-includes/catch2"
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
	FILES_MATCHING
	PATTERN "*.hpp"
)
