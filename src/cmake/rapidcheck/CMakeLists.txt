cmake_minimum_required(VERSION 3.30)
project(rapidcheck)

include(FetchContent)

set(RC_ENABLE_CATCH ON)

FetchContent_Declare(
	rapidcheck
	GIT_SHALLOW    TRUE
	GIT_REPOSITORY https://github.com/emil-e/rapidcheck.git
	GIT_TAG        ff6af6fc683159deb51c543b065eba14dfcf329b)
FetchContent_MakeAvailable(rapidcheck)
