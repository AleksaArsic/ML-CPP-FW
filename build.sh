# Usage:
# This script will build project source using CMake 
# e.g. $ build.sh --build
#!/bin/bash 

function usage() {
    cat <<USAGE

    Usage: $0 [--clean-build | --build]

    Options:
        --clean-build:  Clean build      
        --build:        Build without clean
USAGE
    exit 1
}

# clean build folder and all of it's content
function clean()
{
    echo "Removing old build files"
    rm -rf build
}

# use cmake to build project source
function build()
{
    mkdir build
    cd build 
    cmake ..
    cmake --build .
}

# if no arguments are provided or more than one argument is provided, return usage function
if [ $# -eq 0 ] || [ $# -ne 1 ]; then
    usage # run usage function
    exit 1
fi

# start build
case $1 in
--clean-build) # start clean build
    echo "Starting clean build"
    clean
    build
    ;;
--build) # standard build
    echo "Starting build"
    build
    ;;
*)
    usage
    exit 1
    ;;
esac




