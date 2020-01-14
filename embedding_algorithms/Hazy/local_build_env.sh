mkdir -p lib_cpp
export HAZY_C_PREFIX=$(pwd)/lib_cpp
if [ "$(uname)" == "Darwin" ]; then
    # Do nothing under Mac OS X platform
    export DYLD_LIBRARY_PATH=$HAZY_C_PREFIX:$DYLD_LIBRARY_PATH
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
    export LD_LIBRARY_PATH=$HAZY_C_PREFIX:$LD_LIBRARY_PATH
else
    echo "ERROR: Platform not supported..."
fi