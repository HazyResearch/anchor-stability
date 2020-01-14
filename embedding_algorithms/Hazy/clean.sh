echo "Cleaning directories..."
rm -rf lib_cpp/*
rm -rf build
rm -rf *.egg*
rm -rf *.so

if [ $(id -u) = 0 ]; then
   echo "Cleaning /usr/lib..."
   rm -rf /usr/lib/libcppembeddings*
   rm -rf /usr/local/lib/libcppembeddings*
fi

if [ -x "$(command -v pip3)" ]; then
    echo "Uninstalling python packages..."
    pip3 uninstall hazy
    pip3 uninstall hazytensor
fi

if [ -x "$(command -v conda)" ]; then
  echo "Purging conda build..."
  conda uninstall hazy
  conda build purge
fi
