set -e
echo "Downloading intrinsic testsets..."
url=http://i.stanford.edu/hazy/share/hazy_embedding/hazy_intrinsic_testsets.tar.gz
data_tar=hazy_intrinsic_testsets
if type curl &>/dev/null; then
    curl -RLO $url
elif type wget &>/dev/null; then
    wget -N -nc $url
fi

mkdir -p testsets
echo "Unpacking intrinsic testsets..."
tar -zxvf $data_tar.tar.gz -C testsets

echo "Deleting tar file..."
rm $data_tar.tar.gz

echo "Done!"
