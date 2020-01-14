cd embedding_algorithms

# build GloVe
cd GloVe && make && cd ..

# build Hazy
cd Hazy
mkdir -p build && cd build && cmake ..
make && cd ..
cd ..

# build word2vec
cd word2vec && make
cd ..

cd ..