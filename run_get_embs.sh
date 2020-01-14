# create folders for the embeddings
mkdir -p runs/embs/wiki_2017
mkdir -p runs/embs/wiki_2018

# download embeddings from the Google Cloud bucket
wget https://storage.googleapis.com/mlsys-artifact-evaluation/mc.2017.tar.gz
wget https://storage.googleapis.com/mlsys-artifact-evaluation/mc.2018.tar.gz
wget https://storage.googleapis.com/mlsys-artifact-evaluation/w2v_cbow.2017.tar.gz
wget https://storage.googleapis.com/mlsys-artifact-evaluation/w2v_cbow.2018.tar.gz

# extract the embeddings into corresponding folders
tar -xzvf mc.2017.tar.gz -C runs/embs/wiki_2017
tar -xzvf mc.2018.tar.gz -C runs/embs/wiki_2018
tar -xzvf w2v_cbow.2017.tar.gz -C runs/embs/wiki_2017
tar -xzvf w2v_cbow.2018.tar.gz -C runs/embs/wiki_2018