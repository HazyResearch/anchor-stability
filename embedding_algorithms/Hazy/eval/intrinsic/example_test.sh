#Put your embedding folder here
folder=embedding_evaluation/test/test/text8

#Put your embeddintg files that in the embedding folder here
embs=(vectors.txt)

ws_tasks=(ws353_similarity.txt ws353_relatedness.txt bruni_men.txt radinsky_mturk.txt luong_rare.txt simlex999.txt)
ws_taskfolder=testsets/ws

an_tasks=(google.txt msr.txt)
an_taskfolder=testsets/analogy

for emb in ${embs[*]};
do
    for task in ${ws_tasks[*]};
    do
        python ws_eval.py GLOVE ${folder}/${emb} ${ws_taskfolder}/${task}
    done
    for task in ${an_tasks[*]};
    do
        python analogy_eval.py GLOVE ${folder}/${emb} ${an_taskfolder}/${task}
    done
done

