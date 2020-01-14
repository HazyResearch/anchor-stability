ANALYSISDIR=runs/analysis
EMBDIR=runs/embs
MODELDIR=runs/models

mkdir -p ${ANALYSISDIR}
mkdir -p ${ANALYSISDIR}/distances

## Model instability
bash scripts/analysis/eval_wiki_compressed.sh sst mc 0.001 ${ANALYSISDIR}/distances ${MODELDIR}
bash scripts/analysis/eval_wiki_compressed.sh sst w2v_cbow 0.0001 ${ANALYSISDIR}/distances ${MODELDIR}

## Embedding instability

# Eigenspace instability
python scripts/analysis/create_anchors.py --algo mc
python scripts/analysis/create_anchors.py --algo w2v_cbow
bash scripts/analysis/eval_wiki_compressed_embs_ei.sh mc 3 ${ANALYSISDIR}/distances ${EMBDIR}
bash scripts/analysis/eval_wiki_compressed_embs_ei.sh w2v_cbow 3 ${ANALYSISDIR}/distances ${EMBDIR}

# k-NN
bash scripts/analysis/eval_wiki_compressed_embs_nn.sh mc 5 1000 ${ANALYSISDIR}/distances ${EMBDIR}
bash scripts/analysis/eval_wiki_compressed_embs_nn.sh w2v_cbow 5 1000 ${ANALYSISDIR}/distances ${EMBDIR}

# Semantic displacement
bash scripts/analysis/eval_wiki_compressed_embs_baselines.sh mc sem_disp ${ANALYSISDIR}/distances ${EMBDIR}
bash scripts/analysis/eval_wiki_compressed_embs_baselines.sh w2v_cbow sem_disp ${ANALYSISDIR}/distances ${EMBDIR}

# PIP loss
bash scripts/analysis/eval_wiki_compressed_embs_baselines.sh mc pip ${ANALYSISDIR}/distances ${EMBDIR}
bash scripts/analysis/eval_wiki_compressed_embs_baselines.sh w2v_cbow pip ${ANALYSISDIR}/distances ${EMBDIR}

# 1-Eigenspace overlap
bash scripts/analysis/eval_wiki_compressed_embs_baselines.sh mc eigen_overlap ${ANALYSISDIR}/distances ${EMBDIR}
bash scripts/analysis/eval_wiki_compressed_embs_baselines.sh w2v_cbow eigen_overlap ${ANALYSISDIR}/distances ${EMBDIR}

# Gather all results into a single CSV per embedding algorithm
python scripts/analysis/gather_results.py --algo mc --datadir ${ANALYSISDIR}/distances --resultdir ${ANALYSISDIR} \
    --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 \
    --ds_metrics la_sst_no_emb_norm
python scripts/analysis/gather_results.py --algo w2v_cbow --datadir ${ANALYSISDIR}/distances --resultdir ${ANALYSISDIR} \
    --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 \
    --ds_metrics la_sst_no_emb_norm