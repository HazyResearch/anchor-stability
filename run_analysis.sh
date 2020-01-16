# runs all analysis scripts using saved csvs
DATASET=$1
ANALYSISDIR=runs/analysis

if [ ${DATASET} == "all" ]; then
    # Read from saved CSVs of all results
    echo 'Computing trends for linear-log model'
    python scripts/analysis/fit_trend.py --csv-files results/mc_optimal_no_emb_norm_top_10000.csv \
                                results/w2v_cbow_optimal_no_emb_norm_top_10000.csv \

    python scripts/analysis/fit_trend.py --csv-files results/mc_optimal_no_emb_norm_top_10000.csv \
                                results/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                --prec

    python scripts/analysis/fit_trend.py --csv-files results/mc_optimal_no_emb_norm_top_10000.csv \
                                results/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                --dim

    echo 'Computing spearman correlation results'
    python scripts/analysis/get_correlation.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 \
                                    --ds_metrics la_sst_no_emb_norm la_mr_no_emb_norm la_subj_no_emb_norm la_mpqa_no_emb_norm rnn_no_crf_ner \
                                    --csv-file results/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                    --out-file results/w2v_cbow_correlation.csv \

    python scripts/analysis/get_correlation.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 \
                                    --ds_metrics la_sst_no_emb_norm la_mr_no_emb_norm la_subj_no_emb_norm la_mpqa_no_emb_norm rnn_no_crf_ner \
                                    --csv-file results/mc_optimal_no_emb_norm_top_10000.csv \
                                    --out-file results/mc_correlation.csv \

    echo 'Generate selection criterion results'
    python scripts/analysis/selection_criterion.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 \
                                    --ds_metrics la_sst_no_emb_norm la_mr_no_emb_norm la_subj_no_emb_norm la_mpqa_no_emb_norm rnn_no_crf_ner \
                                    --csv-file results/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                    --acc-file results/w2v_cbow_selection_error.csv \
                                    --rob-file results/w2v_cbow_selection_robustness.csv

    python scripts/analysis/selection_criterion.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 \
                                    --ds_metrics la_sst_no_emb_norm la_mr_no_emb_norm la_subj_no_emb_norm la_mpqa_no_emb_norm rnn_no_crf_ner \
                                    --csv-file results/mc_optimal_no_emb_norm_top_10000.csv \
                                    --acc-file results/mc_selection_error.csv \
                                    --rob-file results/mc_selection_robustness.csv

    echo 'Generate results for difference to the oracle (same memory budget)'
    python scripts/analysis/diff_to_oracle.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 baseline_fp baseline_lp \
                                    --ds_metrics la_sst_no_emb_norm la_mr_no_emb_norm la_subj_no_emb_norm la_mpqa_no_emb_norm rnn_no_crf_ner \
                                    --csv-file results/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                    --out-file results/w2v_cbow_diff_to_oracle.csv

    python scripts/analysis/diff_to_oracle.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip eigen_overlap_top_10000 baseline_fp baseline_lp \
                                    --ds_metrics la_sst_no_emb_norm la_mr_no_emb_norm la_subj_no_emb_norm la_mpqa_no_emb_norm rnn_no_crf_ner \
                                    --csv-file results/mc_optimal_no_emb_norm_top_10000.csv \
                                    --out-file results/mc_diff_to_oracle.csv

else
    # Only run single task/dataset (e.g., SST-2 for reproducibility example)
    echo 'Only running analysis for single dataset'
    echo 'Computing trends for linear-log model'
    python scripts/analysis/fit_trend.py --csv-files ${ANALYSISDIR}/mc_optimal_no_emb_norm_top_10000.csv \
                                ${ANALYSISDIR}/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                --tasks la_${DATASET}_no_emb_norm

    python scripts/analysis/fit_trend.py --csv-files ${ANALYSISDIR}/mc_optimal_no_emb_norm_top_10000.csv \
                                ${ANALYSISDIR}/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                --prec --tasks la_${DATASET}_no_emb_norm

    python scripts/analysis/fit_trend.py --csv-files ${ANALYSISDIR}/mc_optimal_no_emb_norm_top_10000.csv \
                                ${ANALYSISDIR}/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                --dim --tasks la_${DATASET}_no_emb_norm

    echo 'Computing spearman correlation results'
    python scripts/analysis/get_correlation.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip_top_10000 eigen_overlap_top_10000 \
                                    --ds_metrics la_${DATASET}_no_emb_norm \
                                    --csv-file ${ANALYSISDIR}/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                    --out-file ${ANALYSISDIR}/w2v_cbow_correlation.csv \

    python scripts/analysis/get_correlation.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip_top_10000 eigen_overlap_top_10000 \
                                    --ds_metrics la_${DATASET}_no_emb_norm \
                                    --csv-file ${ANALYSISDIR}/mc_optimal_no_emb_norm_top_10000.csv \
                                    --out-file ${ANALYSISDIR}/mc_correlation.csv \

    echo 'Generate selection criterion results'
    python scripts/analysis/selection_criterion.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip_top_10000 eigen_overlap_top_10000 \
                                    --ds_metrics la_${DATASET}_no_emb_norm \
                                    --csv-file ${ANALYSISDIR}/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                    --acc-file ${ANALYSISDIR}/w2v_cbow_selection_error.csv \
                                    --rob-file ${ANALYSISDIR}/w2v_cbow_selection_robustness.csv

    python scripts/analysis/selection_criterion.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip_top_10000 eigen_overlap_top_10000 \
                                    --ds_metrics la_${DATASET}_no_emb_norm \
                                    --csv-file ${ANALYSISDIR}/mc_optimal_no_emb_norm_top_10000.csv \
                                    --acc-file ${ANALYSISDIR}/mc_selection_error.csv \
                                    --rob-file ${ANALYSISDIR}/mc_selection_robustness.csv

    echo 'Generate results for difference to the oracle (same memory budget)'
    python scripts/analysis/diff_to_oracle.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip_top_10000 eigen_overlap_top_10000 baseline_fp baseline_lp \
                                    --ds_metrics la_${DATASET}_no_emb_norm \
                                    --csv-file ${ANALYSISDIR}/w2v_cbow_optimal_no_emb_norm_top_10000.csv \
                                    --out-file ${ANALYSISDIR}/w2v_cbow_diff_to_oracle.csv

    python scripts/analysis/diff_to_oracle.py --emb_metrics anchor_eigen_overlap_3.0_top_10000 knn_top_10000_nquery_1000_nn_5 sem_disp_top_10000 pip_top_10000 eigen_overlap_top_10000 baseline_fp baseline_lp \
                                    --ds_metrics la_${DATASET}_no_emb_norm \
                                    --csv-file ${ANALYSISDIR}/mc_optimal_no_emb_norm_top_10000.csv \
                                    --out-file ${ANALYSISDIR}/mc_diff_to_oracle.csv
fi