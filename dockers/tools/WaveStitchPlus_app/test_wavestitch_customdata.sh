# python3 synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py \
#   -d custom_csv \
#   -input_csv /path/to/your.csv \
#   -prepared_dir ./work/prepared \
#   -synth_mask C \
#   -window_size 32 \
#   -stride 1 \
#   -out_csv ./work/generated/full_imputed.csv

python3 synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py \
  -d custom_csv \
  -prepared_dir ./work/EUR/prepared_amf \
  -synth_mask gap_imputation \
  -n_trials 3 \
  -out_csv ./work/EUR/generated_amf/wavestitch_full_imputed.csv


# python3 synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py \
#   -d custom_csv \
#   -prepared_dir ./work/EUR/prepared_python \
#   -synth_mask gap_imputation \
#   -n_trials 3 \
#   -out_csv ./work/EUR/generated_python/wavestitch_full_imputed.csv


# python3 synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py \
#   -d custom_csv \
#   -prepared_dir ./work/EUR/prepared_golang \
#   -synth_mask gap_imputation \
#   -n_trials 3 \
#   -out_csv ./work/EUR/generated_golang/wavestitch_full_imputed.csv

# python3 synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py \
#   -d custom_csv \
#   -prepared_dir ./work/EUR/prepared_rabbitmq \
#   -synth_mask gap_imputation \
#   -n_trials 3 \
#   -out_csv ./work/EUR/generated_rabbitmq/wavestitch_full_imputed.csv
  
# python3 custom_pipeline/eval.py \
#   -prepared_dir ./work/EUR/prepared_amf \
#   -pred_csv ./generated/custom_csv/gap_imputation/full_imputed_stride_1_trial_0.csv