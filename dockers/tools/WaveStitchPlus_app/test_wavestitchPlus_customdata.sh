# python3 synthesis_wavestitch_pipeline_strided_preconditioning_customdata.py \
#   -d custom_csv \
#   -prepared_dir ./work/EUR/prepared_amf \
#   -synth_mask gap_imputation \
#   -n_trials 1 \
#   -out_csv ./work/EUR/generated_amf/full_imputed.csv

# 使用 EM 训练的模型推理
python synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py \
    -d custom_csv \
    -prepared_dir ./work/EUR/prepared_amf \
    -out_csv ./work/EUR/generated_amf/wavestitchPlus_full_imputed.csv \
    -model_type em \
    -guidance_scale 0.1 \
    -n_trials 3

# python synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py \
#     -d custom_csv \
#     -prepared_dir ./work/EUR/prepared_python \
#     -out_csv ./work/EUR/generated_python/wavestitchPlus_full_imputed.csv \
#     -model_type em \
#     -guidance_scale 0.1 \
#     -n_trials 3

# python synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py \
#     -d custom_csv \
#     -prepared_dir ./work/EUR/prepared_golang \
#     -out_csv ./work/EUR/generated_golang/wavestitchPlus_full_imputed.csv \
#     -model_type em \
#     -guidance_scale 0.1 \
#     -n_trials 3

# python synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py \
#     -d custom_csv \
#     -prepared_dir ./work/EUR/prepared_rabbitmq \
#     -out_csv ./work/EUR/generated_rabbitmq/wavestitchPlus_full_imputed.csv \
#     -model_type em \
#     -guidance_scale 0.1 \
#     -n_trials 3
    

# 启用 RePaint 增强（可能更好但更慢）
# python synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py \
#     -d custom_csv \
#     -prepared_dir ./work/EUR/prepared_python \
#     -use_repaint \
#     -repaint_steps 3 \
#     -guidance_scale 0.15