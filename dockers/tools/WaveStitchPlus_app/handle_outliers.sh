# # 从 meta.json 读取 target_cols
# python ./helper/outlier_removal.py \
#     -i ./work/EUR/generated_python/wavestitch_full_imputed.csv \
#     -o ./work/EUR/generated_python/wavestitch_full_imputed_cleaned.csv \
#     -prepared_dir ./work/prepared_python

# python ./helper/outlier_removal.py \
#     -i ./work/EUR/generated_python/wavestitchPlus_full_imputed.csv \
#     -o ./work/EUR/generated_python/wavestitchPlus_full_imputed_cleaned.csv \
#     -prepared_dir ./work/prepared_python


python ./helper/outlier_removal.py \
    -i ./work/EUR/generated_amf/wavestitch_full_imputed.csv \
    -o ./work/EUR/generated_amf/wavestitch_full_imputed_cleaned.csv \
    -prepared_dir ./work/prepared_amf

python ./helper/outlier_removal.py \
    -i ./work/EUR/generated_amf/wavestitchPlus_full_imputed.csv \
    -o ./work/EUR/generated_amf/wavestitchPlus_full_imputed_cleaned.csv \
    -prepared_dir ./work/prepared_amf

# python ./helper/outlier_removal.py \
#     -i ./work/EUR/generated_golang/wavestitch_full_imputed.csv \
#     -o ./work/EUR/generated_golang/wavestitch_full_imputed_cleaned.csv \
#     -prepared_dir ./work/prepared_golang

# python ./helper/outlier_removal.py \
#     -i ./work/EUR/generated_golang/wavestitchPlus_full_imputed.csv \
#     -o ./work/EUR/generated_golang/wavestitchPlus_full_imputed_cleaned.csv \
#     -prepared_dir ./work/prepared_golang


# python ./helper/outlier_removal.py \
#     -i ./work/EUR/generated_rabbitmq/wavestitch_full_imputed.csv \
#     -o ./work/EUR/generated_rabbitmq/wavestitch_full_imputed_cleaned.csv \
#     -prepared_dir ./work/prepared_rabbitmq

# python ./helper/outlier_removal.py \
#     -i ./work/EUR/generated_rabbitmq/wavestitchPlus_full_imputed.csv \
#     -o ./work/EUR/generated_rabbitmq/wavestitchPlus_full_imputed_cleaned.csv \
#     -prepared_dir ./work/prepared_rabbitmq