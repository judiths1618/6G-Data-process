# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "./datasets/EUR/amf-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_amf \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log
# 新增：DiffPuter EM 模式
python train_wavestitchPlus_customdata.py \
    -d custom_csv \
    -input_csv "/home/Yuandou/Desktop/projects/6G-Data-process/6GDALI_Datasets/EUR/6907619/amf-performance.csv" \
    -prepared_dir ./work/EUR/prepared_amf \
    -use_em -em_iterations 5 -epochs_per_em 200


# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "./datasets/EUR/golang-web-server-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_golang \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log

# 新增：DiffPuter EM 模式
# python train_wavestitchPlus_customdata.py \
#     -d custom_csv \
#     -input_csv "../datasets/EUR/golang-web-server-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_golang \
#     -use_em -em_iterations 5 -epochs_per_em 200


# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "./datasets/EUR/python-web-server-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_python \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log  

# 新增：DiffPuter EM 模式
# python train_wavestitchPlus_customdata.py \
#     -d custom_csv \
#     -input_csv "../datasets/EUR/python-web-server-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_python \
#     -use_em \
    # -em_iterations 5  -epochs_per_em 200 
    # \
    # -batch_size 128 \
    # -e_step_batch_size 16 \
    # -ddim_steps 50

# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "./datasets/EUR/rabbitmq-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_rabbitmq \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log  

# 新增：DiffPuter EM 模式
# python train_wavestitchPlus_customdata.py \
#     -d custom_csv \
#     -input_csv "../datasets/EUR/rabbitmq-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_rabbitmq \
#     -use_em -em_iterations 5 -epochs_per_em 200