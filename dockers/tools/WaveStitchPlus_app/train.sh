python3 train_wavestitch_customdata.py \
    -d 'custom_csv' \
    -input_csv "/home/Yuandou/Desktop/projects/6G-Data-process/6GDALI_Datasets/EUR/6907619/amf-performance.csv" \
    -prepared_dir ./work/EUR/prepared_amf \
    -lr 1e-3 \
    -stride 1 \
    -epochs 500 \
    -normalize True 2>&1 | tee training.log

# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "../datasets/EUR/golang-web-server-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_golang \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log


# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "../datasets/EUR/python-web-server-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_python \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log  

# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "../datasets/EUR/rabbitmq-performance.csv" \
#     -prepared_dir ./work/EUR/prepared_rabbitmq \
#     -lr 1e-3 \
#     -stride 1 \
#     -epochs 500 \
#     -normalize True 2>&1 | tee training.log  

# python3 train_wavestitch_customdata.py \
#     -d 'custom_csv' \
#     -input_csv "./datasets/DeepSense/scenario33.csv" \
#     -lr 1e-3 \
#     -stride 4 \
#     -epochs 300 \
#     -normalize True 2>&1 | tee training.log