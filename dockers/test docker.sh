# 直接运行
# sudo docker run --rm --gpus all \
#     -e S3_ENDPOINT=http://host.docker.internal:8333 \
#     -e S3_ACCESS_KEY=anykey \
#     -e S3_SECRET_KEY=anysecret \
#     -e S3_BUCKET=airflow-bucket \
#     wavestitchplus-gpu:latest \
#     python /app/run_pipeline.py \
#     --input-s3-key test/amfperformance.csv \
#     --output-s3-prefix wavestitch/test001 \
#     --epochs 100

# build 
# sudo docker build -f Dockerfile.wavestitchplus-gpu -t wavestitchplus-gpu:latest .

# 查看 SeaweedFS 所在的网络
sudo docker network ls
# 查看网络中的容器
sudo docker network inspect dockers_airflow_net --format '{{range .Containers}}{{.Name}} {{end}}'

# 运行时加入相同网络
sudo docker run --rm --gpus all \
    --network dockers_airflow_net \
    -e S3_ENDPOINT=http://seaweed-s3:8333 \
    -e S3_ACCESS_KEY=anykey \
    -e S3_SECRET_KEY=anysecret \
    -e S3_BUCKET=airflow-bucket \
    wavestitchplus-gpu \
    python /app/run_pipeline.py \
    --mode full \
    --dataset-name amf-performance \
    --input-s3-key test/amf-performance.csv
    # --output-s3-prefix cleaned/wavestitchPlus/test001

# clean unused docker images
# sudo docker system prune -a -f