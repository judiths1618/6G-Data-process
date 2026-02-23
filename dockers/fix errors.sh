# # Generate a valid Airflow Fernet key:

# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# export FERNET_KEY=6Zq8PpWflUoF8RCHvQB5NOYL9qnR1JV6FOYWsz84wbQ=

# sudo docker exec -it airflow airflow users create \
#     --username admin \
#     --firstname Admin \
#     --lastname User \
#     --role Admin \
#     --email admin@example.com \
#     --password admin

# sudo docker exec -it airflow airflow users reset-password \
#     --username admin \
#     --password admin2026HaHa

# 进入 airflow 容器执行
# sudo docker exec -it airflow python3 -c "
# import boto3
# s3 = boto3.client('s3', endpoint_url='http://seaweed-s3:8333')
# bucket = 'airflow-bucket' # 替换为你的真实 bucket 名
# response = s3.list_objects_v2(Bucket=bucket)
# for obj in response.get('Contents', []):
#     print(f'S3 Key: {obj['Key']}')
# "

# 方法 1：使用 docker exec 直接修改
sudo docker exec wavestitchplus-gpu bash -c "sed -i 's|get_save_dir(args.prepared_dir)|os.path.join(args.prepared_dir, \"saved_model\")|g' /app/WaveStitchPlus_app/synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py"

# 验证修改
sudo docker exec wavestitchplus-gpu grep -A 2 "saved_dir = " /app/WaveStitchPlus_app/synthesis_wavestitchPlus_pipeline_strided_preconditioning_customdata.py | head -5

# 重新运行测试
bash test\ docker.sh