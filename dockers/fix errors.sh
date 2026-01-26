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
sudo docker exec -it airflow python3 -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://seaweed-s3:8333')
bucket = 'airflow-bucket' # 替换为你的真实 bucket 名
response = s3.list_objects_v2(Bucket=bucket)
for obj in response.get('Contents', []):
    print(f'S3 Key: {obj['Key']}')
"

