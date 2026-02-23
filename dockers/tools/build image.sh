
# clean unused docker images
sudo docker system prune -a -f

# build
sudo docker build --no-cache -f Dockerfile.wavestitchplus-gpu -t wavestitchplus-gpu:latest .