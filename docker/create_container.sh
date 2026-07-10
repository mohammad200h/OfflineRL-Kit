
sudo xhost +local:root


sudo docker run --runtime=nvidia -it --name offline_rl \
  -v $(pwd)/../../OfflineRL-Kit:/workspace/OfflineRL-Kit \
  --net=host  --privileged offline_rl
