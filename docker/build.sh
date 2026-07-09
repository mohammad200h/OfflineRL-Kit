USER=mamad
USER_ID=1000
GROUP_ID=1000

echo $USER
echo $USER_ID
echo $GROUP_ID

sudo docker build \
    --network=host \
    --build-arg UNAME=$USER \
    --build-arg UID=$USER_ID \
    --build-arg GID=$GROUP_ID \
    -f Dockerfile \
    -t offline_rl ..

