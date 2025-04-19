# Dockerfile
FROM google/cloud-sdk:slim

# install bash (already there), curl & unzip for rclone download
RUN apt-get update \
 && apt-get install -y curl unzip bash \
 && rm -rf /var/lib/apt/lists/*

# download & install rclone
RUN curl -Of https://downloads.rclone.org/rclone-current-linux-amd64.zip \
 && unzip rclone-current-linux-amd64.zip \
 && cp rclone-*-linux-amd64/rclone /usr/bin/ \
 && chmod 755 /usr/bin/rclone \
 && rm -rf rclone-*-linux-amd64*

# gsutil is already present via google/cloud-sdk
