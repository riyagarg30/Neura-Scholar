curl https://rclone.org/install.sh | sudo bash
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
mkdir -p ~/.config/rclone
cp ./rg_rclone.conf ~/.config/rclone/rclone.conf
export RCLONE_CONTAINER=object-persist-project-22
