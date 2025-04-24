#!/bin/bash

declare -a IPS=("$@"); 

[ -f ~/.ssh/id_ed25519 ] && rm ~/.ssh/id_ed25519
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -q -N ""

for IP in "${IPS[@]}"; do
    printf "NODE IP ADDRESS : %s\n" "$IP"
    ssh-copy-id -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ~/.ssh/id_ed25519.pub cc@$IP;
    ssh cc@$IP "sudo service firewalld stop"
done


git clone --branch release-2.26 https://github.com/kubernetes-sigs/kubespray
sudo apt update; sudo apt -y install virtualenv
virtualenv -p python3 myenv

source myenv/bin/activate;  cd kubespray;   pip3 install -r requirements.txt; pip3 install ruamel.yaml;

cd; mv kubespray/inventory/sample kubespray/inventory/mycluster;
sed -i "s/container_manager: containerd/container_manager: docker/" kubespray/inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml;
sed -i "s/metrics_server_enabled: false/metrics_server_enabled: true/" kubespray/inventory/mycluster/group_vars/k8s_cluster/addons.yml;

cd; source myenv/bin/activate;  cd kubespray;  
CONFIG_FILE=inventory/mycluster/hosts.yaml python3 contrib/inventory_builder/inventory.py "${IPS[@]}";

cat ~/kubespray/inventory/mycluster/hosts.yaml

cd; source myenv/bin/activate; cd kubespray; ansible-playbook -i inventory/mycluster/hosts.yaml  --become --become-user=root cluster.yml

cd; sudo cp -R /root/.kube /home/cc/.kube; sudo chown -R cc /home/cc/.kube; sudo chgrp -R cc /home/cc/.kube
