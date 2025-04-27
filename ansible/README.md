# Ansible setup for the project

set floating ip addresses:

```sh
export FLOATING_IP_KVM_TACC=""
export FLOATING_IP_CHI_UC=""
export FLOATING_IP_CHI_TACC=""
```

```sh
docker run --rm -it --mount type=bind,source="$(pwd)",dst=/inventory \
    --mount type=bind,source="${HOME}/.ssh/id_mac",dst=/root/.ssh/id_rsa \ 
    -e FLOATING_IP_KVM_TACC=$FLOATING_IP_KVM_TACC \
    -e FLOATING_IP_CHI_UC=$FLOATING_IP_CHI_UC \
    -e FLOATING_IP_CHI_TACC=$FLOATING_IP_CHI_TACC \
    -e ANSIBLE_ROLES_PATH=roles \
    quay.io/kubespray/kubespray:v2.27.0 /bin/bash -c 'cp /inventory/ansible.cfg . && /bin/bash'
```

Run the following inside the container : 


```sh
ansible all -i /inventory/myclusters -m ping
```

```sh
ansible-playbook -i /inventory/myclusters /inventory/pre_k8s/pre_k8s_configure.yml
```

```sh
ansible-playbook -i /inventory/myclusters/kvm_tacc cluster.yml --become --become-user=root --check
ansible-playbook -i /inventory/myclusters/chi_uc cluster.yml --become --become-user=root --check
```
