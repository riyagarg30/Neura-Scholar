# Ansible setup for the project

set floating ip addresses:

```sh
export FLOATING_IP_KVM_TACC=""
export FLOATING_IP_CHI_UC=""
export FLOATING_IP_CHI_TACC=""
```

Add the `.docker/config.json` auth file in this directory (./ansible/.docker/config.json).

```sh
docker run --rm -it --mount type=bind,source="$(pwd)",dst=/inventory \
    --mount type=bind,source="${HOME}/.ssh/id_ghost",dst=/root/.ssh/id_rsa \ 
    -e FLOATING_IP_KVM_TACC=$FLOATING_IP_KVM_TACC \
    -e FLOATING_IP_CHI_UC=$FLOATING_IP_CHI_UC \
    -e FLOATING_IP_CHI_TACC=$FLOATING_IP_CHI_TACC \
    -e ANSIBLE_ROLES_PATH=roles \
    -e ANSIBLE_CONFIG=/inventory/ansible.cfg \
    quay.io/kubespray/kubespray:v2.27.0 /bin/bash -c "source /inventory/scripts.sh; exec /bin/bash"
```

Run the following inside the container : 

```sh
./inventory/scripts.sh
```

