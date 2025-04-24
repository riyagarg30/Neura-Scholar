
# Provision resources on kvm@tacc

```sh
source create_instance.sh
```

```sh
# create_instance "<node_name>" "<fixed_ip_address>"
create_instance node1 192.168.1.10
```

```sh
openstack server add floating ip node1 $FLOATING_IP
```

```sh
create_instance node2 192.168.1.11
```

```sh
openstack server show | grep project22
```

```sh
ssh -A cc@$FLOATING_IP "bash -s" < setup-kubespray.sh "192.168.1.10" "192.168.1.11"
```

```sh
ssh -A cc@$FLOATING_IP
```

```sh
cc@node1$ kubectl get pods
```
