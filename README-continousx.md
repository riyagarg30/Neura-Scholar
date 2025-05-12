# Project Infrastructure


![systemdiagram](./continousx.png)

# (1) To provision and bootstrap the infrastructure:

- Using Openstack CLI to create volumes and a reservation lease at ./openstack/README.md .

- Using Terraform to bring up the infrastructure at ./terraform/README.md :
    - 3x m1.large cpu nodes at KVM@TACC site with a private network.
    - 1x RTX 6000 GPU node at CHI@UC site.

- Using Ansible to bootstrap the machine with kubernetes, ArgoCD applications and setup secrets at ./ansible/README.md :
    - Add docker auth file ( `.docker` ) to the cpu and gpu nodes.
    - Bootstrap K8s on _both_ the nodes with kube-spray.
    - Create K8 namespaces and add secrets, and configMaps.
    - Bring up ArgoCD applications.
    - Create a ssh tunnel from the gpu node at CHI_UC to node1 at KVM_TACC in order to access the Ray Cluster endpoint and metrics at KVM_TACC.

- Using Argo Workflows to register workflows.

