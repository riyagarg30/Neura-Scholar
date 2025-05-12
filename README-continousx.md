# Project Infrastructure


![systemdiagram](./continousx.png)

# To provision and bootstrap the infrastructure:

- Using Openstack CLI to create volumes and a reservation lease at [./openstack/README.md](./openstack/README.md).

- Using Terraform to bring up the infrastructure at [./terraform/README.md](./terraform/README.md) :
    - 3x m1.large cpu nodes at KVM@TACC site with a private network.
    - 1x RTX 6000 GPU node at CHI@UC site.

- Using Ansible to bootstrap the nodes with kubernetes, ArgoCD applications and setup secrets at [./ansible/README.md](./ansible/README.md) :
    - Add docker auth file ( `.docker` ) to the cpu and gpu nodes.
    - Bootstrap K8s on _both_ the nodes with kube-spray.
    - Create K8 namespaces and add secrets, and configMaps.
    - Bringing up ArgoCD applications.
    - Create a ssh tunnel from the gpu node at CHI_UC to node1 at KVM_TACC in order to access the Ray Cluster endpoint and metrics at KVM_TACC.

- Using Argo Workflows to register workflows.

# Kubernetes deployments

- [mlflow helm chart](./kubernetes/mlflow)
- [test model serving helm chart](./kubernetes/serve_testing)
- [staging helm chart](./kubernetes/serve_staging)
- [canary helm chart](./kubernetes/serve_canary)
- [production helm chart](./kubernetes/serve_prod)



