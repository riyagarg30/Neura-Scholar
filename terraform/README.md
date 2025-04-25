# Terraform infra code

Use the `clouds.yaml.format` file as a template for `clouds.yaml` file.

```sh
cp clouds.yaml.format ~/.config/openstack/clouds.yaml
```

Then update the credentials in the `clouds.yaml` file:

```
      application_credential_id: ""
      application_credential_secret: ""
```

```sh
source create_reservation.sh
```

```sh
create_reservation
```

This create a lease and prints the reservation ID; the script waits for the status to turn to `ACTIVE` ...

```sh
cd chi_us; terraform init; terraform plan
```

```sh
terraform apply
```

Copy and paste the reservations ID.

---

When DONE WITH, RELEASE RESOURCES BY :
```sh
delete reservation
```
