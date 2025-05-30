resource "openstack_networking_network_v2" "private_net" {
  name = "private-net-mlops-${var.suffix}"
  port_security_enabled = false
}

resource "openstack_networking_subnet_v2" "private_subnet" {
  name       = "private-subnet-mlops-${var.suffix}"
  network_id = openstack_networking_network_v2.private_net.id
  cidr       = "192.168.1.0/24"
  no_gateway = true
}

resource "openstack_networking_port_v2" "private_net_ports" {
  for_each              = var.nodes
  name                  = "port-${each.key}-mlops-${var.suffix}"
  network_id            = openstack_networking_network_v2.private_net.id
  port_security_enabled = false

  fixed_ip {
    subnet_id  = openstack_networking_subnet_v2.private_subnet.id
    ip_address = each.value
  }
}

resource "openstack_networking_port_v2" "sharednet1_ports" {
  for_each   = var.nodes
    name       = "sharednet1-${each.key}-mlops-${var.suffix}"
    network_id = data.openstack_networking_network_v2.sharednet1.id
    security_group_ids = [
      data.openstack_networking_secgroup_v2.allow_ssh.id,
      data.openstack_networking_secgroup_v2.allow_http_80.id,
      data.openstack_networking_secgroup_v2.allow_30000_32767.id,
      data.openstack_networking_secgroup_v2.allow_services_for_project22.id
    ]
}

# Create the instance, referencing the baremetal flavor, and scheduler hint
#resource "openstack_compute_instance_v2" "gpu_node" {
#
#  for_each   = var.nodes
#
#  name = "${each.key}-mlops-${var.suffix}"
#  image_name = "CC-Ubuntu24.04-CUDA"
#  flavor_name = "baremetal"
#  key_pair = "${var.key}"
#
#  network {
#    port = openstack_networking_port_v2.sharednet1_ports[each.key].id
#  }
#
#  network {
#    port = openstack_networking_port_v2.private_net_ports[each.key].id
#  }
#
#  scheduler_hints {
#    additional_properties = {
#        "reservation" = "${var.reservation_id}"
#    }
#  }
#
#  user_data = <<-EOF
#    #! /bin/bash
#    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
#    su cc -c /usr/local/bin/cc-load-public-keys
#  EOF
#  
#}

resource "openstack_compute_instance_v2" "nodes" {
  for_each = var.nodes

  name        = "${each.key}-mlops-${var.suffix}"
  image_name  = "CC-Ubuntu24.04"
  flavor_name = "m1.large"
  key_pair    = "${var.key}"

  network {
    port = openstack_networking_port_v2.sharednet1_ports[each.key].id
  }

  network {
    port = openstack_networking_port_v2.private_net_ports[each.key].id
  }

  user_data = <<-EOF
    #! /bin/bash
    sudo echo "127.0.1.1 ${each.key}-mlops-${var.suffix}" >> /etc/hosts
    su cc -c /usr/local/bin/cc-load-public-keys
  EOF

}

resource "openstack_compute_volume_attach_v2" "attach_volume" {
    volume_id = "${var.volume_id}"
    instance_id = openstack_compute_instance_v2.nodes["node1"].id
}

resource "openstack_networking_floatingip_v2" "floating_ip" {
  pool        = "public"
  description = "MLOps IP for ${var.suffix}"
  port_id     = openstack_networking_port_v2.sharednet1_ports["node1"].id
}

