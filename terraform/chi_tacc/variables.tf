variable "suffix" {
  description = "Suffix for resource names (project ID)"
  type        = string
  nullable = false
  default = "project22"
}

variable "reservation_id" {
    description = "Reservation ID"
    type = string
    nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "ghost"
}

variable "nodes" {
  type = map(string)
  default = {
    "gpu_node" = "192.168.1.10"
#    "node2" = "192.168.1.12"
#    "node3" = "192.168.1.13"
  }
}

