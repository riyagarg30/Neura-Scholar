variable "suffix" {
  description = "Suffix for resource names (project ID)"
  type        = string
  nullable = false
  default = "project22"
}

variable "volume_id" {
    description = "The Volume ID to attach node1 to"
    type = string
    nullable = false
    default = "d0c861fe-5016-4154-929d-e90bf45bf6df"
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "ghost"
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
    "node2" = "192.168.1.12"
    "node3" = "192.168.1.13"
  }
}

