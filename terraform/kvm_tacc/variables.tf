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
    default = "96be5163-ab12-4477-9180-558cedf9f772"
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

