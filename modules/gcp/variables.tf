variable "project_id" {
  description = "The project ID to deploy to"
  type        = string
  default = ""
}

variable "file_gateway_image" {
  type        = string
  description = "The Docker image for the file-gateway server."
  default = ""
}