output "gateway_uri" {
  value = google_cloud_run_v2_service.file-gateway.uri
  description = "The URI of the file gateway service."
}