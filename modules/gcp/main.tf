terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.58.0"
    }
  }
}

# GOOGLE PROVIDER

provider "google" {
  credentials = file("auth.json")
  project = var.project_id
  region  = "europe-west1"
}

# Bucket

resource "google_storage_bucket" "bucket" {
  name          = "fileexchange-bucket"
  location      = "europe-west1"
  force_destroy = true
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
}

# GCP Policies

data "google_iam_policy" "bucket_public" {
  binding {
    role = "roles/storage.objectViewer"
    members = [
      "allUsers",
    ]
  }
}

resource "google_storage_bucket_iam_policy" "bucket_policy" {
  bucket      = google_storage_bucket.bucket.name
  policy_data = data.google_iam_policy.bucket_public.policy_data
}
