import boto3
import hashlib
import botocore
from pathlib import Path

def download_s3_folder(bucket_name, s3_prefix, local_dir, profile="receipt-analytics", area_filter=None, sampling_rate=1.0):
  """
  Downloads files from S3 using a forced profile session.
  """
  try:
    # Create a session explicitly ignoring environment variables if they are broken
    session = boto3.Session(profile_name=profile)
    s3 = session.client('s3')
    
    # Test connection immediately
    s3.list_buckets()
  except botocore.exceptions.ClientError as e:
    if "InvalidAccessKeyId" in str(e):
      print("❌ The Access Key ending in ...TLCX is REJECTED by AWS.")
      print("Please generate a new key in IAM and run: aws configure --profile receipt-analytics")
      return
    raise e
  except botocore.exceptions.ProfileNotFound:
    print(f"❌ Profile '{profile}' not found in ~/.aws/credentials")
    return

  local_dir = Path(local_dir)
  paginator = s3.get_paginator('list_objects_v2')
  
  print(f"📡 Scanning S3: s3://{bucket_name}/{s3_prefix}")
  
  download_count = 0
  skip_count = 0

  for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
    for obj in page.get('Contents', []):
      key = obj['Key']
      
      # Filters
      if area_filter and f"area={area_filter}" not in key:
          continue

      # Deterministic Sampling
      hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
      if hash_val >= (sampling_rate * 100):
          skip_count += 1
          continue
          
      relative_path = key.replace(s3_prefix, "").lstrip("/")
      local_file_path = local_dir / relative_path
      local_file_path.parent.mkdir(parents=True, exist_ok=True)
      
      print(f"📥 [{hash_val}] Downloading: {relative_path}")
      s3.download_file(bucket_name, key, str(local_file_path))
      download_count += 1

  print(f"\n✅ Summary: {download_count} files downloaded, {skip_count} skipped.")

if __name__ == "__main__":
  download_s3_folder(
    bucket_name="ibc-receipts-processed",
    s3_prefix="processed/normalized/venduto/",
    local_dir="./data/raw_normalized/venduto/",
    profile="receipt-analytics",
    area_filter="382",
    sampling_rate=0.1
  )