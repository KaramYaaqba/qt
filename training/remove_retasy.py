#!/usr/bin/env python3
"""Remove all retasy entries from manifest files and delete audio files."""

import json
import os
from pathlib import Path

def remove_retasy_from_manifests():
    data_dir = Path("/Users/kyaaqba/quran-recitation-checker/training/data")
    manifests = ["manifest_train.json", "manifest_test.json", "manifest_dev.json"]
    audio_dir = data_dir / "audio"
    
    for manifest_file in manifests:
        manifest_path = data_dir / manifest_file
        print(f"\nProcessing {manifest_file}...")
        
        # Read manifest
        entries = []
        retasy_files = []
        
        with open(manifest_path, 'r') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if "retasy" in entry["audio_filepath"]:
                        retasy_files.append(entry["audio_filepath"])
                    else:
                        entries.append(entry)
        
        print(f"  Found {len(retasy_files)} retasy entries")
        print(f"  Keeping {len(entries)} entries")
        
        # Write filtered manifest
        with open(manifest_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        # Delete audio files
        deleted_count = 0
        for audio_file in retasy_files:
            audio_path = Path(audio_file)
            if audio_path.exists():
                try:
                    audio_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"  Failed to delete {audio_path}: {e}")
        
        print(f"  Deleted {deleted_count} audio files")

if __name__ == "__main__":
    remove_retasy_from_manifests()
    print("\n✅ Done!")
