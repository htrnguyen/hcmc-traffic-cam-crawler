#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from pathlib import Path
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow


def check_drive_folders():
    """Check folder structure on Google Drive"""

    # Load config
    cfg_path = "config.yaml"
    if not Path(cfg_path).exists():
        print(f"Missing config file: {cfg_path}")
        return

    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    # Setup Drive connection
    scopes = ["https://www.googleapis.com/auth/drive"]
    creds = None

    token_json = cfg["oauth_token_file"]
    if os.path.exists(token_json):
        creds = Credentials.from_authorized_user_file(token_json, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                cfg["oauth_client_secret"], scopes=scopes
            )
            creds = flow.run_local_server(port=0)
        Path(token_json).write_text(creds.to_json(), encoding="utf-8")

    session = AuthorizedSession(creds)
    root_folder_id = cfg["drive_folder_id"]

    # Get all folders in root
    q = (
        f"mimeType='application/vnd.google-apps.folder' and trashed=false and "
        f"'{root_folder_id}' in parents"
    )

    r = session.get(
        "https://www.googleapis.com/drive/v3/files",
        params={
            "q": q,
            "fields": "files(id,name)",
            "pageSize": 1000,
            "orderBy": "name",
        },
    )
    r.raise_for_status()
    folders = r.json().get("files", [])

    print(f"Found {len(folders)} folders in Google Drive:")
    print("-" * 60)

    # Categorize folders
    numbered_folders = []
    unnumbered_folders = []

    for folder in folders:
        name = folder["name"]
        if name.startswith("cam_") and "_" in name[4:]:
            numbered_folders.append(name)
        else:
            unnumbered_folders.append(name)

    # Show numbered folders (correct format)
    if numbered_folders:
        print(f"✓ Correctly formatted folders ({len(numbered_folders)}):")
        for name in sorted(numbered_folders)[:10]:  # Show first 10
            print(f"  {name}")
        if len(numbered_folders) > 10:
            print(f"  ... and {len(numbered_folders) - 10} more")
        print()

    # Show unnumbered folders (old format)
    if unnumbered_folders:
        print(f"⚠ Old format folders without numbers ({len(unnumbered_folders)}):")
        for name in sorted(unnumbered_folders)[:10]:  # Show first 10
            print(f"  {name}")
        if len(unnumbered_folders) > 10:
            print(f"  ... and {len(unnumbered_folders) - 10} more")
        print()

    # Summary
    print("Summary:")
    print(f"  Total folders: {len(folders)}")
    print(f"  Correct format (cam_XXX_id): {len(numbered_folders)}")
    print(f"  Old format (id only): {len(unnumbered_folders)}")

    if unnumbered_folders:
        print("\n⚠ Some folders still use old format without numbers.")
        print("  New uploads will create correct numbered folders.")
    else:
        print("\n✓ All folders use correct numbered format!")


if __name__ == "__main__":
    check_drive_folders()
