#!/usr/bin/env python3
"""
Simple PocketBase admin management script for Railway deployments
Run this after deployment to create admin account
"""

import os
import sys
import time
import requests
import subprocess
from pathlib import Path

def wait_for_pocketbase(max_attempts=30):
    """Wait for PocketBase to be ready"""
    print("Waiting for PocketBase to start...")
    for i in range(max_attempts):
        try:
            response = requests.get("http://localhost:8090/api/health")
            if response.status_code == 200:
                print("✅ PocketBase is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(f"Attempt {i+1}/{max_attempts}...")
    return False

def create_admin_account(email, password):
    """Create admin account via PocketBase API"""
    try:
        response = requests.post(
            "http://localhost:8090/api/admins",
            json={
                "email": email,
                "password": password,
                "passwordConfirm": password
            }
        )
        
        if response.status_code == 200:
            print(f"✅ Admin account created: {email}")
            return True
        else:
            print(f"❌ Failed to create admin: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        return False

def main():
    """Main function"""
    # Get credentials from environment or use defaults
    admin_email = os.getenv("POCKETBASE_ADMIN_EMAIL", "admin@cbtapi.com")
    admin_password = os.getenv("POCKETBASE_ADMIN_PASSWORD", "SecureAdminPass123!")
    
    print("🚀 PocketBase Admin Setup")
    print("=" * 50)
    
    # Check if PocketBase is running
    if not wait_for_pocketbase():
        print("❌ PocketBase is not running. Start it first.")
        sys.exit(1)
    
    # Create admin account
    if create_admin_account(admin_email, admin_password):
        print("\n✅ Setup complete!")
        print(f"Admin Email: {admin_email}")
        print(f"Admin Password: {'*' * len(admin_password)}")
        print("\n⚠️  IMPORTANT: Save these credentials securely!")
        print("You can now use these to authenticate via the API")
    else:
        print("\n❌ Setup failed. Admin might already exist.")

if __name__ == "__main__":
    main()