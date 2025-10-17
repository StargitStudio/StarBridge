import os
import secrets
from pathlib import Path

def generate_api_key(length=32):
    """Generate a secure random API key."""
    return secrets.token_hex(length)

def ensure_gitignore_includes_env():
    """Ensure .env is in .gitignore to prevent accidental commits."""
    gitignore_path = Path('.gitignore')
    env_entry = '.env\n'

    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
        if env_entry not in content:
            with open(gitignore_path, 'a') as f:
                f.write(env_entry)
            print("Added '.env' to .gitignore to prevent accidental commits.")
    else:
        with open(gitignore_path, 'w') as f:
            f.write(env_entry)
        print("Created .gitignore with '.env' entry.")

def setup_starbridge():
    """Set up StarBridge by creating a .env file with a secure API key."""
    env_path = Path('.env')

    if env_path.exists():
        print("A .env file already exists. To regenerate the API key, delete .env and rerun this script.")
        return

    # Generate a secure API key
    api_key = generate_api_key()
    
    # Write to .env
    with open(env_path, 'w') as f:
        f.write(f"STARBRIDGE_API_KEY={api_key}\n")
    
    # Ensure .env is in .gitignore
    ensure_gitignore_includes_env()

    # Print instructions
    print("\nStarBridge setup complete!")
    print(f"Your API key is: {api_key}")
    print("⚠️  Store this key securely! It will not be shown again.")
    print("Next steps:")
    print("1. Copy 'example-settings.json' to 'settings.json' and configure your repository paths and SSL settings.")
    print("2. Ensure your SSL certificates are correctly set up as specified in settings.json.")
    print("3. Run the application: `python app.py` or use the starbridge.service file.")
    print("4. Use the API key in requests (e.g., set 'x-api-key' header to the value above).")

if __name__ == '__main__':
    print("Starting StarBridge setup...")
    setup_starbridge()
