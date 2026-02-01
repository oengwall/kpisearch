import hashlib
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from kpisearch import DATA_DIR

ADMIN_PASSWORD_FILE = DATA_DIR / 'admin_password.txt'
DEFAULT_PASSWORD = 'change_this_now_really!'

security = HTTPBasic()


def hash_password(password: str, salt: bytes | None = None) -> str:
    """Hash a password with PBKDF2-SHA256. Returns 'salt:hash' format."""
    if salt is None:
        salt = secrets.token_bytes(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations=100000)
    return f'{salt.hex()}:{pw_hash.hex()}'


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored 'salt:hash' string."""
    try:
        salt_hex, hash_hex = stored.split(':')
        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)
        actual_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations=100000)
        return secrets.compare_digest(actual_hash, expected_hash)
    except (ValueError, AttributeError):
        return False


def get_stored_password_hash() -> str:
    """Get the stored admin password hash, creating default if needed."""
    if not ADMIN_PASSWORD_FILE.exists():
        set_admin_password(DEFAULT_PASSWORD)
    with open(ADMIN_PASSWORD_FILE) as f:
        return f.read().strip()


def set_admin_password(password: str) -> None:
    """Set the admin password."""
    ADMIN_PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ADMIN_PASSWORD_FILE, 'w') as f:
        f.write(hash_password(password))


def verify_admin_password(password: str) -> bool:
    """Check if password matches the admin password."""
    stored_hash = get_stored_password_hash()
    return verify_password(password, stored_hash)


def is_default_password() -> bool:
    """Check if the current password is the default."""
    stored_hash = get_stored_password_hash()
    return verify_password(DEFAULT_PASSWORD, stored_hash)


def get_current_admin(credentials: Annotated[HTTPBasicCredentials, Depends(security)]) -> str:
    """Dependency to verify admin authentication.

    Username is ignored, only password matters.
    Raises HTTPException 401 if not authenticated.
    """
    if not verify_admin_password(credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Invalid password',
            headers={'WWW-Authenticate': 'Basic'},
        )
    return 'admin'


CurrentAdmin = Annotated[str, Depends(get_current_admin)]


def main() -> None:
    """CLI to manage admin password."""
    import sys

    if len(sys.argv) < 2:
        print('Usage:')
        print('  python -m kpisearch.auth set <password>  - Set admin password')
        print('  python -m kpisearch.auth reset           - Reset to default password')
        sys.exit(1)

    command = sys.argv[1]

    if command == 'set':
        if len(sys.argv) != 3:
            print('Usage: python -m kpisearch.auth set <password>')
            sys.exit(1)
        password = sys.argv[2]
        set_admin_password(password)
        print('Admin password updated.')

    elif command == 'reset':
        set_admin_password(DEFAULT_PASSWORD)
        print(f'Admin password reset to default: {DEFAULT_PASSWORD}')

    else:
        print(f'Unknown command: {command}')
        sys.exit(1)


if __name__ == '__main__':
    main()
