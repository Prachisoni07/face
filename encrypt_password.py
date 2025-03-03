from cryptography.fernet import Fernet

# Generate encryption key (Run once and save it securely)
key = Fernet.generate_key()
print(f"Save this key securely: {key.decode()}")

# Encrypt the password
cipher_suite = Fernet(key)
encrypted_password = cipher_suite.encrypt(b"saritasoni@7")
print(f"Encrypted Password: {encrypted_password.decode()}")