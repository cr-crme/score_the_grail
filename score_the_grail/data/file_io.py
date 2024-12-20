from cryptography.fernet import Fernet


class FileEncryptor:
    def __init__(self, encryption_key: str):
        self._cipher = Fernet(encryption_key.encode())

    def encrypt_file(self, source_file_path: str, target_file_path: str):
        # Read the file content
        with open(source_file_path, "rb") as file:
            file_data = file.read()

        # Encrypt the file content
        encrypted_data = self._cipher.encrypt(file_data)

        # Write the file with encrypted content
        with open(target_file_path, "wb") as file:
            file.write(encrypted_data)

    @staticmethod
    def generate_key() -> str:
        """
        Generate a new encryption key. This should be done only once and the key should be securely stored.
        """
        return Fernet.generate_key().decode()


class FileReader:
    def __init__(self, encryption_key: bytes):
        self.cipher = Fernet(encryption_key)

    def read_encrypted_file(self, file_path: str) -> bytes:
        # Read the encrypted file content
        with open(file_path, "rb") as file:
            encrypted_data = file.read()

        # Decrypt the content
        decrypted_data = self.cipher.decrypt(encrypted_data)

        return decrypted_data
