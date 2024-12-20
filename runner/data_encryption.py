import os

from score_the_grail.data.file_io import FileEncryptor, FileReader


def main():
    # Get the decryption key from the environment variable
    key = os.environ["NORMATIVE_GRAIL_DATA_KEY"] if "NORMATIVE_GRAIL_DATA_KEY" in os.environ else None
    if key is None:
        # If None is found, create one and give some instructions to the user to store it securely
        print(
            "Encryption key not found. Please store the following key securely and set the environment variable\n"
            "\n"
            f'NORMATIVE_GRAIL_DATA_KEY="{str(FileEncryptor.generate_key())}"\n'
        )
        # Wait for a key press
        input("Press any key when ready...\n")
    encryptor = FileEncryptor(encryption_key=key)

    # Files to encrypt (These files were then deleted to avoid storing sensitive data)
    files = [
        "../score_the_grail/data/normative_normal.csv",
        "../score_the_grail/data/normative_normal_std.csv",
        "../score_the_grail/data/normative_crouchgait.txt",
    ]

    for file in files:
        encryptor.encrypt_file(file, file + ".encrypted")
        print(f"Encrypted {file} to {file}.encrypted")

    # Decrypt the files to make sure the file is properly encrypted
    decryptor = FileReader(encryption_key=key)
    for file in files:
        decrypted_data = decryptor.read_encrypted_file(file + ".encrypted")
        # Compare the decrypted data with the original data
        with open(file, "rb") as original_file:
            original_data = original_file.read()
            assert decrypted_data == original_data, f"Decrypted data does not match the original data for {file}"


if __name__ == "__main__":
    main()
