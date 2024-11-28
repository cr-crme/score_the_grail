import configparser
import os
import shutil
import subprocess

# Clear any previous build files
if os.path.isdir("dist"):
    shutil.rmtree("dist")
if os.path.isdir("build"):
    shutil.rmtree("build")
if os.path.isfile("gui.spec"):
    os.remove("gui.spec")

# Read setup.cfg
config = configparser.ConfigParser()
config.read("../setup.cfg")

# Get dependencies from install_requires
dependencies = config["options"]["install_requires"].strip().split("\n")

# Create hidden-import arguments for PyInstaller
hidden_imports = [f"--hidden-import={dep.split('=')[0].strip()}" for dep in dependencies]

# Find all the ".txt" files in the library and add them to add-data
add_data = []
for root, _, files in os.walk("../score_the_grail"):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            dest_path = root[3:]  # Remove the "../" from the path
            add_data.append(f"--add-data={file_path}:{dest_path}")


# Build the PyInstaller command
cmd = [
    "pyinstaller",
    "--onefile",
    "--windowed",
    *hidden_imports,
    *add_data,
    "gui.py",
]

# Run PyInstaller with the dependencies
subprocess.run(cmd)
