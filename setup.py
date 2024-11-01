from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

class CustomInstallCommand(install):
    """Customized install command to run post-install commands."""
    def run(self):
        # Run the standard install process
        install.run(self)

        def detect_cuda_version():
            """Attempts to detect the CUDA version through multiple methods, prioritizing `nvcc` if available."""

            def get_version_from_nvcc():
                try:
                    output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT).decode()
                    for line in output.splitlines():
                        if "release" in line:
                            return line.split("release")[-1].strip().split(",")[0]
                except (FileNotFoundError, subprocess.CalledProcessError):
                    return None

            def get_version_from_env():
                cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                if cuda_home:
                    version_file = os.path.join(cuda_home, "version.txt")
                    try:
                        with open(version_file, "r") as f:
                            line = f.readline().strip()
                            if "CUDA Version" in line:
                                return line.split("CUDA Version")[-1].strip()
                    except FileNotFoundError:
                        return None
                return None

            def get_version_from_nvidia_smi():
                try:
                    output = subprocess.check_output(["nvidia-smi"]).decode()
                    for line in output.split("\n"):
                        if "CUDA Version" in line:
                            return line.split("CUDA Version:")[-1].strip()
                except (FileNotFoundError, subprocess.CalledProcessError):
                    return None

            # Try each method in sequence until a CUDA version is detected
            cuda_version = (get_version_from_nvcc() or
                            get_version_from_env() or
                            get_version_from_nvidia_smi())
            if cuda_version is not None:
                cuda_version = cuda_version.split(" ")[0]
            
            return cuda_version


        cuda_version = detect_cuda_version()
        print(f"\033[92mDetected CUDA version: {cuda_version}\033[0m") 
        if cuda_version in ["11.7", "11.8", "12.0"]:
            torch_index_url = "https://download.pytorch.org/whl/cu117"
        elif cuda_version in ["11.3", "11.4", "11.5", "11.6"]:
            torch_index_url = "https://download.pytorch.org/whl/cu113"
        elif cuda_version in ["12.1", "12.2", "12.3"]:
            torch_index_url = "https://download.pytorch.org/whl/cu121"
        elif cuda_version >= "12.4":
            torch_index_url = "https://download.pytorch.org/whl/cu124"
        else:
            torch_index_url = None

        # Install torch with the specific index URL
        if torch_index_url:
            if cuda_version >= "12.1":
                subprocess.run([
                    "pip", "install", "torch", "torchvision", "xformers", "--index-url", torch_index_url
                ])
            else:
                subprocess.run([
                "pip", "install", "torch", "torchvision", "--index-url", torch_index_url
                ])
                subprocess.run([
                    "pip", "install", "xformers==0.0.20"
                ])
        else:
            subprocess.run([
                "pip", "install", "torch", "torchvision"
            ])
            
        # Import torch after installation to check the version
        import torch
        torch_version = torch.__version__
        print(f"\033[92mDetected Torch version: {torch_version}\033[0m")

        # Install the appropriate pytorch-lightning version based on the PyTorch version
        if torch_version >= "2.1":
            lightning_version = ">=2.1,<=2.4"  # Adjust version range if needed
        elif torch_version >= "2.0":
            lightning_version = ">=2.0,<=2.3"
        elif torch_version >= "1.13":
            lightning_version = ">=1.9,<=2.2"
        elif torch_version >= "1.12":
            lightning_version = "<=2.1"
        else:
            lightning_version = ""  # Install the latest if no specific match

        # Install pytorch-lightning with the specified version if needed
        if lightning_version:
            subprocess.run(["pip", "install", f"pytorch-lightning{lightning_version}"])
        else:
            subprocess.run(["pip", "install", "pytorch-lightning"]) 

        # Install additional submodules
        subprocess.run(["pip", "install", "-e", "./submodules/diff-gaussian-rasterization"])
        subprocess.run(["pip", "install", "-e", "./submodules/simple-knn"])
        subprocess.run(["pip", "install", "-e", "./submodules/threestudio"])
        
        additional_dependencies = [
            "opencv-python",
            "imageio",
            "imageio-ffmpeg",
            "tyro",
            "viser",
            "nerfview",
            "roma",
            "yacs",
            "libigl",
            "envlight",
            "pytorch_lightning",
            "einops",
            "omegaconf",
            "controlnet_aux",
            "diffusers",
            "transformers",
            "nerfacc",
            "matplotlib",
            "transforms3d",
            "open3d",
            "wandb",
            "pymcubes",
            "nerfstudio",
            "ninja",
            "lpips",
            "open-clip-torch==2.7.0",
            "mvdream @ git+https://github.com/bytedance/MVDream",
            "imagedream @ git+https://github.com/bytedance/ImageDream/#subdirectory=extern/ImageDream",
            "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast/",
            "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
            "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git",
            "accelerate>=0.26.0",
        ]
        
        for package in additional_dependencies:
            subprocess.run(["pip", "install", package])

        # Create symbolic links
        current_dir = os.getcwd()
        subprocess.run(["ln", "-sf", f"{current_dir}/soar/threestudio-soar", f"{current_dir}/submodules/threestudio/custom/"])
        subprocess.run(["ln", "-sf", f"{current_dir}/submodules/threestudio/outputs", f"{current_dir}/outputs"])


install_requires = []

setup(
    name="soar",
    version="0.0.0",
    python_requires=">=3.10",
    packages=find_packages(include=["soar"]),
    install_requires=install_requires,
    extras_require={
        "dev": ["black", "isort", "ipdb"],
    },
    cmdclass={
        'install': CustomInstallCommand,
    },
)
