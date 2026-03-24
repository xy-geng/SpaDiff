from setuptools import setup

# with open("README.rst", "r", encoding="utf-8") as f:
#     __long_description__ = f.read()

if __name__ == "__main__":
    setup(
        name = "SpaDiff",
        version = "1.0.0",
        description = "Generating spatially coherent tissue structures across spatial multi‑slice multi‑omics data by spatial diffusion dynamics",
        url = "https://github.com/",
        author = "Xinyi Geng",
        author_email = "",
        license = "MIT",
        packages = ["SpaDiff"],
        install_requires = ["requests"],
        zip_safe = False,
        include_package_data = True,
        long_description = """ Long Description """,
        long_description_content_type="text/markdown",
    )