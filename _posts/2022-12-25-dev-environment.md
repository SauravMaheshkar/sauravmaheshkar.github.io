---
title: Quest for the Perfect Python Development Environment
date: 2022-12-25
categories: python
tags: python docker
---

This blog post has been in work for a little over a year and I'll continue making changes as things progress overtime.

When one usually starts working on python project at a reasonable calibre, we start thinking about code quality (testing, type hints, documentation, dependency management, packaging, ....). Now this is a dangerous rabithole. There are a myriad of tools and configurations to pick from and obviously this blogpost is an **opinionated piece. You might prefer different linters or package managers. This is my setup and how I prefer to work.**

-

Let's get into it

## Project Structure

```lang-bash
├── .github
│   ├── codeql
│   │   └── codeql-config.yml
│   ├── dependabot.yml
│   └── workflows
│       ├── codeql-analysis.yml
│       ├── containers.yml
│       ├── publish-docker.yml
│       └── python.yml
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── Containerfile
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── requirements.txt
├── src
│   └── __init__.py
└── tests
    └── test_basic.py
```

## 🏠 Dependency Management

IMO this is the big kahuna. Dependency Management is arguably one of the most crucial aspects of any python project. Based on what packages you depend on and what python versions your project is compatible with your target audience will differ.

There are a couple of options to choose from:-

- ⭐️ [venv](https://docs.python.org/3/library/venv.html): Initially introduced in python 3.3, `venv` comes inbuilt with the default python installation and is my preferred dependency management tool of choice.
- [virtualenv](https://packaging.python.org/en/latest/key_projects/#virtualenv): Almost similar to `venv` but also supports python 2.7 and needs to be **installed seperately**. You can refer to this [video](https://youtu.be/MGTX5qI2Jts) by [@asottile](https://github.com/asottile) on the differences between `venv` and `virtualenv`. The only reason why I prefer `venv` over `virtualenv` is because it's not an external package.
- [conda](https://docs.conda.io/en/latest/): Another extremely popular tool which is closely tied with the anaconda toolset that is widely popular in the data science industry. I was a heavy user of `conda` prior to 2022 but recently switched over to `venv` for simplicity.
- [mamba](https://mamba.readthedocs.io/en/latest/index.html): Almost similar to `conda` but offering offering higher speed and more reliable environment solutions. While I haven't experimented with mamba often I do remember it being a key tool in some projects.
- [poetry](https://python-poetry.org/): I don't have much experience with poetry but along the same lines as `virtualenv`, I wanted to have a lean template repository with low overhead and poetry itself depends on a lot of other projects.

Thus, I usually start every project by creating a python `venv` within the project folder.

```lang-bash
python3 -m venv venv
make requirements # more about this later
```

## 🐳 Containerization

Now we're starting to get into the fancy bits. Now my cynicism isn't totally lost on me. You might think it's excessive to containerize every single project but in my line of work, having a container is extremely helpful. I often build and push containers to both Docker Hub and the Github Container Registry containing all the pre-configured dependencies and sometimes even training data for various deep learning projects. If you're working with projects that require GPUs you might not want to install various drivers on your system and risk messing up your system configuration. That's when maybe you'd like to use the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) which come preconfigured with GPU drivers and package versions for your project. I for instance only use GPUs with Tensorflow on Colab notebooks or by building on top of the [tensorflow-gpu](https://www.tensorflow.org/install/docker#examples_using_gpu-enabled_images) image. If you're trying to work on distributing on a particular research project where the process of pre-processing the dataset into the required format required a lot of work and time, it might be easier to just package the data within the container and share the container instead (however this is not an excuse to not include your pre-processing and data-acquisition script).

let's take a look at our file:

```dockerfile
# Use an ubuntu image
FROM ubuntu:22.04 AS builder

# metainformation
LABEL version="0.0.1"
LABEL maintainer="Saurav Maheshkar"

# Helpers
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Essential Installs
RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		gcc \
		gfortran \
		libopenblas-dev \
		python3.10 \
		python3-pip \
		python3.10-dev \
		python3.10-venv \
		&& apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3.10 install --no-cache-dir --upgrade pip setuptools wheel isort
RUN pip3.10 install --no-cache-dir -r requirements.txt

RUN find /opt/venv/lib/ -follow -type f -name '*.a' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.pyc' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.txt' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.mc' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.js.map' -delete \
    && find /opt/venv/lib/ -name '*.c' -delete \
    && find /opt/venv/lib/ -name '*.pxd' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.md' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.png' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.jpg' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.jpeg' -delete \
    && find /opt/venv/lib/ -name '*.pyd' -delete \
    && find /opt/venv/lib/ -name '__pycache__' | xargs rm -r

# Runner Image
FROM ubuntu:22.04 AS runner
RUN apt update && apt install -y --no-install-recommends \
		python3.10 \
		python3-pip \
		python3.10-dev \
		python3.10-venv \
		&& apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home user
WORKDIR /home/user
USER user

ENTRYPOINT ["/bin/bash"]
```

Below I describe my decisions on why I used certain features

<details>
<summary>Multi-Stage Builds</summary>
<br>
Notice the `FROM ubuntu:22.04 AS <"builder/runner">` and `COPY --from=builder ...` commands. These are used to utilise something called as Multi-Stage Builds. If you're building images meant for development rather than deployment it's easy to have builds which occupy gigabytes of memory. Each of the `RUN` and `COPY` commands create another layer and as a result lots of artifacts. These can lead to essentially bloated images. **One** of the tricks you can employ to reduce the size is to use stages.

Each use of the `FROM` creates another base/stage allowing you to selectively copy artifacts from other stages leading to a much slimmer and less bloated final image. In our case, we use a "builder" stage to install the essential packages needed to install our dependencies viz. `libopenblas-dev` or `gfortran`, and then create a virtual env using the desired python version. Now you might think this is too excessive ! isn't the entire purpose of using Docker for development is to isolate packages. Why do we need to create a virtual environment within docker itself ?!. If you read further and see where we use the "builder" stage again you'll notice we only copy the environment from the "builder" stage into our "runner" stage, every other artifact from the "builder" stage is left behind. This allows us to significantly reduce our image size. The only thing we need to ensure is that we are using the same python version in the "runner" image and we have updated our path to where the packages are placed.

For more details refer to the Docker manual on [Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/).

</details>

<details>
<summary>Why create a new user ?</summary>
Simply put security 🤷🏻‍♂️. Unless a program requires `sudo` privilege it's probably best to create another user and use that instead.

```dockerfile
RUN useradd --create-home user
WORKDIR /home/user
USER user
```

We use the following lines to create another switch and switch to it. If you decide to include files within your image, you could place your data within a directory under `/home/user`.

</details>

<details>
<summary>cache and entrypoints</summary>

When you install packages using `pip` or `apt` it usually creates a cache directory somewhere in your system. Now while this is usually helpful, it's not really required within a image.

In the case of `apt` we use:-

```lang-bash
apt install --no-install-recommends ...
apt-get clean
rm -rf /var/lib/apt/lists/*
```

In the case of `pip` we use:-

```lang-bash
pip install --no-cache-dir ...
```

Lastly being slightly more nit-picky about various non-essential files, we also look within the `/opt/venv/lib/` dir to search for various other files which might get installed with the various packages (`*.a`, `*.pyc`, `*.md`, `__pycache__/`).

```dockerfile
RUN find /opt/venv/lib/ -follow -type f -name '*.a' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.pyc' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.txt' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.mc' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.js.map' -delete \
    && find /opt/venv/lib/ -name '*.c' -delete \
    && find /opt/venv/lib/ -name '*.pxd' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.md' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.png' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.jpg' -delete \
    && find /opt/venv/lib/ -follow -type f -name '*.jpeg' -delete \
    && find /opt/venv/lib/ -name '*.pyd' -delete \
    && find /opt/venv/lib/ -name '__pycache__' | xargs rm -r
```

We also set a entrypoint of the `bash` terminal incase you want to use the container in an interactive manner.

</details>

<details>
<summary>Using Labels</summary>

Incase you decide to upload your image to some container registry you might want to include various metadata about your image. This could include the base image, the authors, version or license. The [Open Container Initiative](https://www.opencontainers.org/) has a list of Pre-Defined Annotation Keys available [here](https://github.com/opencontainers/image-spec/blob/main/annotations.md#pre-defined-annotation-keys).

At a minimum I like to mention the version number and maintainer information in my dockerfile.

```dockerfile
# metainformation
LABEL version="0.0.1"
LABEL maintainer="Saurav Maheshkar"
```

This information is later included within the image manifest. I have included an example below which contains the manifest of an image I created for reproducing the experiments in the [NeRF paper](https://www.matthewtancik.com/nerf)

<details>
<summary>Example Manifest with labels</summary>

```json
"labels": {
    "org.opencontainers.image.source": "= https://github.com/SauravMaheshkar/NeRF",
    "org.opencontainers.image.licenses": "= MIT",
    "org.opencontainers.image.base.name": "= index.docker.io/tensorflow/tensorflow:latest-gpu",
    "org.opencontainers.image.authors": "= Saurav Maheshkar",
    "org.opencontainers.image.version": "= 0.0.3"
  }
```

</details>
<br>
</details>

<details>
<summary>Why is it named Containerfile and not Dockerfile ?</summary>

Now that's a good question.

</details>

## References

1. [Creating the Perfect Python Dockerfile by Luis Sena](https://luis-sena.medium.com/creating-the-perfect-python-dockerfile-51bdec41f1c8)

2. [Matthijs Brouns - 10x smaller docker containers for Data Science, PyData Eindhoven 2020](https://youtu.be/Z1Al4I4Os_A)
