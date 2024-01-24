
# Project Setup Guide

This guide provides step-by-step instructions on how to set up the project from the `ds_nickfarrell` Git repository. Please follow these steps to ensure a correct setup.

## 1. Clone the Git Repository

First, clone the repository from the provided URL. You can do this by opening your terminal and running the following command:

```bash
git clone http://23.29.118.76:3000/mkdlabs/ds_nickfarrell.git
```

This will create a `ds_nickfarrell` directory with the project files in your current working directory.


## 2. Run init.sh

With the project path set, you can now run the `init.sh` script. In your terminal, navigate to the project base directory and execute the following command:

```bash
sudo -s
bash scripts/init.sh
```

This script initializes the project environment. This command is only for the first time.

## 4. Run setup.sh (Optional)

Run the `setup.sh` script to run the server and cronjob. It's included in init.sh as well, so, if you run init.sh, no need to run it. However, when you restart the server or your computer, you should run this script. Ensure you're in the project base directory and execute:

```bash
sudo -s
bash scripts/setup.sh
```

This script will perform the necessary setup tasks defined for the project.
