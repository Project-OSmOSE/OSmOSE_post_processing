🐳 Installation
===============

.. _installation:

To use **OSmOSE_post_processing**, you first need to install it using one of the following:

From Git
--------

You can install **OSmOSE_post_processing** from `git <https://git-scm.com/>`_ if you want to have an editable version of it. This can be usefull for contributing to **OSmOSE_post_processing**, or to be able to easily update to the latest version.

The package will be installed with `uv <https://docs.astral.sh/uv/>`_, which is a Python package and project manager. Please refer to the `documentation <https://docs.astral.sh/uv/getting-started/installation/>`_ for installation.

To download the repository, simply clone it from git: ::

    git clone https://github.com/Project-OSmOSE/OSmOSE_post_processing.git

Then, you can pull the latest update: ::

    git pull origin main

You can now install the package using uv from the cloned repository: ::

    uv sync

This will create a virtual environment within your cloned repository and install all required dependencies for using and contributing to **OSmOSE_post_processing**.


With pip
--------

Alternatively, you can install it with a wheel file for any of our releases:

* Get the ``.whl`` wheel file from `the github repository <https://github.com/Project-OSmOSE/OSmOSE_post_processing/releases>`_.
* Install it in a virtual environment using pip: ::

    pip install post_processing-x.x.x.-py3-none-any.whl