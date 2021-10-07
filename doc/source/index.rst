.. ssml documentation master file, created by
   sphinx-quickstart on Thu Apr 18 15:34:45 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. figure:: figures/logo.png
    :scale: 50 %
    :alt: Logo

Welcome to Shadow's Documentation!
==================================

Shadow is a `PyTorch <https://pytorch.org/>`_ based library for semi-supervised machine learning.
The ``shadow`` python 3 package includes implementations of Virtual Adversarial Training,
Mean Teacher, and Exponential Averaging Adversarial Training.
Semi-supervised learning enables training a model (gold dashed line) from both labeled (red and
blue) and unlabeled (grey) data, and is typically used in contexts in which labels are expensive
to obtain but unlabeled examples are plentiful.

.. figure:: figures/ssml-halfmoons.png
    :align: center

Github development page:
------------------------

https://github.com/sandialabs/shadow



Installation
------------
Shadow can be installed directly from pypi as:

.. code-block:: shell

    pip install shadow-ssml

Hello World
============

Incorporating consistency regularizers into an existing supervised workflow for semi-supervised learning is straightforward.
First, Shadow provides techniques that wrap an existing PyTorch model:

.. code-block:: python

    model = ...  # PyTorch torch.nn.Module
    eaat = shadow.eaat.Eaat(model)  # Wrapped model

The wrapped model is used during training and inference.
The model wrapper provides a `get_technique_cost` method for computed the consistency cost based on unlabeled data.
This loss can be added to an existing loss computation to enable semi-supervised learning:

.. code-block:: python

    for x, y in trainloader:
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = eaat(x)

        # get semi-supervised loss, using supervised criterion and unsupervised criterion
        # provided by the model wrapper
        loss = criterion(x, y) + eaat.get_technique_cost(x)
        loss.backward()
        optimizer.step()


For a full working example, see the :doc:`MNIST Example <examples/mnist_example>`.

Citing Shadow
=============
To cite shadow, use the following reference:

* Linville, L., Anderson, D., Michalenko, J., Galasso, J., & Draelos, T. (2021). Semisupervised Learning for Seismic Monitoring Applications. Seismological Society of America, 92(1), 388-395. doi: https://doi.org/10.1785/0220200195


Contents
========
.. toctree::
    :maxdepth: 1

    overview.rst
    examples/halfmoons_example
    examples/mnist_example
    documentation.rst
    copyrightlicense.rst
    developers.rst
    references.rst


Contributors
============

* Dylan Anderson
* Lisa Linville
* Joshua Michalenko
* Jennifer Galasso
* Brian Evans
* Henry Qiu
* Christopher Murzyn
* Brodderick Rodriguez

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Sandia National Laboratories is a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia LLC, a wholly owned subsidiary of Honeywell International Inc. for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525.
