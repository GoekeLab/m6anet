.. m6anet documentation master file, created by
   sphinx-quickstart on Fri Sep 11 16:32:52 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to m6anet's documentation!
==================================
m6anet is a python tool that leverages Multiple Instance Learning framework to detect m6a modifications from Nanopore Direct RNA Sequencing data.

m6anet requires Python version 3.7 or higher. To install the latest release with PyPI (recommended) run::

        pip install m6anet

See our :ref:`Installation page <installation>` for details.

To detect m6A modifications from your direct RNA sequencing sample, you can follow the instructions in our :ref:`Quickstart page <quickstart>`.
m6Anet is trained on dataset sequenced using the SQK-RNA002 kit and has been validated on dataset from SQK-RNA001 kit.
Newer pore version might alter the raw squiggle and affect segmentation and classification results and in such cases m6Anet might need to be retrained.

Contents
--------------------------
.. toctree::
   :maxdepth: 3

   installation
   quickstart
   cmd
   training
   help
   release_notes
   citing

Citing m6Anet
--------------------------
If you use m6Anet in your research, please cite
`Christopher Hendra, et al.,Detection of m6A from direct RNA sequencing using a Multiple Instance Learning framework. *Nat Methods* (2022) <https://doi.org/10.1038/s41592-022-01666-1>`_

Contacts
--------------------------
m6anet is developed and maintained by `Christopher Hendra <https://github.com/chrishendra93>`_ and `Jonathan Göke <https://github.com/jonathangoeke>`_ from the Genome Institute of Singapore, A*STAR. If you want to contribute, please leave an issue in `our repo <https://github.com/GoekeLab/m6anet/issues>`_

Thank you!
