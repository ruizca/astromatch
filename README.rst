Statistical crossmatching of astronomical catalogues
----------------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge

NOTE: THIS IS A PRELIMINARY DEVELOPMENT VERSION,
FAR FROM BEING FULLY TESTED AND DOCUMENTED.

``astromatch`` is a Python 3 package for statistical cross-matching of
astronomical catalogues. We offer three different methods: 'lr', our own
implementation of the likelihood ratio method; 'xmatch', which uses an external
server hosted by the CDS; and 'nway', using the NWAY package.

``astromatch`` gives a consistent output for different cross-matching methods,
enabling an easier comparison of their results. It also offers a useful
framework, well integrated within the Astropy framework, for including new
methods.

Dependencies
------------
``astromatch`` depends on:

* ``numpy`` 
* ``astropy``
* ``mocpy``

Certain functionalities also requiere:

* ``nway``
* ``matplotlib``
* ``request``

Installation
------------

``astromatch`` can be easily installed using ``pip``::

    pip install astromatch

Alternatively, clone the repository and use pip to install the downloaded code::

    git clone https://github.com/ruizca/astromatch.git
    pip install ./astromatch

Examples
--------

You can find a Jupyter notebook explaining how to use astromatch in
``docs/astropy/examples.ipynb``.

License
-------

This project is Copyright (c) A. Ruiz and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause licence. See the licenses folder for
more information.
