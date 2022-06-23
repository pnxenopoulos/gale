gale
===================
|Downloads| |Colab| |Github| |Build| |License|

.. |Downloads| image:: https://static.pepy.tech/personalized-badge/gale-topo?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads
   :target: https://pepy.tech/project/gale-topo

.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/drive/16JhR2nhm9J-9KtqmuxGsKMIGL_5c_42N?usp=sharing
   
.. |Github| image:: https://img.shields.io/badge/github-repo-yellowgreen
   :target: https://github.com/pnxenopoulos/gale
   
.. |Build| image:: https://github.com/pnxenopoulos/gale/actions/workflows/build.yml/badge.svg
   :target: https://github.com/pnxenopoulos/gale/actions/workflows/build.yml
   
.. |License| image:: https://img.shields.io/badge/license-MIT-lightgrey
   :target: https://github.com/pnxenopoulos/gale/blob/main/LICENSE
   
`gale` provides functions to globally compare local explanations. To install the library, run ``pip install gale-topo``.

.. _repository: https://github.com/pnxenopoulos/gale

Using this library to parse CSGO demos is as easy as the few lines of code shown below. To see what output looks like, check out :doc:`parser_output`.

.. code-block:: python

   from gale import create_mapper

   create_mapper(data)


Using gale
----------
:doc:`installation`
   How to install `gale`.

:doc:`examples`
   Examples code and Jupyter notebooks to help get you started.

:doc:`faq`
   Need help? Check the FAQs first.

:doc:`license`
   License and acknowledgments.

gale Modules
------------
:doc:`mapper`
   Mapper module.

.. Hidden TOCs

.. toctree::
   :caption: Getting Started
   :maxdepth: 2
   :hidden:

   installation
   examples
   faq
   license

.. toctree::
   :caption: Documentation
   :maxdepth: 2
   :hidden:

   mapper