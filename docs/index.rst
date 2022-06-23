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

GALE is a Python library used to assess the similarity of local explanations from methods such as LIME, SHAP or generalized additive models (GAMs). To do so, GALE models the relationship between the explanation space and the model predictions as a scalar function. Then, we compute the topological skeleton of this function. This topological skeleton acts as a signature, which we use to compare outputs from different explanation methods. GALE is easy to use and can be deployed in just a few lines of code, as we show below

.. code-block:: python

   from gale import create_mapper, bottleneck_distance

   model_one_mapper = create_mapper(explanations_one, predictions_one, resolution=10, gain=0.3, dist_thresh=0.5)
   model_two_mapper = create_mapper(explanations_two, predictions_two, resolution=10, gain=0.3, dist_thresh=0.5)

   bottleneck_distance(model_one_mapper, model_two_mapper)


Using gale
----------
:doc:`installation`
   How to install `gale`.

:doc:`examples`
   Examples to help get you started with GALE.

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