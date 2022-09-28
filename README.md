# My M.Sc. project

# Beta cell detection, segmentation, and tracking

Welcome to the repository of my master thesis project in biomedical imaging for computer vision! You can view my thesis here: https://t.ly/m6qF

The thesis abstract:

The field of deep learning for computer vision has seen an explosion of results in the last decade, and has become the current state­of­the­art approach to the tasks of object detection and image segmentation. Medical imaging has certainly benefited from this rapid development of deep learning, but it is hampered by costly annotations; even though there is more biological data available than ever before, with new image capturing technologies
and advancements in data storage and processing incessantly unfolding. It is therefore
in the interest of the research community to be efficient with data annotation procedures,
which is explored in this thesis.

The malfunction of insulin production in the human body is linked to various illnesses, such
as diabetes types I and II. Insulin is produced in the pancreas by cells called β-cells, which
are the result of a stem cell which specialized in producing insulin. It is well understood
which long-range factors, such as hormone proteins in the blood, influence stem cells to
differentiate into insulin-producing β-cells; what’s less known, however, is what factors
in the stem cell’s immediate environment facilitate this differentiation. If those factors
were discovered, β-cells could be produced in vitro, leading to potentially new treatments
of insulin-related diseases. It is therefore in the interest of researchers to investigate
the β-cells’ environment for clues as to what causes them to differentiate. To that end,
various computational tools are needed; in particular, tools which enable the tracking of
the movement of the β-cells in 3D will be of great interest, which is the subject of this
thesis project.

In essence, in this thesis project, computational tools were provisioned which accomplished the following:

- Extraction of the position of a β-cell in 3D space;

- Procurement of the trajectory of a β-cell in 3D space over time;

- Extraction of the 3D shape of a β-cell;

- Estimating the amount of insulin produced in a β-bell at a given time.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
