# Building ML Powered Applications

![Book cover](/images/ML_Powered_cover.jpg)

Welcome to the companion code repository for the O'Reilly book [Building ML Powered Applications](http://bit.ly/mlpowered-oreilly).
The book is availble on [Amazon](http://bit.ly/mlpowered).

This repositoryconsists of three parts:
- A set of Jupyter notebooks in the `notebook` folder serve to illustrate concepts covered in the book.
- A library in the `ml_editor` folder contains core functions for the book's case study example, a Machine Learning driven writing assistant.
- A Flask app demonstrates a simple way to serve results to users

## Setup instructions

### Python environment

I recommend first creating a python virtual environment using [virtualenv](https://pypi.org/project/virtualenv/):

`virtualenv ml_editor`

You can then activate it by running:

`source ml_editor/bin/activate`

Then, install project requirements by using:

`pip install -r requirements.txt`

### Downloading required data

## Notebook examples

The notebook folder contains usage examples for concepts covered in the book.

To run the notebooks yourselves:

- Download stackoverflow [archives](https://archive.org/details/stackexchange)

- Run `parse_xml_to_csv` to convert one of them to DataFrames

- Run `generate_model_text_features` to generate DataFrames with features precomputed

## Running the prototype Flask app

To run the app, simply navigate to the root of the repository and run:

`export FLASK_APP=app.py`

Followed by:

`flask run `

The above command should spin up a local web-app you can access at ` http://127.0.0.1:5000/`


Project structure inspired by the great [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).