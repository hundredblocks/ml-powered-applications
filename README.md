# Building ML Powered Applications

Companion code for the O'Reilly book [Building ML Powered Applications](http://bit.ly/mlpowered-oreilly).

Project structure inspired by the great [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).

## Setup instructions

I recommend first creating a virtualenvironment as such:

`virtualenv ml_editor`

You can then activate it by running:

`source ml_editor/bin/activate`

Then, install requirements using:

`pip install -r requirements.txt`

## Notebook examples

The notebook folder contains usage examples for concepts covered in the book.

To run the notebooks yourselves:

- Download stackoverflow [archives](https://archive.org/details/stackexchange)

- Run `parse_xml_to_csv` to convert one of them to DataFrames

- Run `generate_model_text_features` to generate DataFrames with features precomputed

## Prototype Flask app

To run the app, run:

`export FLASK_APP=app.py`
`flask run `