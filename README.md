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

This repository has been tested on Python 3.6. It aims to support any Python 3 version.

To setup, start by cloning the repository:

`git clone https://github.com/hundredblocks/ml-powered-applications.git`

Then, navigate to the repository and create a python virtual environment using [virtualenv](https://pypi.org/project/virtualenv/):

`cd ml-powered-applications`

`virtualenv ml_editor`

You can then activate it by running:

`source ml_editor/bin/activate`

Then, install project requirements by using:

`pip install -r requirements.txt`

The library uses a few models from spacy. To download the small English model (required to run the app), run this command from a terminal with your virtualenv activated:

`python -m spacy download en_core_web_sm`

Finally, the notebooks and library leverage the `nltk` package.
The package comes with a set of resources that need to be individually downloaded.
To do so, open a Python session in an activated virtual environment, import `nltk`, and download the required resource.

Here is an example of how to do this for the `punkt` package from an active vierutalenvironment with `nltk` installed:

`python`

`import nltk`

`nltk.download('punkt')`

### Downloading required data

## Notebook examples

The notebook folder contains usage examples for concepts covered in the book.
Most of the examples only use one of the subfolders in archive (the one that contains data for writers.stackexchange.com).
I've includeda processed version of the data as a `.csv` for convenience.

If you wanted to generate this data yourself, or generate it for another subfolder, you should:

- Download a subfolder from the stackoverflow [archives](https://archive.org/details/stackexchange)

- Run `parse_xml_to_csv` to convert it to a DataFrame

- Run `generate_model_text_features` to generate a DataFrames with precomputed features

The notebooks belong to a few categories of concepts, described below.

### Data Exploration and Transformation

- [Dataset Exploration][DatasetExploration]
- [Splitting Data][SplittingData]
- [Vectorizing Text][VectorizingText]
- [Clustering Data][ClusteringData]
- [Tabular Data Vectorization][TabularDataVectorization]
- [Exploring Data To Generate Features][ExploringDataToGenerateFeatures]

### Initial Model Training and Performance Analysis

- [Train Simple Model][TrainSimpleModel]
- [Comparing Data To Predictions][ComparingDataToPredictions]
- [Top K][TopK]
- [Feature Importance][FeatureImportance]
- [Black Box Explainer][BlackBoxExplainer]

### Model Comparison

- [Second Model][SecondModel]
- [Comparing Models][ComparingModels]

### Generating Suggestions from Models

- [Third Model][ThirdModel]
- [Generating Recommendations][GeneratingRecommendations]

[BlackBoxExplainer]: ./notebooks/black_box_explainer.ipynb
[ClusteringData]: ./notebooks/clustering_data.ipynb
[ComparingDataToPredictions]: ./notebooks/comparing_data_to_predictions.ipynb
[ComparingModels]: ./notebooks/comparing_models.ipynb
[DatasetExploration]: ./notebooks/dataset_exploration.ipynb
[ExploringDataToGenerateFeatures]: ./notebooks/exploring_data_to_generate_features.ipynb
[FeatureImportance]: ./notebooks/feature_importance.ipynb
[GeneratingRecommendations]: ./notebooks/generating_recommendations.ipynb
[SecondModel]: ./notebooks/second_model.ipynb
[SplittingData]: ./notebooks/splitting_data.ipynb
[TabularDataVectorization]: ./notebooks/tabular_data_vectorization.ipynb
[ThirdModel]: ./notebooks/third_model.ipynb
[TopK]: ./notebooks/top_k.ipynb
[TrainSimpleModel]: ./notebooks/train_simple_model.ipynb
[VectorizingText]: ./notebooks/vectorizing_text.ipynb

## Pretrained models

You can train and save models using the notebooks in the `notebook` folder.
For convenience, I've included three trained models and two vectorizers, serialized in the `models` folder.
These models are loaded by notebooks demonstrating methods to compare model results, as well as in the flask app.

## Running the prototype Flask app

To run the app, simply navigate to the root of the repository and run:

`export FLASK_APP=app.py`

Followed by:

`flask run `

The above command should spin up a local web-app you can access at ` http://127.0.0.1:5000/`

## Troubleshooting

If you have any questions or encounter any roadblocks, please feel free to open an issue or email me at mlpoweredapplications@gmail.com.


Project structure inspired by the great [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/).