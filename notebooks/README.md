This folder contains notebooks showcasing concepts covered in the book.
Most of the examples only use one of the subfolders in archive
(the one that contains data for writers.stackexchange.com).

I've included a processed version of the data as a `.csv` for convenience.

If you want to generate this data yourself, or generate it for another subfolder,
you should:

- Download a subfolder from the stackoverflow [archives][archives]

- Run `parse_xml_to_csv` to convert it to a DataFrame

- Run `generate_model_text_features` to generate a DataFrames with precomputed features

[archives]: https://archive.org/details/stackexchange

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

### Improving the Model

- [Second Model][SecondModel]
- [Third Model][ThirdModel]

### Model Comparison

- [Comparing Models][ComparingModels]

### Generating Suggestions from Models

- [Generating Recommendations][GeneratingRecommendations]

[BlackBoxExplainer]: ./black_box_explainer.ipynb
[ClusteringData]: ./clustering_data.ipynb
[ComparingDataToPredictions]: ./comparing_data_to_predictions.ipynb
[ComparingModels]: ./comparing_models.ipynb
[DatasetExploration]: ./dataset_exploration.ipynb
[ExploringDataToGenerateFeatures]: ./exploring_data_to_generate_features.ipynb
[FeatureImportance]: ./feature_importance.ipynb
[GeneratingRecommendations]: ./generating_recommendations.ipynb
[SecondModel]: ./second_model.ipynb
[SplittingData]: ./splitting_data.ipynb
[TabularDataVectorization]: ./tabular_data_vectorization.ipynb
[ThirdModel]: ./third_model.ipynb
[TopK]: ./top_k.ipynb
[TrainSimpleModel]: ./train_simple_model.ipynb
[VectorizingText]: ./vectorizing_text.ipynb
