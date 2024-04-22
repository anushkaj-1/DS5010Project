# DS5010Project: Dataset Preprocessor

This module provides tools to preprocess Excel spreadsheets, image files and text files. This module can handle scaling, missing values, and more.

## Data Loading.
The class DatasetPreprocessor creates objects with a dataset_path, with dataset_type and dataset attributes. The dataset_path refers to the file path necessary to access where the data is stored. This class has a method to determine dataset_type (Excel file, image file, or text file). The dataset itself defaults to None, but is updated with the dataset depending on its dataset_type.

## Data Processing Techniques
Once the dataset is loaded into the object, the package provides processing functionality depending on the dataset. The package can handle missing values, normalization, and encoding for tabular data. For images, the package can resize, convert to grayscale, and apply filters. For text data, the package can remove stop works and special characters to clean and normalize text.

## Credits
This module was worked on by Riddhi Gupta, Lohith Vardireddygari, and Anushka Jami
