# PyPreprocessor

PyPreprocessor is a comprehensive Python library designed to facilitate the preprocessing of both structured and unstructured data for data science and machine learning applications. This tool aims to automate essential preprocessing tasks such as data cleaning, normalization, and transformation, supporting a variety of data types including Excel files, images, and text documents.

## Features

- **Data Loading**: Automatically detects and loads data based on file types, utilizing popular Python libraries like pandas for tabular data and OpenCV for image data.
- **Data Processing**:
  - **Tabular Data**: Includes normalization, encoding for categorical variables, and imputation for missing values.
  - **Image Data**: Supports resizing, converting to grayscale, and applying filters to enhance features.
  - **Text Data**: Features removal of stop words and special characters to prepare text for analysis.
- **Modular Design**: Allows for easy extension and customization to meet the specific needs of different data processing tasks.

## Usage

This section provides basic instructions on how to use PyPreprocessor to preprocess various types of data. You can extend these examples based on your specific project requirements.

### Preprocessing Image Data

Here is how to preprocess image files using PyPreprocessor:

```python
from pypreprocessor_lib import DatasetPreprocessor

# Specify the path to your image file
path = 'path_to_your_image.jpg'
preprocessor = DatasetPreprocessor(path)

# Preprocess the image
preprocessed_data = preprocessor.preprocess()

# Output the shape of the processed image data
print(preprocessed_data.shape)

### Preprocessing Excel Data

To preprocess data from an Excel file:

```python
from pypreprocessor_lib import DatasetPreprocessor

# Specify the path to your Excel file
path = 'path_to_your_excel_file.csv'
preprocessor = DatasetPreprocessor(path, target_label='YourTargetColumn')

# Preprocess the data
preprocessed_data = preprocessor.preprocess()

# Display the first few rows of the processed data
print(preprocessed_data.head())




