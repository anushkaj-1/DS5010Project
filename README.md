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

Below is a simple example demonstrating how to use PyPreprocessor to preprocess an Excel dataset.

```python
from pypreprocessor_lib import DatasetPreprocessor
import pandas as pd

# Load data
df = pd.read_csv("/path/to/your/data.csv")
print('Initial Shape of Data:', df.shape)

# Initialize preprocessor
preprocessor = DatasetPreprocessor('/path/to/your/data.csv')

# Preprocess data
preprocessed_data = preprocessor.preprocess(target_label='YourTargetLabel')
print(preprocessed_data.head())
```
## Contributors

- **Riddhi Gupta**: Developed image preprocessing techniques.
- **Anushka Jami**: Designing  preprocessing methods for categorical and text data.
- **Lohith Vardireddygari**: Developed Excel data preprocessing functionalities.


## Acknowledgements

Thanks to the following resources that have been instrumental for this project:
- [Data Cleaning Using Pandas](https://www.analyticsvidhya.com/blog/2021/06/data-cleaning-using-pandas/)
- [Top Ten Ways to Clean Your Data](https://support.microsoft.com/en-us/office/top-ten-ways-to-clean-your-data)
- [Cleaning Image Dataset](https://visual-layer.readme.io/docs/cleaning-image-dataset)
- [Video Games Dataset](https://www.kaggle.com/datasets/shivamvadalia27/video-games)
- [Banana Dataset](https://www.kaggle.com/datasets/l3llff/banana)

  




