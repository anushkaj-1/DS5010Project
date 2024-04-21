import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class DatasetPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset_type = self._check_dataset_type()
        self.dataset = None

    def _check_dataset_type(self):
        """
        Checks the type of dataset (Excel or image) based on file extension.

        Returns:
        - 'excel' if the dataset is an Excel file
        - 'image' if the dataset is an image file
        - 'txt' if the dataset is a text file
        - None if the dataset type is unsupported
        """

        file_extension = self.dataset_path.split('.')[-1].lower()

        if file_extension in ['xlsx', 'xls', 'csv']:
            return 'excel'
        elif file_extension in ['jpg', 'jpeg', 'png']:
            return 'image'
        elif file_extension in ['txt']:
            return 'text'
        else:
            return None

    def _load_excel_dataset(self):
        """
        Load dataset from an Excel file.

        Returns:
        - DataFrame containing the dataset
        """
        try:
            if self.dataset_path.endswith('.csv'):
                return pd.read_csv(self.dataset_path)
            else:
                return pd.read_excel(self.dataset_path)
        except Exception as e:
            raise Exception("Error loading Excel dataset: {}".format(str(e)))

    def _load_image_dataset(self):
        """
        Load dataset from an image file.

        Returns:
        - NumPy array representing the image
        """
        try:
            return cv2.imread(self.dataset_path)
        except Exception as e:
            raise Exception("Error loading image dataset: {}".format(str(e)))

    def _load_text_dataset(self):
        """
        Load dataset from a text file.

        Returns:
        - List of strings representing each line in the text file
        """
        try:
            with open(self.dataset_path, 'r', encoding='latin-1') as file:
                return file.readlines()
        except Exception as e:
            raise Exception("Error loading text dataset: {}".format(str(e)))

    def load_dataset(self):
        """
        Load dataset based on its type.

        Returns:
        - DataFrame if the dataset is Excel
        - NumPy array if the dataset is an image
        """
        if self.dataset_type == 'excel':
            return self._load_excel_dataset()
        elif self.dataset_type == 'image':
            return self._load_image_dataset()
        elif self.dataset_type == 'text':
            return self._load_text_dataset()
        else:
            raise ValueError("Unsupported dataset type")

#---------------------------------------------------------------------------------------------------------------------------

# Excel Data Preprocessing

    def _process_categorical_columns(self, df):
        """
        Process categorical columns in a DataFrame.

        Args:
        - df: DataFrame containing the data

        Returns:
        - Processed DataFrame with categorical columns converted using one-hot or label encoding
        """
        categorical_columns = df.select_dtypes(include=['object']).columns
        print('yo', categorical_columns)
        for col in categorical_columns:
            print('yoyo', df[col].nunique())
            if df[col].nunique() > 4 and df[col].nunique()< 10:
                encoder = OneHotEncoder(sparse=False, drop='first')
                encoded_cols = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded_cols, columns=[f"{col}_{val}" for val in encoder.categories_[0][1:]])
                df = pd.concat([df, encoded_df], axis=1)
                df.drop(columns=categorical_columns, inplace=True)
            else:
                print('here')
                encoder = LabelEncoder()
                encoded_col = encoder.fit_transform(df[col])
                df[col] = encoded_col

        # Drop the original categorical columns
        return df

    def _scale_data(self, data, method='standard'):
        """
        Scale numerical data.

        Args:
        - data: DataFrame containing numerical data
        - method: 'standard' for standardization, 'minmax' for min-max scaling

        Returns:
        - Scaled DataFrame
        """
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid scaling method")

            scaled_data = scaler.fit_transform(data)
            return pd.DataFrame(scaled_data, columns=data.columns)
        except Exception as e:
            raise Exception("Error scaling data: {}".format(str(e)))

    def _handle_missing_values(self, data, strategy='mean'):
        """
        Handle missing values in the dataset.

        Args:
        - data: DataFrame containing the dataset
        - strategy: Imputation strategy ('mean', 'median', 'most_frequent')

        Returns:
        - DataFrame with missing values imputed
        """
        try:
            imputer = SimpleImputer(strategy=strategy)
            imputed_data = imputer.fit_transform(data)
            imputed_data = pd.DataFrame(imputed_data, columns=data.columns)
            print(imputed_data.head())
            return imputed_data
        except Exception as e:
            raise Exception("Error handling missing values: {}".format(str(e)))

    def _remove_highly_correlated(self, data, threshold=0.95):
        """
        Remove highly correlated features from the dataset.

        Args:
        - data: DataFrame containing the dataset
        - threshold: Threshold for correlation coefficient

        Returns:
        - DataFrame with highly correlated features removed
        """
        try:
            corr_matrix = data.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            print(to_drop)
            return data.drop(columns=to_drop)
        except Exception as e:
            raise Exception("Error removing highly correlated features: {}".format(str(e)))


    def _detect_problem_type(self, dataset, target):
        """
        Automatically detect the type of problem (classification or regression).

        Args:
        - dataset: DataFrame containing the dataset with the target variable

        Returns:
        - 'classification' if the problem is detected as a classification problem
        - 'regression' if the problem is detected as a regression problem
        - 'unknown' if the problem type cannot be determined
        """
        target_variable = dataset[target]

        if target_variable.dtype == 'object':
            return 'classification'

        num_unique_values = target_variable.nunique()
        if num_unique_values <= 10:
            return 'classification'

        if len(target_variable.unique()) < len(dataset) / 10:
            return 'classification'

        return 'regression'

#---------------------------------------------------------------------------------------------------------------------------

# Text Data Preprocessing

    def _remove_stopwords(self, text):
        """
        Remove stopwords from text.

        Args:
        - text: Input text as a string

        Returns:
        - Text with stopwords removed
        """
        try:
            nltk.download('stopwords')
            nltk.download('punkt')
            stop_words = set(stopwords.words('english'))
            word_tokens = word_tokenize(text)
            filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
            return ' '.join(filtered_text)
        except Exception as e:
            raise Exception("Error removing stopwords from text: {}".format(str(e)))

    def _remove_special_characters(self, text):
        """
        Remove special characters from text.

        Args:
        - text: Input text as a string

        Returns:
        - Text with special characters removed
        """
        try:
            return re.sub(r'[^a-zA-Z0-9\s]', '', text)
        except Exception as e:
            raise Exception("Error removing special characters from text: {}".format(str(e)))


#---------------------------------------------------------------------------------------------------------------------------

# Image Data Preprocessing

    def _resize_image(self, image, target_size=(224, 224)):
        """
        Resize the image to a target size.

        Args:
        - image: NumPy array representing the image
        - target_size: Tuple specifying the target size (width, height)

        Returns:
        - Resized image as a NumPy array
        """
        try:
            return cv2.resize(image, target_size)
        except Exception as e:
            raise Exception("Error resizing image: {}".format(str(e)))

    def _convert_to_grayscale(self, image):
        """
        Convert the image to grayscale.

        Args:
        - image: NumPy array representing the image

        Returns:
        - Grayscale image as a NumPy array
        """
        try:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise Exception("Error converting image to grayscale: {}".format(str(e)))

    def _apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        """
        Apply Gaussian blur to the image.

        Args:
        - image: NumPy array representing the image
        - kernel_size: Tuple specifying the kernel size

        Returns:
        - Image with Gaussian blur applied
        """
        try:
            return cv2.GaussianBlur(image, kernel_size, 0)
        except Exception as e:
            raise Exception("Error applying Gaussian blur to image: {}".format(str(e)))


    def _detect_edges(self, image, min_val=100, max_val=200):
        """
        Detect edges in the image using the Canny edge detector.

        Args:
        - image: NumPy array representing the image
        - min_val: Lower threshold for edge detection
        - max_val: Upper threshold for edge detection

        Returns:
        - Image with detected edges
        """
        try:
            return cv2.Canny(image, min_val, max_val)
        except Exception as e:
            raise Exception("Error detecting edges in image: {}".format(str(e)))

    def _equalize_histogram(self, image):
        """
        Equalize the histogram of the image to enhance contrast.

        Args:
        - image: NumPy array representing the image

        Returns:
        - Image with equalized histogram
        """
        try:
            return cv2.equalizeHist(image)
        except Exception as e:
            raise Exception("Error equalizing histogram of image: {}".format(str(e)))

    def _apply_thresholding(self, image, threshold=128, max_value=255, type=cv2.THRESH_BINARY):
        """
        Apply thresholding to the image to convert it to binary format.

        Args:
        - image: NumPy array representing the image
        - threshold: Threshold value for binarization
        - max_value: Maximum value to use with binary thresholding
        - type: Type of thresholding (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, etc.)

        Returns:
        - Binary image
        """
        try:
            _, binary_image = cv2.threshold(image, threshold, max_value, type)
            return binary_image
        except Exception as e:
            raise Exception("Error applying thresholding to image: {}".format(str(e)))

    def _rotate_image(self, image, angle):
        """
        Rotate the image by a specified angle.

        Args:
        - image: NumPy array representing the image
        - angle: Angle of rotation in degrees

        Returns:
        - Rotated image
        """
        try:
            rows, cols = image.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            return cv2.warpAffine(image, rotation_matrix, (cols, rows))
        except Exception as e:
            raise Exception("Error rotating image: {}".format(str(e)))

    def _crop_image(self, image, x, y, width, height):
        """
        Crop a region of interest from the image.

        Args:
        - image: NumPy array representing the image
        - x: x-coordinate of the top-left corner of the ROI
        - y: y-coordinate of the top-left corner of the ROI
        - width: Width of the ROI
        - height: Height of the ROI

        Returns:
        - Cropped region of interest
        """
        try:
            return image[y:y+height, x:x+width]
        except Exception as e:
            raise Exception("Error cropping image: {}".format(str(e)))

    def _denoise_image(self, image):
        """
        Apply denoising filters to reduce noise in the image.

        Args:
        - image: NumPy array representing the image

        Returns:
        - Denoised image
        """
        try:
            # Convert the image to the required format (if necessary)
            # if image.shape[2] == 1:
            #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            # elif image.shape[2] == 4:
            #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        except Exception as e:
            raise Exception("Error denoising image: {}".format(str(e)))




    def preprocess(self,target_label=''):
        """
        Preprocess the dataset based on its type.

        Returns:
        - Preprocessed DataFrame if the dataset is Excel
        - Preprocessed image as a NumPy array if the dataset is an image
        """
        dataset = self.load_dataset()


        if self.dataset_type == 'excel':
            problem_type = self._detect_problem_type(dataset, target_label)
            print("Problem Type = {}".format(problem_type))
            removed_column = dataset[target_label]
            dataset.drop(columns = [target_label], inplace = True)
            dataset = self._process_categorical_columns(dataset)
            dataset = self._scale_data(dataset)
            dataset = self._handle_missing_values(dataset)
            dataset = self._remove_highly_correlated(dataset)
            dataset[target_label] = removed_column
            return dataset

        elif self.dataset_type == 'image':
            dataset = self._resize_image(dataset)
            dataset = self._convert_to_grayscale(dataset)
            dataset = self._apply_gaussian_blur(dataset)
            dataset = self._detect_edges(dataset)
            dataset = self._equalize_histogram(dataset)
            dataset = self._apply_thresholding(dataset)
            dataset = self._rotate_image(dataset, angle=45)
            dataset = self._crop_image(dataset, x=100, y=100, width=200, height=200)
            dataset = self._denoise_image(dataset)
            return dataset

        elif self.dataset_type == 'text':
            dataset = [self._remove_stopwords(text) for text in dataset]
            dataset = [self._remove_special_characters(text) for text in dataset]
            return dataset

        else:
            raise ValueError("Unsupported dataset type")
