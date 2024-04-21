"""#Run the library: PyPreprocessor_Lib"""

from pypreprocessor_lib import DatasetPreprocessor
import pandas as pd

"""# Excel Dataset"""

df = pd.read_csv("/content/banana_quality.csv")
df.head(10)
print('Initial Shape of Data', df.shape)

preprocessor = DatasetPreprocessor('/content/banana_quality.csv')
preprocessed_data = preprocessor.preprocess(target_label='Quality')
preprocessed_data.head()

"""#Image Data"""

path = '/content/google.jpeg'
preprocessor = DatasetPreprocessor(path)
preprocessed_data = preprocessor.preprocess()
preprocessed_data.shape

"""#Text Data"""

df = pd.read_table("/content/My_old_man.txt", header=None, encoding='latin-1')
df.columns = ['Text']

preprocessor = DatasetPreprocessor('/content/My_old_man.txt')
preprocessed_data = preprocessor.preprocess()

print(preprocessed_data[:10])
#print(preprocessed_data[:1000])

"""#Excel Data with categorical data"""

df = pd.read_csv("/content/games.csv")
df.head(10)

preprocessor = DatasetPreprocessor("/content/games.csv")
preprocessed_data = preprocessor.preprocess(target_label='rating')
preprocessed_data.head(10)
