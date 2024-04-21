"""#Run the library: PyPreprocessor_Lib"""

from pypreprocessor_lib import DatasetPreprocessor
import pandas as pd

"""# Excel Dataset"""

df = pd.read_csv("/TestFiles/banana_quality.csv")
df.head(10)
print('Initial Shape of Data', df.shape)

preprocessor = DatasetPreprocessor('/TestFiles/banana_quality.csv')
preprocessed_data = preprocessor.preprocess(target_label='Quality')
preprocessed_data.head()

"""#Image Data"""

path = '/TestFiles/thumbnail_google.jpg'
preprocessor = DatasetPreprocessor(path)
preprocessed_data = preprocessor.preprocess()
preprocessed_data.shape

"""#Text Data"""

df = pd.read_table("/TestFiles/My_old_man.txt", header=None, encoding='latin-1')
df.columns = ['Text']

preprocessor = DatasetPreprocessor('/TestFiles/My_old_man.txt')
preprocessed_data = preprocessor.preprocess()

print(preprocessed_data[:10])
#print(preprocessed_data[:1000])

"""#Excel Data with categorical data"""

df = pd.read_csv("/TestFiles/games.csv")
df.head(10)

preprocessor = DatasetPreprocessor("/TestFiles/games.csv")
preprocessed_data = preprocessor.preprocess(target_label='rating')
preprocessed_data.head(10)
