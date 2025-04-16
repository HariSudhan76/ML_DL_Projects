import sys
import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from logger import logging
from exception import CustomException
from utils import save_object

proj_root = Path(__file__).resolve().parent.parent.parent
csv_path = proj_root / "notebook" / "data" / "exams.csv"
df = pd.read_csv(csv_path)
num_col = [f for f in df.columns if df[f].dtype != 'O' and f != 'math score']##Math score need to be excluded otherwise it will create issues in function inititate_data_transformation()
cat_col = [f for f in df.columns if df[f].dtype == 'O']

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            
            numerical_feature = num_col
            categorical_feature = cat_col

            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy='median')), # Handle missing numerical data
                    ("scaler",StandardScaler())  # Normalize/standardize the features

                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),  # Handle missing categorical data
                    ("one_hot_encoder",OneHotEncoder()),   # Convert categories to binary vectors
                    ("scaler",StandardScaler(with_mean=False))   # Standardize the binary vectors
                    
                    #  with_mean=False -> without this it will throw error:
                    # OneHotEncoder by default returns a sparse matrix.
                    # StandardScaler tries to center the data (i.e., subtract mean = with_mean=True by default).
                    # But centering doesn't work on sparse matrices — it would destroy the sparsity (and be very inefficient)!
                    # If you're not doing any scaling on the one-hot encoded values (which are usually 0/1), you can even skip StandardScaler for the categorical part. It's not always necessary.
                    # But if you're planning to feed the preprocessed data into an ML model like Logistic Regression or SVM, scaling all features to the same range helps. So keeping it is fine — just disable with_mean.

                ]
            )
            ##So far pipeline is like mixer jar is created no fruits added to get the juice. 

            logging.info("Numerical columns encoding completed")

            logging.info("Categorical columns encoding completed")
            
            ##Combine both numerical and categorical pipeline together so will use column transformer
            ##ColumnTransformer will combine the above pipeline(Mixer) with data(Fruits) will get the numerical data(Juice)
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_feature), 
                    ("cat_pipeline",cat_pipeline,categorical_feature)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def inititate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column = 'math score'
            # numerical_columns = self.numerical_feature.drop(columns = target_column,axis=1)

            input_feature_train_df = train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info(
                f"applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
                             ]

            logging.info(f"Saved Preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)