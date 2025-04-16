

import os
import sys 
from exception import CustomException
from logger import logging
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformation,DataTransformConfig
from model_trainer import ModelTrainerConfig,ModelTrainer

proj_root = Path(__file__).resolve().parent.parent.parent
csv_path = proj_root / "notebook" / "data" / "exams.csv"
##input like where to save train data, test data, raw data this is 
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(csv_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set = train_test_split(df,test_size=.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is complete")

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomException(e,sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.inititate_data_transformation(train_data,test_data) ##last not required as we have pkl file
    modeltrainer = ModelTrainer()
    scr,model = modeltrainer.initiate_model_trainer(train_arr,test_arr)
    print(f"The best model is {model} and it has scored: {round(scr*100,2)}")



