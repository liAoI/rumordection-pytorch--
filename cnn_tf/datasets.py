import csv
import numpy as np
import logging
'''
处理数据集
'''
#########配置log日志方便打印#############

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)
test_percentage = 10
validation_percentage = 10



current_label = 1  #非谣言为0  谣言为1   roumordataset*.csv文件前一千条为谣言，后一千条为非谣言

dataFile = '../../data/roumordatasetPCA.csv'
OUTPUT_file = '../../data/roumorPCAdata.npy'

def create_dataSet(dataFile):
    training_feature = []
    testing_feature = []
    validation_feature = []

    training_label = []
    testing_label = []
    validation_label = []
    count = 0
    with open(dataFile, "r") as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_feature.append(line)
                if count <1000:
                    validation_label.append([1,0])

                else:
                    validation_label.append([0, 1])
            elif chance < (test_percentage+validation_percentage):
                testing_feature.append(line)
                if count < 1000:
                    testing_label.append([1, 0])
                else:
                    testing_label.append([0, 1])
            else:
                training_feature.append(line)

                if count < 1000:
                    training_label.append([1, 0])
                else:
                    training_label.append([0, 1])
            count+=1


    logger.info(count)
    logger.info(len(training_feature))
    logger.info(len(training_label))

    logger.info(len(validation_feature))
    logger.info(len(validation_label))

    logger.info(len(testing_feature))
    logger.info(len(testing_label))

    #将list转换成matrix
    training_feature = np.matrix(training_feature,dtype=float)
    testing_feature = np.matrix(testing_feature,dtype=float)
    validation_feature = np.matrix(validation_feature,dtype=float)

    training_label = np.matrix(training_label,dtype=float)
    testing_label = np.matrix(testing_label,dtype=float)
    validation_label = np.matrix(validation_label,dtype=float)

    state = np.random.get_state()
    np.random.shuffle(training_feature)
    np.random.set_state(state)
    np.random.shuffle(training_label)


    return np.asarray([training_feature,training_label,
                       validation_feature,validation_label,
                       testing_feature,testing_label])

def main():
    processed_data = create_dataSet(dataFile)
    np.save(OUTPUT_file,processed_data)

if __name__ == '__main__':
    # main()
    data = np.load(OUTPUT_file)
    for i in data:
        print(i.shape)