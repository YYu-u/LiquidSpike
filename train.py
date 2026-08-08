import os
from linecache import cache

from ultralytics import YOLO

os.environ['WANDB_DISABLED'] = 'true'
fr_dict = {}

from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()
    model = YOLO("snn_RME_yolov8s.yaml", task='detect').load('/path/to/checkpoint/.pt')
    print(model)
    model.train(data="gen1.yaml",device=[0,1,2,3],epochs=100)




#测试模型

