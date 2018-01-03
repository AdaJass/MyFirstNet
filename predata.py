"""
these codes help to prepare the raw data to trainable pickale output.
"""
import pickle
from datetime import datetime, timedelta
import math
from pandas import DataFrame
import random
from tqdm import tqdm

PERIOD = 5  # time period is 5 minutes, it will process 5 minite period data.
SAMPLE_LENGTH = 552  # a trainning sample length, 5 days' data
PREDICT_LENGTH = 48   #predict the next 4 hours data
GRID_HIGH = 1440    #it means that all price data will in the interval of [0,14400]
RAW_FILE_NAME = './XTIUSD5.csv'

filename = lambda : 'PER_'+str(PERIOD)+'_SAM_'+str(SAMPLE_LENGTH)\
+'_PRE_'+str(PREDICT_LENGTH)+'_'+RAW_FILE_NAME[2:-4]+'.pickle'

def makeTime(dt):
    w = dt.weekday()    
    w_s = w*24*60/PERIOD
    h = dt.hour
    h_s = h*60/PERIOD
    m = dt.minute
    m_s = math.ceil(m/PERIOD)
    return int(w_s + h_s + m_s)

def pickleRawData():
    f = open(RAW_FILE_NAME,'r')
    middata = []  #  [h l close vo time]
    rawdata = f.readlines()
    f.close()
    # rawdata = rawdata[-2500:]
    print('read and format data:\n')
    for line in tqdm(rawdata):
        cell = line.split(',')
        time = cell[0]+' '+cell[1]
        time = datetime.strptime(time,'%Y.%m.%d %H:%M')
        highest = float(cell[3])
        lowest = float(cell[4])
        close = float(cell[5])
        volume = int(cell[6])
        time = makeTime(time)
        middata.append([highest, lowest, close, volume, time])

    datalen = len(middata)
    assert( datalen> SAMPLE_LENGTH)
    num = datalen - SAMPLE_LENGTH +1
    lastdata = []
    print('read completed! now enter core processing:\n')
    for i in tqdm(range(num)):
        d = middata[i:i+SAMPLE_LENGTH]
        df = DataFrame(d)
        max_v = max(df[0])
        min_v = min(df[1])
        f_k = lambda x: int(math.floor(GRID_HIGH*(x-min_v)/(max_v-min_v)))
        df['h'] = df[0].apply(f_k)
        df['l'] = df[1].apply(f_k)
        df['c'] = df[2].apply(f_k)
        df['v'] = df[3]
        df['t'] = df[4]
        
        df['v'].astype('int')
        df['t'].astype('int')
        for i in range(5):
            del df[i]
        # print(df)
        matrix = df.as_matrix()
        matrix = matrix.transpose()
        matrix = matrix/1440
        # print(matrix)
        # exit()
        lastdata.append(matrix)

    lastlen = len(lastdata)
    assert(lastlen > PREDICT_LENGTH)
    lastnum = lastlen - PREDICT_LENGTH
    outxy = []
    print('core completed!, make output tuple:\n')
    for i in tqdm(range(lastnum)):
        outxy.append((lastdata[i],lastdata[i+PREDICT_LENGTH]))

    random.shuffle(outxy)
    # print(outxy)

    with open(filename(), 'wb') as f:
        pickle.dump(outxy, f)

def loadData():
    """return [(x,y), ...]
    """
    with open(filename(), 'rb') as f:
        data=pickle.load(f)
        return data

if __name__ == "__main__":
    pickleRawData()
