import numpy as np

def sharpRatio(data):
    """
    return the weird sharp-ratio
    """
    def cal_return(data):
        ret = []
        num = len(data)-1
        for i in range(num):
            ret.append(data[i+1]-data[i])
        return ret

    ret = cal_return(data)
    re = np.array(ret)
    np.std
    return re.mean()/re.std()

if __name__ == '__main__':
    aa=[.1,.2,.3,.4,.5,.6,.7,.8,1.1,15,16,17,20,22]
    print(sharpRatio(aa))