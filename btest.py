import datetime
import backtrader as bt
import backtrader.feeds as btf
import STGC as PRD
import numpy as np
import math


# Create a Stratey
class TestStrategy(bt.Strategy): 

    @staticmethod   
    def makeTime(dt):
        w = dt.weekday()    
        w_s = w*24*60/PRD.PERIOD
        h = dt.hour
        h_s = h*60/PRD.PERIOD
        m = dt.minute
        m_s = math.ceil(m/PRD.PERIOD)
        return int(w_s + h_s + m_s) 

    def prepare(self):
        #make data
        self.dataclose.append(self.data.close[0])
        self.datavolume.append(self.data.volume[0])
        self.datahigh.append(self.data.high[0])
        self.datalow.append(self.data.low[0])
        self.datatime.append(TestStrategy.makeTime(self.data.datetime.datetime()))

        # print(dir(self.datatime.datetime()))
        # dt = dt or self.datas[0].datetime.datetime()
        # print('%s, %s' % (dt, txt))

    def __init__(self):
        # Keep a reference to the "close" line in the datas[0] dataseries
        self.dataclose = [] 
        self.datahigh = []
        self.datalow = []
        self.datavolume = []
        self.datatime = []
        
        # To keep track of pending orders
        self.order = None
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None
    
    def next(self):

        self.prepare()

        if len(self.dataclose) < PRD.INPUT_WIDTH:
            return
        highest = self.datahigh[-PRD.INPUT_WIDTH:]
        lowest = self.datalow[-PRD.INPUT_WIDTH:]
        closed = self.dataclose[-PRD.INPUT_WIDTH:]
        volumed = self.datavolume[-PRD.INPUT_WIDTH:]
        timed = self.datatime[-PRD.INPUT_WIDTH:]
        midX = [highest, lowest, closed, volumed, timed]
        midX = np.array(midX)
        max_v = max(midX[0])
        min_v = min(midX[1])
        f_k = lambda x: int(math.floor(PRD.GRID_HIGH*(x-min_v)/(max_v-min_v)))
        for ii in range(3):
            for i,v in enumerate(midX[ii]):
                midX[ii][i] = f_k(v)

        midX = midX / PRD.GRID_HIGH    
        midX = midX.reshape((1, PRD.INPUT_HEIGHT, PRD.INPUT_WIDTH))
        print(midX)
        print(PRD.PredictNext(midX))
        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] < self.dataclose[-1]:
                    # current close less than previous close

                    if self.dataclose[-1] < self.dataclose[-2]:
                        # previous close less than the previous close

                        # BUY, BUY, BUY!!! (with default parameters)
                        self.log('BUY CREATE, %.2f' % self.dataclose[0])

                        # Keep track of the created order to avoid a 2nd order
                        self.order = self.buy()

        # else:

        #     # Already in the market ... we might sell
        #     if len(self) >= (self.bar_executed + 5):
        #         # SELL, SELL, SELL!!! (with all possible default parameters)
        #         self.log('SELL CREATE, %.2f' % self.dataclose[0])

        #         # Keep track of the created order to avoid a 2nd order
        #         self.order = self.sell()


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(TestStrategy)

    # Create a Data Feed
    data = bt.feeds.GenericCSVData(
        dataname='./XTIUSD.csv',
        dtformat = ('%Y.%m.%d %H:%M'),
        nullvalue=0.0,
        datetime=0,
        open=1,
        high=2,
        timeframe = bt.TimeFrame.Minutes,
        compression=1,
        fromdate=datetime.datetime(2017, 6, 1),
        todate=datetime.datetime(2017, 6, 20),
        low=3,        
        close=4,
        volume=5,
        openinterest=-1
    )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())