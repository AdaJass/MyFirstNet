f = open('./XTIUSD5.csv','r')
middata = []  #  [h l close vo time]
rawdata = f.readlines()
f.close()
# rawdata = rawdata[-2500:]
f = open('./xtiusd.csv','w')
print('read and format data:\n')
for line in rawdata:
    line = line.replace(',',' ',1)    
    f.write(line) 
f.close()      
    