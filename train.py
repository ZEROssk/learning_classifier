import numpy as np
import random
import vec2train
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Link,Chain,ChainList
from chainer import optimizers
from chainer import training,utils,Variable
import pickle

f = open("train_cov","rb")
train_cov = pickle.load(f)
f.close()

f = open("train_label_cov","rb")
train_label = pickle.load(f)
f.close()

f = open("test_cov","rb")
test_cov = pickle.load(f)
f.close()

f = open("test_label_cov","rb")
test_label = pickle.load(f)
f.close()

def test():
    ok = 0

    for a in range(len(test_cov)):
        x = Variable(np.array([test_cov[a]],dtype=np.float32))
        t = test_label[a]

        out = model.fwd(x)
        ons = np.argmax(out.data)

        if ans==t:
            ok+=1

    result = ok/len(test_cov)

    print(result)

class model(Chain):
    def __init__(self):
        super(model.self).__init__(
            cn1=L.Convolution2D(3,20,5),
            cn2=L.Convolution2D(20,50,5),
            l1=L.Linear(800,500),
            l2=L.Linear(500,2),
        )

    def __call__(self,x,t):
        return F.softmax_cross_rntropy(self.fwd(x),t)

    def fwd(self,x):
        h1=F.max_pooling_2d(F.relu(self.cn1(x)),2)
        h2=F.max_pooling_2d(F.relu(self.cn2(h1)),2)
        h3=F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)

model = model()
optimizer = optimizers.Adam()
optimizer.setup(model)

x = Variable(np.array(train_cov,dtype=np.float32))
t = Variable(np.array(train_label,dtype=np.int32))

for a in range(10):
    model.cleargrads()
    loss = model(x,t)
    loss.backward()
    optimizer.update()
    if a%1==0:
        test()
