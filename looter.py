import random


def randompiratekillvote(lootpercent):
    return not (random.random() < lootpercent)


def randompiratelootdistribution(numberofpirates):
    l = [random.random() for i in range(numberofpirates)]
    sm=sum(l)
    l = list(map(lambda f: f/sm, l))
    return l


print(randompiratekillvote(1.0))
print(randompiratekillvote(0.0))
print(randompiratekillvote(0.5))
print(randompiratekillvote(0.5))
print(randompiratekillvote(0.5))
print(randompiratekillvote(0.5))
print(randompiratelootdistribution(3))
print(sum(randompiratelootdistribution(3)))

from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], tensor.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')