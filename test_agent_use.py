from functools import reduce
a=int(input())
print(reduce(lambda x,y:x*y,range(1,a+1)))

