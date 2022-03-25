# import torch
import os
from statistics import mode
import sys
paths=[
    r'E:\anaconda\envs\py37_ml_pt\Scripts',
    r'E:\anaconda\envs\py37_ml_pt\Library',
    r'E:\anaconda\envs\py37_ml_pt\Library\bin',
    r'E:\anaconda\envs\py37_ml_pt\Library\mingw-w64\bin'
]
for path in paths:
    os.environ['PATH']+=os.pathsep+path
from pyscipopt import Model,quicksum
if __name__=='__main__':
    '''
    两种信息如数输入模式：
    1.基于文件读取
        客户属性：name、商品体积、时间窗、和其他客户、车场之间的距离(运输成本)、运输时长
        车辆属性：种类、数量、容量、休整时间
        车场属性：车场name
    2.基于控件
        客户控件
        车场控件
        车辆控件

    不管是哪种输入，最终都会使用fluid算法重新获取一个邻接矩阵
    '''
    # 
    # randomly genetate data
    start=0
    end=4
    C=[1,2,3]
    V=[start,end]+C
    c={
        (0,1):1,(0,2):1,(0,3):1,(0,4):0,
        (1,4):1,
        (2,4):1,
        (3,4):1
    }
    tau={
        (0,1):1,(0,2):1,(0,3):1,(0,4):0,
        (1,4):1,
        (2,4):1,
        (3,4):1
    }
    q={
        1:1,
        2:1,
        3:0.5,
        4:0
    }

    Q={
        0:1,
        1:0.5
    }

    a={
        1:0.5,
        2:4.5,
        3:2
    }

    b={
        1:1.5,
        2:5.5,
        3:3
    }
    r={
        0:1,
        1:1
    }
    
    s={
        0:0,
        1:1,
        2:1,
        3:1
    }

    K=2
    N=2
    M=10
    model = Model("TW-VRP")
    x={}
    t={}
    
    for k in range(K):
        for n in range(N):
            t[end,k,n]=model.addVar(name='t[{:d},{:d},{:d}]'.format(end,k,n),vtype='C',ub=M)
            t[start,k,n]=model.addVar(name='t[{:d},{:d},{:d}]'.format(start,k,n),vtype='C',ub=M)
            for i in C:
                t[i,k,n]=model.addVar(name='t[{:d},{:d},{:d}]'.format(i,k,n),vtype='C',lb=a[i],ub=b[i])
            for (i,j) in c.keys():
                x[i,j,k,n]=model.addVar(name='x[{:d},{:d},{:d},{:d}]'.format(i,j,k,n),vtype='B')
    model.setObjective(quicksum(c[i,j]*x[i,j,k,n] for (i,j) in c.keys() for k in range(K) for n in range(N))+quicksum(t[end,k,N-1] for k in range(K)),'minimize')

    # fulfill
    for j in C:
        model.addCons(quicksum(x[i,j,k,n] for k in range(K) for n in range(N) for i in V if (i,j) in c)==1,'fulfill-{:d}'.format(j))

    # start/end
    for k in range(K):
        for n in range(N):
            model.addCons(
                quicksum(x[start,j,k,n] for j in V if (start,j) in c)==1,
                'start-({:d},{:d})'.format(k,n)
            )
    for k in range(K):
        for n in range(N):
            model.addCons(
                quicksum(x[i,end,k,n] for i in V if (i,end) in c)==1,
                'end-({:d},{:d})'.format(k,n)
            )
    # in/out
    for p in C:
        for k in range(K):
            for n in range(N):
                model.addCons(
                    quicksum(x[i,p,k,n] for i in V if (i,p) in c)==
                    quicksum(x[p,j,k,n] for j in V if (p,j) in c),
                    'in/out-({:d},{:d},{:d})'.format(p,k,n)
                )
    # capactiy
    for k in range(K):
        for n in range(N):
            model.addCons(
                quicksum(q[j]*x[i,j,k,n] for (i,j) in c)<=Q[k],
                'capacity-({:d},{:d})'.format(k,n)
            )

    # time dynamic
    for (i,j) in c:
            for k in range(K):
                for n in range(N):
                    model.addCons(
                        t[i,k,n]+s[i]+tau[i,j]-M*(1-x[i,j,k,n])<=t[j,k,n],
                        'time-dynamic-({:d},{:d},{:d},{:d})'.format(i,j,k,n)
                    )

    # repair
    for k in range(K):
        for n in range(N-1):
            model.addCons(
                t[end,k,n]+r[k]*(1-x[start,end,k,n])<=t[start,k,n+1],
                'repair-({:d},{:d})'.format(k,n)
            )
    
    # solve
    model.optimize()
    x_sol={}
    t_sol={}
    for (i,j) in c:
            for k in range(K):
                for n in range(N):
                    key=(k,n)
                    if model.getVal(x[i,j,k,n])>1e-3:
                        if key in x_sol:
                            x_sol[key].append((i,j))
                        else:
                            x_sol[key]=[(i,j)]
    for i in V:
        for k in range(K):
            for n in range(N):
                key=(k,n)
                if model.getVal(t[i,k,n])>1e-3:
                    if key in t_sol:
                        t_sol[key].append((i,model.getVal(t[i,k,n])))
                    else:
                        t_sol[key]=[(i,model.getVal(t[i,k,n]))]
    i=end
    for k in range(K):
            for n in range(N):
                key=(k,n)
                if model.getVal(t[i,k,n])>1e-3:
                    if key in t_sol:
                        t_sol[key].append((i,model.getVal(t[i,k,n])))
                    else:
                        t_sol[key]=[(i,model.getVal(t[i,k,n]))]
    for k,n in x_sol:
        print((k,n),'\t',x_sol[k,n])
    for k,n in t_sol:
        print((k,n),'\t',t_sol[k,n])
                    
    
    