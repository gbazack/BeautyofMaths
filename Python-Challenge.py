################ Mathematical Function Challenge########################
###Provide 5 functions to show High School students the bueaty of Mathematics
### Python 2 or 3 can be used to run ########
#### Import useful Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spi
from mpl_toolkits.mplot3d import Axes3D


#### Function 1 ####
# Produce a binary code which is divisible by the the lenght of the code
#Part A: Count the digit of a binary number
def CountNum(v):
    l=[]; i,g=0,0
    while (v!=0):
        v=int(v/10)
        l.append (v%10)
        i=i+1
    for j in range(len(l)):
        if (l[j]==1):
            g=g+1
    return g

#Part B: Method 1 produces binary numbers of type 111....100...0
def FinNum1(k):# example 111000
    i,p,s=1,10,0
    while (i<10000):
        s=p+s*10
        p=p*100
        if (s%k==0 and CountNum(s)==k):
            return s
        i=i+1
    return "not found"

#Part C: Method 2 produces binary numbers of type 1010....100...0
def FinNum2(k):#example 10101000
   i,p,s=1,10,0
   while (i<10000):
        s=p+s*10
        p=p*1000
        if (s%k==0 and CountNum(s)==k):
            return s
        i=i+1
   return "not found"

#### Function 2 ####
# Produce a bueatiful shade of colours

N=100
y=np.linspace(-5,5,N)
M=2*N-1
x=np.linspace(-10,10,M)
def d(a,b): #funtion calculates distance from origin
    return np.sqrt(a**2+b**2)

#Plot of the matrix formed by calling function above
A=np.zeros((M,N))
for i in range(M):
    for j in range(N):
        A[i,j]=d(x[i],y[j]) #calling function

plt.figure('Figure 1')
plt.imshow(A.T)
plt.xlabel('A colunms') 
plt.ylabel('A lines')
plt.ion()
plt.show()


#### Function 3 ####
# Produce a bueatiful flower 

def wierd(a,b): 
    C=a+b*(1j)
    z=[C]
    i=0
    while not (np.abs(z[i])>2 or i==49):
        z.append((z[i]*z[i])+C)
        i+=1
    return i                                     


#------------------------------------------------------------
#Plot the matrix formed by calling the function above with N different
N=100
y=np.linspace(-1.0,1.0,N)
x=np.linspace(-1.5,0.5,N)
A=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        A[i,j]=wierd(x[i],y[j]) 
plt.figure('Figure 2')
plt.imshow(A.T)
plt.ylabel('A lines')
plt.xlabel('A columns')
plt.ion()
plt.show()


#### Function 4 ####
# 3D visual representation of mask


#I decieded to use both methods explained by the teacher, I gave names to the plot depending on the method; Constructed (calculations with a program), Odient (calculations by pakage odeint)

N=10000
t=np.linspace(0,50,N)
def f(z,t,a,b,c): #funtion constructed with different constants (a,b,c)
    return np.array([a*(z[1]-z[0]),z[0]*(b-z[2])-z[1],z[0]*z[1]-c*z[2]])
    
A=np.ones((N,3)) #array created for origin (1,1,1)
B=np.ones((N,3)) #array created for origin (1.001,1,1)
B[0,0]=1.001
P=np.array([10,22,8./3]) #array of constants (a,b,c)
for i in range(1,N): #calculations of the value to build the constructed plot
    A[i,:]=A[i-1,:]+f(A[i-1,:],t[i-1],P[0],P[1],P[2])*(t[i]-t[i-1]) # for (1,1,1)
    B[i,:]=B[i-1,:]+f(B[i-1,:],t[i-1],P[0],P[1],P[2])*(t[i]-t[i-1]) # for (1.001,1,1)
Zspi=spi.odeint(f,A[0,:],t,args=(P[0],P[1],P[2])) #calculation of the odeint array for origin (1,1,1)
Zspi1=spi.odeint(f,B[0,:],t,args=(P[0],P[1],P[2]))#calculation of the odeint array for origin (1.001,1,1)
fig=plt.figure('Figure 1')
ax1=fig.add_subplot(111,projection='3d')
ax1.plot(A[:,0],A[:,1],A[:,2],label='Constructed Plot (1,1,1)') #Constructed plot for origin (1,1,1)
ax1.plot(B[:,0],B[:,1],B[:,2],label='Constructed Plot (1.001,1,1)') #constructed plot for origin (1.001,1,1)
ax1.plot(Zspi[:,0],Zspi[:,1],Zspi[:,2], color='red', label='Odeint Plot (1,1,1)')
ax1.plot(Zspi1[:,0],Zspi1[:,1],Zspi1[:,2], color='yellow', label='Odeint Plot (1.001,1,1)') #the two plots above are the odeint plots for respective origins
plt.legend()
plt.ion()
plt.show()

#--------------------------------------------------------------------------------
#Here the same idea repeated but the constants have changed
A=np.ones((N,3))
B=np.ones((N,3))
B[0,0]=1.001
P=np.array([10,24,8./3]) #contant b has been changed
for i in range(1,N):
    A[i,:]=A[i-1,:]+f(A[i-1,:],t[i-1],P[0],P[1],P[2])*(t[i]-t[i-1])
    B[i,:]=B[i-1,:]+f(B[i-1,:],t[i-1],P[0],P[1],P[2])*(t[i]-t[i-1])
Zspi=spi.odeint(f,A[0,:],t,args=(P[0],P[1],P[2]))
Zspi1=spi.odeint(f,B[0,:],t,args=(P[0],P[1],P[2]))
fig=plt.figure('Figure 2')
ax1=fig.add_subplot(111,projection='3d')
ax1.plot(A[:,0],A[:,1],A[:,2],label='Constructed Plot (1,1,1)')
ax1.plot(B[:,0],B[:,1],B[:,2],label='Constructed Plot (1.001,1,1)')
ax1.plot(Zspi[:,0],Zspi[:,1],Zspi[:,2], color='red', label='Odient Plot (1,1,1)')
ax1.plot(Zspi1[:,0],Zspi1[:,1],Zspi1[:,2], color='yellow', label='Odient Plot (1.001,1,1)')
plt.legend()
plt.ion()
plt.show()

#---------------------------------------------------------------------------------
#Here the same idea repeated but the constants have changed
A=np.ones((N,3))
B=np.ones((N,3))
B[0,0]=1.001
P=np.array([10,25,8./3]) #constant b has changed
for i in range(1,N):
    A[i,:]=A[i-1,:]+f(A[i-1,:],t[i-1],P[0],P[1],P[2])*(t[i]-t[i-1])
    B[i,:]=B[i-1,:]+f(B[i-1,:],t[i-1],P[0],P[1],P[2])*(t[i]-t[i-1])
Zspi=spi.odeint(f,A[0,:],t,args=(P[0],P[1],P[2]))
Zspi1=spi.odeint(f,B[0,:],t,args=(P[0],P[1],P[2]))
fig=plt.figure('Figure 3')
ax1=fig.add_subplot(111,projection='3d')
ax1.plot(A[:,0],A[:,1],A[:,2],label='Constructed Plot (1,1,1)')
ax1.plot(B[:,0],B[:,1],B[:,2],label='Constructed Plot (1.001,1,1)')
ax1.plot(Zspi[:,0],Zspi[:,1],Zspi[:,2], color='red', label='Odient Plot (1,1,1)')
ax1.plot(Zspi1[:,0],Zspi1[:,1],Zspi1[:,2], color='yellow', label='Odient Plot (1.001,1,1)')
plt.legend()
plt.ion()
plt.show()

#--------------------------------------------------------------------------------
#CONCLUSION: When we compare the constructed plot to the Odeint plot we realise there is a great diffence in figure 1 but in figure 3 they are submerged.
#The change in our origin brings a slight shift in the plots. As we change our constants the plot get more submerge


#### Function 5 ####
# Plot Pearsons Correlation and find an aproximation of Pi with diagarm.


#Pearson Correlation of A[i,0] and A[i,1]

A=np.random.randn(1000,3)+50 #make random array and add 50 to each element
n,m=np.shape(A)
Ai0=np.zeros(n)
Ai1=np.zeros(n)
for i in range(n): #selecting the first two colums of interest and placing them as separate arrays 
    Ai0[i]=A[i,0]
    Ai1[i]=A[i,1]
PearC=np.corrcoef(Ai0,Ai1)
print PearC
#Ploting of Ai0 and Ai1
plt.figure('Pearson Correlation of Ai0 and Ai1')
plt.plot(Ai0,Ai1,label= 'Pearson Coeff Ai0 and Ai1',marker='.',ls='None')
plt.legend()
plt.ion()
plt.show()


Bi=np.zeros(n)
Ci=np.zeros(n)
for i in range(n):  #Creating two new arrays from A
    Bi[i]=Ai0[i]/A[i,2]
    Ci[i]=Ai1[i]/A[i,2]
PearC1=np.corrcoef(Bi,Ci)
print PearC1
#Plotting Bi and Ci 
plt.figure('Pearson Correlation of Bi and Ci')
plt.plot(Bi,Ci,label= 'Pearson Coeff Bi and Ci',marker='.',ls='None')
plt.legend()
plt.ion()
plt.show()
#the value of correlation
ValCorr=np.mean(PearC1)
print 'The value of the correlation is %f',ValCorr

# Function that repeats the random generation of an integer and stops at finding 7

def RandGen7(x=0):
    RG=1
    i=0
    while RG%7>0: #when the output of the random integer is not 7 execute
        RG=np.random.randint(0,11)
        i=i+1 #counter for the random call
    return i

RandP=[] #Initialize an empty list
RP=0
i=0
while i<1000:
    RP=RandGen7() #Call the function which generates the integers in Q2.i
    RandP.append(RP)
    i=i+1
RandPA=np.array(RandP) #Type conversion from list to array
MeanR=np.mean(RandPA)
print 'The average is %f',MeanR
StdR=np.std(RandPA)
print 'The standard deviation %f',StdR

#---------------------------------------------------------------------------------


def f(x): #function described in (1)
    if x%2==0:
        y=x/2
    if x%2==1:
        y=3*x+1
    return y

def ST(s): #Function of the Stopping time
    ai=[s]
    i=0
    while not(s==1):
        s=f(ai[i])
        ai.append(s)
        i=i+1
    return i
#---------------------------------------------------------
#Plotting the Stopping time in a range
Yp=np.zeros(200)
Xp=np.arange(1,201)
for k in range(len(Yp)):
    Yp[k]=ST(Xp[k])  #calling Stopping time function
plt.figure('Stopping Time')
plt.plot(Xp,Yp,label= 'Yp against Xp')
plt.legend()
plt.ion()
plt.show()  

#----------------------------------------------------------
#Comparing the values of Stopping time to find maximum point.
for i in range(199):
    if Yp[i]>Yp[i+1]:
        p=Yp[i]
        Yp[i]=Yp[i+1]
        Yp[i+1]=p
for i in range(199):
    if ST(Xp[i])==Yp[199]:
        print 'the Maximum point;',Xp[i],Yp[199]



#--------------------------------------------------------------------------------

#Estimating Pi 
N=0
C=0
while N<100:
    x=np.random.uniform(-1,1)
    y=np.random.uniform(-1,1)
    d=np.sqrt(x**2+y**2)
    if d<1:
        C+=1 #counting the values of d<1
    N+=1
print C



# Ratio
ratio=4.0*C/N
print ratio



#Testing for N>100
N1=0
C1=0
while N1<100000:
    x1=np.random.uniform(-1,1)
    y1=np.random.uniform(-1,1)
    d1=np.sqrt(x1**2+y1**2)
    if d1<1:
        C1+=1
    N1+=1
print 4.0*C1/N1


















