import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as igr
import matplotlib.animation as animation

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
xmin = -6
ymin = xmin
xmax = -xmin
ymax = -xmin
sr = 4 
m1 = 2
m2 = 1
sm = m1 + m2
x1o = (m1 * sr) / sm
x2o = (-m2 * sr) / sm
x, y = [],[]
G = 0.01 #this effects the animation speed because I can't be bothered to tidy that up
#Og = np.sqrt((G * sm)/ (sr**3))
Og = 0
fig, ax = plt.subplots()


sc = ax.scatter(x,y, s=80, marker="o")
plt.xlim(xmin,xmax)
plt.ylim(ymin, ymax)
def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def massDistTerm(x1, y1, x2, y2, ma):
    return ma / distance(x1, y1, x2, y2)

def effPot(xp, yp):
    return -G *  (massDistTerm(x[0], y[0], xp, yp, m1) + massDistTerm(x[1], y[1], xp, yp, m2)  + ((sm / (2 * (sr ** 3))) * (xp**2 + yp**2)))

def gradEffPot(xp, yp):
    dl = xmin / 5000
    return ((effPot(xp + dl,yp) - effPot(xp, yp))/dl, (effPot(xp,yp + dl) - effPot(xp, yp))/dl)

def f(u, t):
    return (u[1], -gradEffPot(u[0, 0], u[0, 1]))

x.append(x1o)
x.append(x2o)
y.append(0)
y.append(0)
delta = 0.025
xgrid = np.arange(xmin, xmax, delta)
ygrid = np.arange(ymin, ymax, delta)
contourvals = np.arange(-0.2, 0, delta/100)
X, Y = np.meshgrid(xgrid, ygrid)
Z = effPot(X,Y)
CS = ax.contour(X, Y, Z, contourvals, colors='k', alpha = 0.3)
xpos = np.linspace(xmin, xmax, 50)
ypos = np.linspace(ymin, ymax, 50)
XPOS, YPOS = np.meshgrid(xpos, ypos)
vx = np.zeros(50)
vy = np.zeros(50)
XVEL, YVEL = np.meshgrid(vx, vy)
delt = 0.4
sc2 = ax.scatter(XPOS, YPOS, s=5)

def animate(i):
    global XVEL
    global YVEL
    global XPOS
    global YPOS
    x.clear()
    y.clear()
    x.append(x1o * np.cos(Og * i))
    x.append(x2o* np.cos(Og * i))
    y.append(x1o * np.sin(Og * i))
    y.append(x2o* np.sin(Og * i))
    Z = effPot(X,Y)
    xp =XPOS + (XVEL * delt)
    yp =YPOS + (YVEL * delt)
    grd = gradEffPot(XPOS,YPOS)
    (xv, yv) = (XVEL - (grd[0] * delt), YVEL- (grd[1] * delt))

    XVEL = xv
    YVEL = yv
    XPOS = xp
    YPOS = yp
    #sc3 = ax.scatter(xp, yp, s=1, c="r", marker="o")
    sc2.set_offsets(np.c_[np.concatenate(XPOS),np.concatenate(YPOS)])
    #CS = ax.contour(X, Y, Z, contourvals, colors='k')
    sc.set_offsets(np.c_[x,y])
    

ani = animation.FuncAnimation(fig, animate, 
                frames=2000, interval=1, repeat=True) 

plt.show()