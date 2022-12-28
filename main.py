import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#np.random.seed(30)


#Creating the Cells Properties
class Cell:
    def __init__(self,X0,Y0):
        self.X = X0
        self.Y = Y0
        self.RX = 0
        self.RY = 0
        self.X_History = np.array([self.X])
        self.Y_History = np.array([self.Y])
        self.Death_Count =500
        self.Death_Count_Rate =1
        self.Energy_Level=5


    def Death_Tick(self):
        self.Death_Count -= self.Death_Count_Rate
#_____________________________________________________

#Creates an N number of cells. Takes in information about the grid which is 'SIZE'
def Create_Cells(N,SIZE):
    Cell_List=[]
    for i in range(N):
        Cell_List.append(Cell(np.random.randint(0,high=SIZE),np.random.randint(0,high=SIZE)))
    return Cell_List


#Creating the Space the Cells will move within, and the free will feild.
def Create_Grid(SIZE):
    Grid=np.random.randint(0,high=100,size=(SIZE,SIZE))
    return Grid

def Create_Energy_Grid(SIZE):
    Energy_Grid=np.zeros((SIZE,SIZE))
    X0=int(SIZE/2)
    Y0=int(SIZE/2)
    Energy_Grid[X0-2:X0+3,Y0-2:Y0+3]=1
    Energy_Grid[X0-1:X0+2,Y0-1:Y0+2]=2
    Energy_Grid[X0,Y0]=3
    return Energy_Grid

EG=Create_Energy_Grid(20)


def Update_Position(Coordinates,Grid):
    #know what the dimensions of the universe are.
    grid_length=Grid.shape[0]
    #Check to see if the cell is on the border of the universe

    dont_look_left_flag =False  #X
    dont_look_right_flag=False  #X
    dont_look_down_flag =False  #Y
    dont_look_up_flag   =False  #Y
    #Cant look to the left flag of X
    if Coordinates.X == 0:
        dont_look_down_flag=True

    #Cant look to the right flag of X
    if Coordinates.X == grid_length-1:
        dont_look_up_flag=True

    #Cant look down flag of X
    if Coordinates.Y == 0:
        dont_look_left_flag=True

    #Cant look up flag of X
    if Coordinates.Y == grid_length-1:
        dont_look_right_flag=True

    #In the instance where all flags are False

    #In a Corner Cases
    #Top Left Corner     - No down,  No left
    if dont_look_left_flag and dont_look_down_flag:
        Coordinates.RX=0
        Coordinates.RY=0
        possible_moves=Grid[Coordinates.X:Coordinates.X+2,Coordinates.Y:Coordinates.Y+2]
        possible_moves=possible_moves/np.sum(possible_moves)


    #Top Right Corner    - No Right, No Down
    elif dont_look_right_flag and dont_look_down_flag:
        Coordinates.RX=0
        Coordinates.RY=1
        possible_moves=Grid[Coordinates.X:Coordinates.X+2,Coordinates.Y-1:Coordinates.Y]
        possible_moves=possible_moves/np.sum(possible_moves)

    #Bottom Left Corner  - No left,  No Up
    elif dont_look_left_flag and dont_look_up_flag:
        Coordinates.RX=1
        Coordinates.RY=0
        possible_moves=Grid[Coordinates.X-1:Coordinates.X,Coordinates.Y:Coordinates.Y+2]
        possible_moves=possible_moves/np.sum(possible_moves)

    #Bottom Right Corner - No Right, No Up
    elif dont_look_right_flag and dont_look_up_flag:
        Coordinates.RX=1
        Coordinates.RY=1
        possible_moves=Grid[Coordinates.X-1:Coordinates.X,Coordinates.Y-1:Coordinates.Y]
        possible_moves=possible_moves/np.sum(possible_moves)

    #This is in the event that the cell is in the left column but not on the top and bottom
    elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
        Coordinates.RX=1
        Coordinates.RY=0
        possible_moves=Grid[Coordinates.X-1:Coordinates.X+2,Coordinates.Y:Coordinates.Y+2]
        possible_moves=possible_moves/np.sum(possible_moves)

    #This is in the event that the cell is in the right column but not on the top and bottom
    elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
        Coordinates.RX=1
        Coordinates.RY=1
        possible_moves=Grid[Coordinates.X-1:Coordinates.X+2,Coordinates.Y-1:Coordinates.Y]
        possible_moves=possible_moves/np.sum(possible_moves)

    #This is in the event that the cell is in the first row but not on the left and right corners
    elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
        Coordinates.RX=0
        Coordinates.RY=1
        possible_moves=Grid[Coordinates.X:Coordinates.X+2,Coordinates.Y-1:Coordinates.Y+2]
        possible_moves=possible_moves/possible_moves.size
     #This is in the event that the cell is in the last row but not on the left and right corners
    elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
        Coordinates.RX=1
        Coordinates.RY=1
        possible_moves=Grid[Coordinates.X-1:Coordinates.X,Coordinates.Y-1:Coordinates.Y+2]
        possible_moves=possible_moves/np.sum(possible_moves)

    else:
        Coordinates.RX=1
        Coordinates.RY=1
        possible_moves=Grid[Coordinates.X-1:Coordinates.X+2,Coordinates.Y-1:Coordinates.Y+2]
        possible_moves=possible_moves/np.sum(possible_moves)


    Index_Max=np.where(possible_moves==np.max(possible_moves))
    if np.size(Index_Max)>2:
        IX=Index_Max[0][0]
        IY=Index_Max[1][0]
    else:
        IX=Index_Max[0]
        IY=Index_Max[1]
    DX=IX-Coordinates.RX
    DY=IY-Coordinates.RY
    Coordinates.X=Coordinates.X+int(DX)
    Coordinates.Y=Coordinates.Y+int(DY)



    return Coordinates

grid_size=25
time=100
C1 = Cell(int(grid_size/2),int(grid_size/2))
C2 = Cell(int(0),int(0))
# print(C1.X)
# print(C1.Y)

# Create a figure and axis
fig, ax = plt.subplots()

# Set the axis limits
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

X1_Position_Data=np.array([C1.X])
Y1_Position_Data=np.array([C1.Y])

X2_Position_Data=np.array([C2.X])
Y2_Position_Data=np.array([C2.Y])



#The number of Cells that we want in the begining of the sim
N=10

Cell_List=Create_Cells(N,grid_size)

for i in range(time):
    Grid=Create_Grid(grid_size)
    for j in range(N):

        Cell_List[j]=Update_Position(Cell_List[j],Grid)

        Cell_List[j].X_History=np.hstack((Cell_List[j].X_History,Cell_List[j].X))

        Cell_List[j].Y_History=np.hstack((Cell_List[j].Y_History,Cell_List[j].Y))

    print(i)

    # C1=Update_Position(C1,Grid)
    # C2=Update_Position(C2,Grid)
    # X1_Position_Data=np.hstack((X1_Position_Data,C1.X))
    # Y1_Position_Data=np.hstack((Y1_Position_Data,C1.Y))
    # X2_Position_Data=np.hstack((X2_Position_Data,C2.X))
    # Y2_Position_Data=np.hstack((Y2_Position_Data,C2.Y))

    # ax.plot(X1_Position_Data,Y1_Position_Data,marker='^')

    #Update the Position of the Cell
    #Call update position
    # pass in C1 and return C1 update it internally in the func
# ax.plot(X_Position_Data,Y_Position_Data)
# plt.show()
# print(Grid)
# ax.plot(X1_Position_Data[0],Y1_Position_Data[0],marker='*')
# ax.plot(X1_Position_Data[-1],Y1_Position_Data[-1],marker='s')
#plt.show()
# Doing the Animation
fig2, ax = plt.subplots()
# Set the axis limits
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
line1, = ax.plot([], [], lw=2,color='k',marker='^')
line2, = ax.plot([], [], lw=2, color='b',marker='*')



def update(num,lines,Cell_List):
    #Create the list of line items
    for i in range(N):
        lines[i].set_data(Cell_List[i].X_History[:num],Cell_List[i].Y_History[:num])

#Creates an empty list of the line items that will be used to do the animation
line_list=[]

for i in range(N):
    line_list.append(ax.plot(Cell_List[i].X_History[0], Cell_List[i].Y_History[0], lw=2,color='k',marker='^'))


    # Update the data of the line plot
    # if num<10:
    #     line1.set_data(X1_Position_Data[:num], Y1_Position_Data[:num])
    #     line2.set_data(X2_Position_Data[:num], Y2_Position_Data[:num])
    # else:
    #     line1.set_data(X1_Position_Data[num-10:num],Y1_Position_Data[num-10:num])
    #     line2.set_data(X2_Position_Data[num-10:num],Y2_Position_Data[num-10:num])
# Create the animation using the update function
anim = animation.FuncAnimation(fig2, update, fargs=(line_list,Cell_List), frames=range(100), repeat=True)
# Show the animation
plt.grid(which="both")
plt.minorticks_on()
plt.show()


