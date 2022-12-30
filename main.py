import time
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patches as patches


np.random.seed(4012345)


# Creating the Cells Properties
class Cell:
    def __init__(self, X0, Y0, e_scale, b_scale):
        self.X = X0
        self.Y = Y0
        self.RX = 0
        self.RY = 0
        self.DX_background = 0
        self.DY_background = 0
        self.DX_energy = 0
        self.DY_energy = 0
        self.X_History = np.array([self.X])
        self.Y_History = np.array([self.Y])
        self.Death_Count = 1000
        self.Death_Count_Rate = 1
        self.Energy_Level = 5
        self.energy_scale = e_scale
        self.background_scale = b_scale
        self.energy_preference = 10

    def Death_Tick(self):
        self.Death_Count -= self.Death_Count_Rate

    def Update_Background_Position(self, Grid):
        # know what the dimensions of the universe are.
        grid_length = Grid.shape[0]
        # Check to see if the cell is on the border of the universe

        dont_look_left_flag = False  # X
        dont_look_right_flag = False  # X
        dont_look_down_flag = False  # Y
        dont_look_up_flag = False  # Y
        # Cant look to the left flag of X
        if self.X == 0:
            dont_look_down_flag = True

        # Cant look to the right flag of X
        if self.X == grid_length - 1:
            dont_look_up_flag = True

        # Cant look down flag of X
        if self.Y == 0:
            dont_look_left_flag = True

        # Cant look up flag of X
        if self.Y == grid_length - 1:
            dont_look_right_flag = True

        # In the instance where all flags are False

        # In a Corner Cases
        # Top Left Corner     - No down,  No left
        if dont_look_left_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 0
            possible_moves = Grid[self.X:self.X + 2, self.Y:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)


        # Top Right Corner    - No Right, No Down
        elif dont_look_right_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = Grid[self.X:self.X + 2, self.Y - 1:self.Y]
            possible_moves = possible_moves / np.sum(possible_moves)

        # Bottom Left Corner  - No left,  No Up
        elif dont_look_left_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = Grid[self.X - 1:self.X, self.Y:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        # Bottom Right Corner - No Right, No Up
        elif dont_look_right_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = Grid[self.X - 1:self.X, self.Y - 1:self.Y]
            possible_moves = possible_moves / np.sum(possible_moves)

        # This is in the event that the cell is in the left column but not on the top and bottom
        elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = Grid[self.X - 1:self.X + 2, self.Y:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        # This is in the event that the cell is in the right column but not on the top and bottom
        elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = Grid[self.X - 1:self.X + 2, self.Y - 1:self.Y]
            possible_moves = possible_moves / np.sum(possible_moves)

        # This is in the event that the cell is in the first row but not on the left and right corners
        elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = Grid[self.X:self.X + 2, self.Y - 1:self.Y + 2]
            possible_moves = possible_moves / possible_moves.size
        # This is in the event that the cell is in the last row but not on the left and right corners
        elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = Grid[self.X - 1:self.X, self.Y - 1:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        else:
            self.RX = 1
            self.RY = 1
            possible_moves = Grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        Index_Max = np.where(possible_moves == np.max(possible_moves))
        if np.size(Index_Max) > 2:
            IX = Index_Max[0][0]
            IY = Index_Max[1][0]
        else:
            IX = Index_Max[0]
            IY = Index_Max[1]
        self.DX_background = self.background_scale * (IX - self.RX)
        self.DY_background = self.background_scale * (IY - self.RY)

    def Update_Energy_Position(self, energy_grid):
        # know what the dimensions of the universe are.
        grid_length = energy_grid.shape[0]
        # Check to see if the cell is on the border of the universe

        dont_look_left_flag = False  # X
        dont_look_right_flag = False  # X
        dont_look_down_flag = False  # Y
        dont_look_up_flag = False  # Y
        # Cant look to the left flag of X
        if self.X == 0:
            dont_look_down_flag = True

        # Cant look to the right flag of X
        if self.X == grid_length - 1:
            dont_look_up_flag = True

        # Cant look down flag of X
        if self.Y == 0:
            dont_look_left_flag = True

        # Cant look up flag of X
        if self.Y == grid_length - 1:
            dont_look_right_flag = True

        # In the instance where all flags are False

        # In a Corner Cases
        # Top Left Corner     - No down,  No left
        if dont_look_left_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 0
            possible_moves = energy_grid[self.X:self.X + 2, self.Y:self.Y + 2]

        # Top Right Corner    - No Right, No Down
        elif dont_look_right_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = energy_grid[self.X:self.X + 2, self.Y - 1:self.Y]

        # Bottom Left Corner  - No left,  No Up
        elif dont_look_left_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = energy_grid[self.X - 1:self.X, self.Y:self.Y + 2]

        # Bottom Right Corner - No Right, No Up
        elif dont_look_right_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = energy_grid[self.X - 1:self.X, self.Y - 1:self.Y]

        # This is in the event that the cell is in the left column but not on the top and bottom
        elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = energy_grid[self.X - 1:self.X + 2, self.Y:self.Y + 2]

        # This is in the event that the cell is in the right column but not on the top and bottom
        elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = energy_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y]

        # This is in the event that the cell is in the first row but not on the left and right corners
        elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = energy_grid[self.X:self.X + 2, self.Y - 1:self.Y + 2]

        # This is in the event that the cell is in the last row but not on the left and right corners
        elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = energy_grid[self.X - 1:self.X, self.Y - 1:self.Y + 2]

        else:
            self.RX = 1
            self.RY = 1
            possible_moves = energy_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]

        if np.sum(possible_moves) < 1:
            self.DX_energy = 0
            self.DY_energy = 0
        else:
            possible_moves = possible_moves/np.sum(possible_moves)

            Index_Max = np.where(possible_moves == np.max(possible_moves))
            if np.size(Index_Max) > 2:
                IX = Index_Max[0][0]
                IY = Index_Max[1][0]
            else:
                IX = Index_Max[0]
                IY = Index_Max[1]

            self.DX_energy = self.energy_scale * (IX - self.RX)
            self.DY_energy = self.energy_scale * (IY - self.RY)

    def Update_Total_Position(self):
        self.X = self.X + int(self.DX_background) + int(self.DX_energy)
        self.Y = self.Y + int(self.DY_background) + int(self.DY_energy)
        # if self.X < 0:
        #     self.X = 0
        # if self.Y < 0:
        #     self.Y = 0

    def Add_History(self):
        self.X_History = np.hstack((self.X_History, self.X))
        self.Y_History = np.hstack((self.Y_History, self.Y))

    def Update_Death_Count_Rate(self):
        self.Death_Count_Rate += self.Death_Count_Rate

    def Update_Energy_Level(self):
        if abs(self.DX_energy) > 0 or abs(self.DY_energy) > 1:
            self.Energy_Level += 1

        if self.Energy_Level > self.energy_preference:
            self.energy_scale = 0
        else:
            self.energy_scale = 1

# _____________________________________________________

# Creates an N number of cells. Takes in information about the grid which is 'SIZE'
def Create_Cells(N, SIZE):
    Cell_List = []
    for i in range(N):
        Cell_List.append(
            Cell(np.random.randint(0, high=SIZE), np.random.randint(0, high=SIZE), 1, 1))
        # Cell_List.append(
        #     Cell(np.random.randint(0, high=SIZE), np.random.randint(0, high=SIZE), np.random.randint(0, high=3),
        #          np.random.randint(0, high=3)))
    return Cell_List


# Creating the Space the Cells will move within, and the free will feild.
def Create_Grid(SIZE):
    Grid = np.random.randint(0, high=10, size=(SIZE, SIZE))
    return Grid


def Create_Energy_Grid(SIZE):
    Energy_Grid = np.zeros((SIZE, SIZE))
    X0 = int(SIZE / 2)
    Y0 = int(SIZE / 2)
    Energy_Grid[X0 - 2:X0 + 3, Y0 - 2:Y0 + 3] = 5
    Energy_Grid[X0 - 1:X0 + 2, Y0 - 1:Y0 + 2] = 10
    Energy_Grid[X0, Y0] = 25
    return Energy_Grid

grid_size = 50
tf = 200

# The number of Cells that we want in the beginning of the sim
N = 10

Cell_List = Create_Cells(N, grid_size)
Energy_Grid = Create_Energy_Grid(grid_size)
Bar_List = np.zeros((N,tf))
Energy_List = np.zeros((N,tf))
for i in range(tf):
    Grid = Create_Grid(grid_size)
    a = 1
    for j in range(N):
        if Cell_List[j].Death_Count > 0:
            Cell_List[j].Update_Background_Position(Grid)
            Cell_List[j].Update_Energy_Position(Energy_Grid)
            Cell_List[j].Update_Energy_Level()
            Cell_List[j].Update_Total_Position()
            Cell_List[j].Add_History()
            Cell_List[j].Death_Tick()
        else:
            Cell_List[j].Add_History()
        Bar_List[j,i] = Cell_List[j].Death_Count
        Energy_List[j,i] = Cell_List[j].Energy_Level

    print(i)

# Doing the Animation
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
fig.tight_layout(pad=1.0)
# Set the axis limits
ax1.set_xlim(-1, grid_size)
ax1.set_ylim(-1, grid_size)
ax2.set_xlim(-1, N)
ax2.set_ylim(0, 1000)
ax3.set_xlim(-1, N)
ax3.set_ylim(0, 12)
ncol = [c for c in colors.get_named_colors_mapping()]

rect1 = patches.Rectangle((int(grid_size/2)-2, int(grid_size/2)-2), 5, 5, color='gold')
ax1.add_patch(rect1)
rect2 = patches.Rectangle((int(grid_size/2)-1, int(grid_size/2)-1), 3, 3, color='orange')
ax1.add_patch(rect2)
rect3 = patches.Rectangle((int(grid_size/2), int(grid_size/2)), 1, 1, color='darkorange')
ax1.add_patch(rect3)
ax1.grid(which="both")
ax1.minorticks_on()
ax2.set_title('death count')
ax3.set_title('energy levels')


def update(num, lines, Cell_List, bars, Bar_List, energy, Energy_List):
    if num > 5:
        for i in range(N):
            lines[i].set_data(Cell_List[i].X_History[num-5:num], Cell_List[i].Y_History[num-5:num])
    else:
        for i in range(N):
            lines[i].set_data(Cell_List[i].X_History[:num], Cell_List[i].Y_History[:num])
    ax1.set_title('iteration: %d' % num)
    ax1.add_patch(rect1)
    ax1.add_patch(rect2)
    ax1.add_patch(rect3)

    for i in range(N):
        bars[i].set_height(Bar_List[i,num])
        energy[i].set_height(Energy_List[i, num])




# Creates an empty list of the line items that will be used to do the animation
line_list = []
energy_list = []
bar_list = []
for i in range(N):
    g, = ax1.plot([], [], marker='*')
    line_list.append(g)
    g, = ax2.bar(i, 0)
    bar_list.append(g)
    g, = ax3.bar(i, 0)
    energy_list.append(g)

# Create the animation using the update function
anim = animation.FuncAnimation(fig, update, fargs=(line_list, Cell_List, bar_list, Bar_List, energy_list, Energy_List), frames=tf, repeat=True)
# Show the animation
plt.show()
