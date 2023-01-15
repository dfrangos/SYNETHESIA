import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patches as patches

# _____________________________________________________________________________________________________________________
# USER DEFINED VARIABLES
# The random Seed.
np.random.seed(48467)
# This is the size of the universe.
Grid_Size = 20
# This is the amount of time the universe will exist for.
TF = 600
# The number of Cells that we want in the beginning of the simulation. This can change as the simulation progresses.
N = 4
#These are the decay rates experienced for all cells no matter what their personal properties are. This is universal
energy_decay_rate=.05
energy_limit_upper = 9
energy_limit_lower = 2



temperature_decay_rate=.05
temperature_limit_upper = 9
temperature_limit_lower = 2

# _____________________________________________________________________________________________________________________
# Creating the Cells Properties. For now some of these are also user defined.
class Cell:
    def __init__(self, X0, Y0, memory):
        # The coordinates relative to the possible moves array
        self.X = X0
        self.Y = Y0

        # The coordinates relative to the possible moves array
        self.RX = 0
        self.RY = 0

        # A array that is composed of all the global x and y coordinates
        self.X_History = np.array([self.X])
        self.Y_History = np.array([self.Y])

        self.death_count = 1000              # The Death count clock. All cells have it and it never stops ticking
        self.death_count_rate = 1            # The minimum rate at which it ticks. Regardless of what the cell is doing.

        # The Change in x and y coordinates to perform the move related to the QFPM

        self.DX_energy = 0                   # The Change in x and y coordinates to perform the move related to the Sensory model
        self.DY_energy = 0

        self.energy_level = 5                # This is the starting energy level of every cell that is created
        self.energy_preference = 10          # The preferred value that all cells want their level to be at.
        self.energy_experience = False
        self.energy_starved = False
        self.energy_memory = False

        self.temperature_level = 5           # This is the starting energy level of every cell that is created
        self.temperature_preference = 5
        self.temperature_experience = False
        self.temperature_starved = False
        self.temperature_memory = False

        self.DX_background = 0
        self.DY_background = 0
        self.memory_bank = memory            # when the cell is created it will have a memory bank filled with 'memory'. Which, for now, will be nothing.

    def Death_Tick(self):
        # When this is called the Death Count will reduce for that cell by the Death count rate amount.
        # Keep in mind these functions are still within the class.
        self.death_count -= self.death_count_rate

    # When this is called the
    def Update_Background_Position(self, background_grid):
        # know what the dimensions of the universe are.
        grid_length = background_grid.shape[0]
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
            possible_moves = background_grid[self.X:self.X + 2, self.Y:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)


        # Top Right Corner    - No Right, No Down
        elif dont_look_right_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = background_grid[self.X:self.X + 2, self.Y - 1:self.Y]
            possible_moves = possible_moves / np.sum(possible_moves)

        # Bottom Left Corner  - No left,  No Up
        elif dont_look_left_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = background_grid[self.X - 1:self.X, self.Y:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        # Bottom Right Corner - No Right, No Up
        elif dont_look_right_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = background_grid[self.X - 1:self.X, self.Y - 1:self.Y]
            possible_moves = possible_moves / np.sum(possible_moves)

        # This is in the event that the cell is in the left column but not on the top and bottom
        elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = background_grid[self.X - 1:self.X + 2, self.Y:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        # This is in the event that the cell is in the right column but not on the top and bottom
        elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = background_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y]
            possible_moves = possible_moves / np.sum(possible_moves)

        # This is in the event that the cell is in the first row but not on the left and right corners
        elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = background_grid[self.X:self.X + 2, self.Y - 1:self.Y + 2]
            possible_moves = possible_moves / possible_moves.size
        # This is in the event that the cell is in the last row but not on the left and right corners
        elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = background_grid[self.X - 1:self.X, self.Y - 1:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        else:
            self.RX = 1
            self.RY = 1
            possible_moves = background_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]
            possible_moves = possible_moves / np.sum(possible_moves)

        weights=np.ravel(possible_moves) # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
        order=np.arange(0,weights.size,1) # This is created the array which contains the order/indexing of the values of the pos move array
        choice=random.choices(order,weights=weights) # This selects a random value from order based off of the weight
        coordinates=np.unravel_index(choice,possible_moves.shape) # returns the unraveled index of the choice.
        IX,IY=coordinates[0],coordinates[1] # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
        self.DX_background = (IX - self.RX)  # The change in coordinates that cell needs to make is the difference between the target location (IX) and the current location of where the cell is in the relative frame. (RX)
        self.DY_background = (IY - self.RY)  # The change in coordinates that cell needs to make is the difference between the target location (IY) and the current location of where the cell is in the relative frame. (RY)

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
            #Check memory bank to see if data exists for this sense
            self.Check_Memory()
            if self.energy_memory:
                X,Y=self.Access_Memory('energy')
                self.DX_energy=(X-self.X)/np.linalg.norm(X-self.X)
                self.DY_energy=(Y-self.Y)/np.linalg.norm(Y-self.Y)
            else:
                self.DX_energy, self.DY_energy = 0,0

        else:
            possible_moves = possible_moves/np.sum(possible_moves)
            weights=np.ravel(possible_moves) # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
            order=np.arange(0,weights.size,1) # This is created the array which contains the order/indexing of the values of the pos move array
            choice=random.choices(order,weights=weights) # This selects a random value from order based off of the weight
            coordinates=np.unravel_index(choice,possible_moves.shape) # returns the unraveled index of the choice.
            IX,IY=coordinates[0],coordinates[1] # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
            self.DX_energy = (IX - self.RX)  # The change in coordinates that cell needs to make is the difference between the target location (IX) and the current location of where the cell is in the relative frame. (RX)
            self.DY_energy = (IY - self.RY)  # The change in coordinates that cell needs to make is the difference between the target location (IY) and the current location of where the cell is in the relative frame.




        # else:
        #     self.RX = 1
        #     self.RY = 1
        #     possible_moves = energy_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]
        #
        # if np.sum(possible_moves) < 1:
        #     self.DX_energy = 0
        #     self.DY_energy = 0
        # else:
        #     possible_moves = possible_moves/np.sum(possible_moves)
        #
        #     Index_Max = np.where(possible_moves == np.max(possible_moves))
        #     if np.size(Index_Max) > 2:
        #         IX = Index_Max[0][0]
        #         IY = Index_Max[1][0]
        #     else:
        #         IX = Index_Max[0]
        #         IY = Index_Max[1]
        #
        #     self.DX_energy = self.energy_scale * (IX - self.RX)
        #     self.DY_energy = self.energy_scale * (IY - self.RY)

    def Update_Temperature_Position(self,temperature_grid):
        # know what the dimensions of the universe are.
        grid_length = temperature_grid.shape[0]
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
            possible_moves = temperature_grid[self.X:self.X + 2, self.Y:self.Y + 2]

        # Top Right Corner    - No Right, No Down
        elif dont_look_right_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = temperature_grid[self.X:self.X + 2, self.Y - 1:self.Y]

        # Bottom Left Corner  - No left,  No Up
        elif dont_look_left_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = temperature_grid[self.X - 1:self.X, self.Y:self.Y + 2]

        # Bottom Right Corner - No Right, No Up
        elif dont_look_right_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = temperature_grid[self.X - 1:self.X, self.Y - 1:self.Y]

        # This is in the event that the cell is in the left column but not on the top and bottom
        elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = temperature_grid[self.X - 1:self.X + 2, self.Y:self.Y + 2]

        # This is in the event that the cell is in the right column but not on the top and bottom
        elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = temperature_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y]

        # This is in the event that the cell is in the first row but not on the left and right corners
        elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = temperature_grid[self.X:self.X + 2, self.Y - 1:self.Y + 2]

        # This is in the event that the cell is in the last row but not on the left and right corners
        elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = temperature_grid[self.X - 1:self.X, self.Y - 1:self.Y + 2]


        else:
            self.RX = 1
            self.RY = 1
            possible_moves = temperature_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]

        if np.sum(possible_moves) < 1:
            #Check memory bank to see if data exists for this sense
            self.Check_Memory()
            if self.energy_memory:
                X,Y=self.Access_Memory('temperature')
                self.DX_temperature=(X-self.X)/np.linalg.norm(X-self.X)
                self.DY_temperature=(Y-self.Y)/np.linalg.norm(Y-self.Y)
            else:
                self.DX_temperature, self.DY_temperature = 0,0

        else:
            possible_moves = possible_moves/np.sum(possible_moves)
            weights=np.ravel(possible_moves) # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
            order=np.arange(0,weights.size,1) # This is created the array which contains the order/indexing of the values of the pos move array
            choice=random.choices(order,weights=weights) # This selects a random value from order based off of the weight
            coordinates=np.unravel_index(choice,possible_moves.shape) # returns the unraveled index of the choice.
            IX,IY=coordinates[0],coordinates[1] # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
            self.DX_temperature = (IX - self.RX)  # The change in coordinates that cell needs to make is the difference between the target location (IX) and the current location of where the cell is in the relative frame. (RX)
            self.DY_temperature = (IY - self.RY)  # The change in coordinates that cell needs to make is the difference between the target location (IY) and the current location of where the cell is in the relative frame.




        # else:
        #     self.RX = 1
        #     self.RY = 1
        #     possible_moves = energy_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]
        #
        # if np.sum(possible_moves) < 1:
        #     self.DX_energy = 0
        #     self.DY_energy = 0
        # else:
        #     possible_moves = possible_moves/np.sum(possible_moves)
        #
        #     Index_Max = np.where(possible_moves == np.max(possible_moves))
        #     if np.size(Index_Max) > 2:
        #         IX = Index_Max[0][0]
        #         IY = Index_Max[1][0]
        #     else:
        #         IX = Index_Max[0]
        #         IY = Index_Max[1]
        #
        #     self.DX_energy = self.energy_scale * (IX - self.RX)
        #     self.DY_energy = self.energy_scale * (IY - self.RY)

    def Update_Total_Position(self):
        if self.energy_starved==False and self.temperature_starved ==False:
            self.X = self.X + int(self.DX_background)
            self.Y = self.Y + int(self.DY_background)
        elif self.energy_starved and self.temperature_starved == False:
            weights=np.array([.1,.9]) # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
            order=np.arange(0,weights.size,1) # This is created the array which contains the order/indexing of the values of the pos move array
            choice=random.choices(order,weights=weights) # This selects a random value from order based off of the weight
            if choice== 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            else:
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
        elif self.energy_starved==False and self.temperature_starved:
            weights=np.array([.1,.9]) # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
            order=np.arange(0,weights.size,1) # This is created the array which contains the order/indexing of the values of the pos move array
            choice=random.choices(order,weights=weights) # This selects a random value from order based off of the weight
            if choice== 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            else:
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)
        elif self.energy_starved and self.temperature_starved:
            weights=np.array([.1,.45,.45]) # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
            order=np.arange(0,weights.size,1) # This is created the array which contains the order/indexing of the values of the pos move array
            choice=random.choices(order,weights=weights) # This selects a random value from order based off of the weight
            if choice== 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            elif choice== 1:
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
            elif choice== 2:
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)


    def Add_History(self):
        self.X_History = np.hstack((self.X_History, self.X))
        self.Y_History = np.hstack((self.Y_History, self.Y))

    def Update_Death_Count_Rate(self):
        self.death_count_rate += self.death_count_rate

    def Update_Sensory_Level(self, energy_grid, temperature_grid):
        #Updating the Energy Level
        if self.energy_experience:
            self.energy_level+=energy_grid[self.X, self.Y]
            self.energy_level-=energy_grid[self.X, self.Y]/4
        else:
            self.energy_level-=energy_decay_rate

        #Updating the Temperature Level
        if self.temperature_experience:
            self.temperature_level+=temperature_grid[self.X, self.Y]
            self.temperature_level-=temperature_grid[self.X, self.Y]/4
        else:
            self.temperature_level-=temperature_decay_rate

    def Update_Memory(self, energy_grid, temperature_grid):

        #Energy Update
        already_known=False
        if energy_grid[self.X, self.Y]>0:
            for i in range(self.memory_bank.shape[0]):
                if self.memory_bank[i,0,0]>0:
                    if self.memory_bank[i,1,0]==self.X and self.memory_bank[i,2,0]== self.Y:
                        already_known=True
            if already_known==False:
                self.memory_bank=np.roll(self.memory_bank, 1,axis=0)
                self.memory_bank[0,0,0],self.memory_bank[0,1,0],self.memory_bank[0,2,0]=energy_grid[self.X, self.Y],self.X,self.Y
        #Temp Update
        already_known=False
        if temperature_grid[self.X, self.Y]>0:
            for i in range(self.memory_bank.shape[0]):
                if self.memory_bank[i,0,1]>0:
                    if self.memory_bank[i,1,1]==self.X and self.memory_bank[i,2,1]== self.Y:
                        already_known=True
            if already_known==False:
                self.memory_bank=np.roll(self.memory_bank, 1,axis=0)
                self.memory_bank[0,0,1],self.memory_bank[0,1,1],self.memory_bank[0,2,1]=temperature_grid[self.X, self.Y],self.X,self.Y

    def Experience_Check(self,energy_grid, temperature_grid):
        # Check if the cell currently is located on a sensory input.
        #Check the Food

        if energy_grid[self.X,self.Y]>0:
            self.energy_experience=True
        else:
            self.energy_experience=False
        if temperature_grid[self.X,self.Y]>0:
            self.temperature_experience=True
        else:
            self.temperature_experience=False

    def Starvation_Check(self):
        # Check if the cell currently is located on a sensory input.
        #Check the Food

        if self.energy_level<energy_limit_lower:
            self.energy_starved=True

        if self.temperature_level<temperature_limit_lower:
            self.temperature_starved=True


    def Surplus_Check(self):
        # Check if the cell currently is located on a sensory input.
        #Check the Food

        if self.energy_level>energy_limit_upper:
            self.energy_surplus=True

        if self.temperature_level>temperature_limit_upper:
            self.temperature_surplus=True

    def Check_Memory(self):
        if np.any(self.memory_bank):

            if np.any(self.memory_bank[:,:,0]):
                self.energy_memory = True

            if np.any(self.memory_bank[:,:,1]):
                self.temperature_memory = True


    def Access_Memory(self,sense):

        #if the energy memory is true find the x and y of the lowest dist
        if sense=='energy':
            dist_store=np.ones((1,10))*100000
            for i in range(self.memory_bank.shape[0]):
                if self.memory_bank[i,0,0]>0:
                    dist_store[i]=np.linalg.norm(np.array([self.memory_bank[i,1,0]-self.X,self.memory_bank[i,2,0]-self.Y]))
            min_index=np.argmin(dist_store)
            x_sense,y_sense=self.memory_bank[min_index,1,0],self.memory_bank[min_index,2,0]

        #if the temp memory is true find the x and y of the lowest dist
        elif sense=='temperature':
            dist_store=np.ones((1,10))*100000
            for i in range(self.memory_bank.shape[0]):
                if self.memory_bank[i,0,1]>0:
                    dist_store[i]=np.linalg.norm(np.array([self.memory_bank[i,1,1]-self.X,self.memory_bank[i,2,1]-self.Y]))
            min_index=np.argmin(dist_store)
            x_sense,y_sense=self.memory_bank[min_index,1,1],self.memory_bank[min_index,2,1]

        return x_sense, y_sense







# ________________________________________________________________________________________________________

# FUNCTIONS

# Creates an N number of cells. Takes in information about the grid which is 'SIZE'
def Create_Cells(n, grid_size):
    Cell_List = []
    for i in range(N):
        x,y= np.random.randint(0, high=grid_size), np.random.randint(0, high=grid_size)
        e_scale,b_scale= np.random.rand()+1, np.random.rand()+1
        memory_bank= np.zeros((10, 3, 2))
        Cell_List.append(Cell(x, y,memory_bank))
    return Cell_List


# Creating the Space the Cells will move within, and the free will field.
def Create_Grid(grid_size):
    Grid = np.random.randint(0, high=10, size=(grid_size, grid_size))
    return Grid

# Creates an array of an energy grid and defines its location and its values
def Create_Sensory_Grid(grid_size,Sense):
    if Sense=="Energy":
        Source_Value=25
        Sensory_Grid = np.zeros((grid_size, grid_size))
        X0 = int(grid_size / 2)
        Y0 = int(grid_size / 2)
        Sensory_Grid[X0 - 2:X0 + 3, Y0 - 2:Y0 + 3] = Source_Value/3
        Sensory_Grid[X0 - 1:X0 + 2, Y0 - 1:Y0 + 2] = Source_Value/2
        Sensory_Grid[X0, Y0] = Source_Value
    elif Sense=="Temperature":
        Source_Value=25
        Sensory_Grid = np.zeros((grid_size, grid_size))
        X0 = int(grid_size / 3)
        Y0 = int(grid_size / 3)
        Sensory_Grid[X0 - 6:X0 + 7, Y0 - 6:Y0 + 7] = Source_Value/7
        Sensory_Grid[X0 - 5:X0 + 6, Y0 - 5:Y0 + 6] = Source_Value/6
        Sensory_Grid[X0 - 4:X0 + 5, Y0 - 4:Y0 + 5] = Source_Value/5
        Sensory_Grid[X0 - 3:X0 + 4, Y0 - 3:Y0 + 4] = Source_Value/4
        Sensory_Grid[X0 - 2:X0 + 3, Y0 - 2:Y0 + 3] = Source_Value/3
        Sensory_Grid[X0 - 1:X0 + 2, Y0 - 1:Y0 + 2] = Source_Value/2
        Sensory_Grid[X0, Y0] = Source_Value
    return Sensory_Grid




Cell_List = Create_Cells(N, Grid_Size)
Energy_Grid = Create_Sensory_Grid(Grid_Size,"Energy")
Temperature_Grid = Create_Sensory_Grid(Grid_Size,"Temperature")
Death_Count_List = np.zeros((N, TF))
Energy_List = np.zeros((N,TF))
for i in range(TF):
    Grid = Create_Grid(Grid_Size)
    for j in range(N):
        if Cell_List[j].death_count > 0:
            Cell_List[j].Experience_Check(Energy_Grid,Temperature_Grid)
            Cell_List[j].Update_Memory(Energy_Grid, Temperature_Grid)
            Cell_List[j].Update_Sensory_Level(Energy_Grid,Temperature_Grid)
            Cell_List[j].Starvation_Check()
            if Cell_List[j].energy_starved or Cell_List[j].temperature_starved:
                Cell_List[j].Surplus_Check()

            #Movement Models
            Cell_List[j].Update_Background_Position(Grid)
            if Cell_List[j].energy_starved:
                Cell_List[j].Update_Energy_Position(Energy_Grid)
            if Cell_List[j].temperature_starved:
                Cell_List[j].Update_Temperature_Position(Temperature_Grid)
            Cell_List[j].Update_Total_Position()

            Cell_List[j].Add_History()
            Cell_List[j].Death_Tick()
        else:
            Cell_List[j].Add_History()
        Death_Count_List[j, i] = Cell_List[j].death_count
        Energy_List[j,i] = Cell_List[j].energy_level
    print(i)



# THIS IS ALL ANIMATION AND PLOTTING

# ________________________________________________________________________________________________________
# Doing the Animation
idx_cells = np.asarray([i for i in range(N)])
fig, axd = plt.subplot_mosaic([['upper left', 'grid'],
                               ['lower left', 'grid']],
                              figsize=(5.5, 3.5))
plt.tight_layout(pad=2)
ax1, ax2, ax3 = axd["grid"], axd["upper left"], axd["lower left"]
# Set the axis limits
ax1.set_xlim(-1, Grid_Size)
ax1.set_ylim(-1, Grid_Size)
ax2.set_xlim(-1, N)
ax2.set_ylim(0, np.max(Death_Count_List))
ax3.set_xlim(-1, N)
ax3.set_ylim(0, np.max(Energy_List))
ax3.set_xticks(idx_cells)
ax2.set_xticks(idx_cells)
prefs = []
for i in range(N):
    prefs.append(str(Cell_List[i].energy_preference))
xlabels = []
for i in range(N):
    xlabels.append(f'E pref = {prefs[i]}')
ax3.set_xticklabels(xlabels, rotation=50)

ncol = [c for c in colors.get_named_colors_mapping()]

#This creates the rectangles for the energy grid
E_rect1 = patches.Rectangle((int(Grid_Size / 2) - 2, int(Grid_Size / 2) - 2), 5, 5, color='gold')
ax1.add_patch(E_rect1)
E_rect2 = patches.Rectangle((int(Grid_Size / 2) - 1, int(Grid_Size / 2) - 1), 3, 3, color='orange')
ax1.add_patch(E_rect2)
E_rect3 = patches.Rectangle((int(Grid_Size / 2), int(Grid_Size / 2)), 1, 1, color='darkorange')
ax1.add_patch(E_rect3)
#This creates the rectangles for the Temperature grid
T_rect1 = patches.Rectangle((int(Grid_Size / 3) - 6, int(Grid_Size / 3) - 6), 13, 13, color=(1, .2, .2))
ax1.add_patch(T_rect1)
T_rect2 = patches.Rectangle((int(Grid_Size / 3) - 5, int(Grid_Size / 3) -5), 11, 11, color=(.9, .2, .2))
ax1.add_patch(T_rect2)
T_rect3 = patches.Rectangle((int(Grid_Size / 3)-4, int(Grid_Size / 3)-4), 9, 9, color=(.8, .2, .2))
ax1.add_patch(T_rect3)
T_rect4 = patches.Rectangle((int(Grid_Size / 3) - 3, int(Grid_Size / 3) - 3), 7, 7, color=(.7, .2, .2))
ax1.add_patch(T_rect4)
T_rect5 = patches.Rectangle((int(Grid_Size / 3) - 2, int(Grid_Size / 3) - 2), 5, 5, color=(.6, .2, .2))
ax1.add_patch(T_rect5)
T_rect6 = patches.Rectangle((int(Grid_Size / 3)-1, int(Grid_Size / 3)-1), 3, 3, color=(.5, .2, .2))
ax1.add_patch(T_rect6)
T_rect7 = patches.Rectangle((int(Grid_Size / 3), int(Grid_Size / 3)), 1, 1, color=(.4, .2, .2))
ax1.add_patch(T_rect7)

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
    #This adds in the Energy rectangle every loop so they dont disapear.
    ax1.add_patch(E_rect1)
    ax1.add_patch(E_rect2)
    ax1.add_patch(E_rect3)
    #This adds in the Temperature rectangles every loop so they dont disapear.
    ax1.add_patch(T_rect1)
    ax1.add_patch(T_rect2)
    ax1.add_patch(T_rect3)
    ax1.add_patch(T_rect4)
    ax1.add_patch(T_rect5)
    ax1.add_patch(T_rect6)
    ax1.add_patch(T_rect7)

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
anim = animation.FuncAnimation(fig, update, fargs=(line_list, Cell_List, bar_list, Death_Count_List, energy_list, Energy_List), interval=5, frames=TF, repeat=True, cache_frame_data=False)
# Show the animation

# for j in range(N):
#     ax1.plot(Cell_List[j].X_History,Cell_List[j].Y_History, marker='*')

plt.show()

