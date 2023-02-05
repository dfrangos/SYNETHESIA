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
random.seed(10)
# This is the size of the universe.
Grid_Size = 40
# This is the amount of time the universe will exist for.
TF = 600
# The number of Cells that we want in the beginning of the simulation. This can change as the simulation progresses.
N = 6

# These are the decay rates experienced for all cells no matter what their personal properties are. This is universal
upper_sensory_cap = 10
lower_sensory_cap = 0

food_decay_rate = .05
food_upper_harm = 8
food_lower_harm = 2
food_x_source = int(Grid_Size / 1.5)
food_y_source = int(Grid_Size / 6)

temperature_decay_rate = .05
temperature_upper_harm = 8
temperature_lower_harm = 2
temperature_x_source = int(Grid_Size / 3)
temperature_y_source = int(Grid_Size / 3)

water_decay_rate = .05
water_upper_harm = 8
water_lower_harm = 2
water_x_source = int(Grid_Size / 3)
water_y_source = int(Grid_Size / 3) + 14

weight_b_s = np.array([0.01, 0.99])
weight_b_s_s = np.array([0.01, 0.495, 0.495])
weight_eq_3 = np.array([.01, .33, .33])
source_value = 5

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
        self.death_count = 1000  # The Death count clock. All cells have it and it never stops ticking
        self.death_count_rate = 1  # The minimum rate at which it ticks. Regardless of what the cell is doing.

        self.DX_energy = 0  # The Change in x and y coordinates to perform the move related to the Sensory model
        self.DY_energy = 0
        self.energy_level = 5  # This is the starting energy level of every cell that is created
        self.energy_preference = 5  # The preferred value that all cells want their level to be at.
        self.energy_experience = False
        self.energy_starved = False
        self.energy_surplus = False
        self.energy_memory = False

        self.DX_temperature = 0  # The Change in x and y coordinates to perform the move related to the Sensory model
        self.DY_temperature = 0
        self.temperature_level = 5  # This is the starting energy level of every cell that is created
        self.temperature_preference = 5
        self.temperature_experience = False
        self.temperature_starved = False
        self.temperature_surplus = False
        self.temperature_memory = False

        self.DX_water = 0  # The Change in x and y coordinates to perform the move related to the Sensory model
        self.DY_water = 0
        self.water_level = 5  # This is the starting energy level of every cell that is created
        self.water_preference = 5
        self.water_experience = False
        self.water_starved = False
        self.water_surplus = False
        self.water_memory = False

        self.DX_background = 0
        self.DY_background = 0
        self.memory_bank = memory  # when the cell is created it will have a memory bank filled with 'memory'. Which, for now, will be nothing.

    def Death_Tick(self):
        # When this is called the Death Count will reduce for that cell by the Death count rate amount.
        # Keep in mind these functions are still within the class.
        self.death_count -= self.death_count_rate

        # Check if any of the cells are in a surplus or starvation zone
        if self.energy_level < food_lower_harm:
            self.death_count -= food_lower_harm - self.energy_level
        elif self.energy_level > food_upper_harm:
            self.death_count -= self.energy_level - food_upper_harm

        if self.temperature_level < temperature_lower_harm:
            self.death_count -= temperature_lower_harm-self.temperature_level
        elif self.temperature_level > temperature_upper_harm:
            self.death_count -= self.temperature_level-temperature_upper_harm

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

        weights = np.ravel(possible_moves)  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
        order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
        choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
        coordinates = np.unravel_index(choice, possible_moves.shape)  # returns the unraveled index of the choice.
        IX, IY = coordinates[0], coordinates[
            1]  # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
        self.DX_background = (
                IX - self.RX)  # The change in coordinates that cell needs to make is the difference between the target location (IX) and the current location of where the cell is in the relative frame. (RX)
        self.DY_background = (
                IY - self.RY)  # The change in coordinates that cell needs to make is the difference between the target location (IY) and the current location of where the cell is in the relative frame. (RY)

    def Update_Sensory_Position(self, sensory_grid, sense):
        # know what the dimensions of the universe are.
        grid_length = sensory_grid.shape[0]
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
            possible_moves = sensory_grid[self.X:self.X + 2, self.Y:self.Y + 2]

        # Top Right Corner    - No Right, No Down
        elif dont_look_right_flag and dont_look_down_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = sensory_grid[self.X:self.X + 2, self.Y - 1:self.Y]

        # Bottom Left Corner  - No left,  No Up
        elif dont_look_left_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = sensory_grid[self.X - 1:self.X, self.Y:self.Y + 2]

        # Bottom Right Corner - No Right, No Up
        elif dont_look_right_flag and dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = sensory_grid[self.X - 1:self.X, self.Y - 1:self.Y]

        # This is in the event that the cell is in the left column but not on the top and bottom
        elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 0
            possible_moves = sensory_grid[self.X - 1:self.X + 2, self.Y:self.Y + 2]

        # This is in the event that the cell is in the right column but not on the top and bottom
        elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = sensory_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y]

        # This is in the event that the cell is in the first row but not on the left and right corners
        elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 0
            self.RY = 1
            possible_moves = sensory_grid[self.X:self.X + 2, self.Y - 1:self.Y + 2]

        # This is in the event that the cell is in the last row but not on the left and right corners
        elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
            self.RX = 1
            self.RY = 1
            possible_moves = sensory_grid[self.X - 1:self.X, self.Y - 1:self.Y + 2]

        else:
            self.RX = 1
            self.RY = 1
            possible_moves = sensory_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]

        if sense == 'energy':
            if np.sum(possible_moves) < 1:
                # Check memory bank to see if data exists for this sense
                self.Check_Memory()
                if self.energy_memory:
                    X, Y = self.Access_Memory(sense)
                    self.DX_energy = np.sign(X-self.X)
                    self.DY_energy = np.sign(Y-self.Y)
                else:
                    self.DX_energy, self.DY_energy = 0, 0

            else:
                possible_moves = possible_moves / np.sum(possible_moves)
                weights = np.ravel(possible_moves)
                order = np.arange(0, weights.size, 1)
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                coordinates = np.unravel_index(choice, possible_moves.shape)  # returns the unraveled index of the choice.
                IX, IY = coordinates[0], coordinates[1]  # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
                self.DX_energy = (IX - self.RX)
                self.DY_energy = (IY - self.RY)
        elif sense == 'temperature':
            if np.sum(possible_moves) < 1:
                # Check memory bank to see if data exists for this sense
                self.Check_Memory()
                if self.energy_memory:
                    X, Y = self.Access_Memory(sense)
                    self.DX_temperature = np.sign(X-self.X)
                    self.DY_temperature = np.sign(Y-self.Y)
                else:
                    self.DX_temperature, self.DY_temperature = 0, 0

            else:
                possible_moves = possible_moves / np.sum(possible_moves)
                weights = np.ravel(possible_moves)
                order = np.arange(0, weights.size, 1)
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                coordinates = np.unravel_index(choice, possible_moves.shape)  # returns the unraveled index of the choice.
                IX, IY = coordinates[0], coordinates[1]  # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
                self.DX_temperature = (IX - self.RX)
                self.DY_temperature = (IY - self.RY)
        elif sense == 'water':
            if np.sum(possible_moves) < 1:
                # Check memory bank to see if data exists for this sense
                self.Check_Memory()
                if self.energy_memory:
                    X, Y = self.Access_Memory(sense)
                    self.DX_water = np.sign(X-self.X)
                    self.DY_water = np.sign(Y-self.Y)
                else:
                    self.DX_water, self.DY_water = 0, 0

            else:
                possible_moves = possible_moves / np.sum(possible_moves)
                weights = np.ravel(possible_moves)
                order = np.arange(0, weights.size, 1)
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                coordinates = np.unravel_index(choice, possible_moves.shape)  # returns the unraveled index of the choice.
                IX, IY = coordinates[0], coordinates[1]  # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
                self.DX_water = (IX - self.RX)
                self.DY_water = (IY - self.RY)

    # def Update_Temperature_Position(self, temperature_grid):
    #     # know what the dimensions of the universe are.
    #     grid_length = temperature_grid.shape[0]
    #     # Check to see if the cell is on the border of the universe
    #
    #     dont_look_left_flag = False  # X
    #     dont_look_right_flag = False  # X
    #     dont_look_down_flag = False  # Y
    #     dont_look_up_flag = False  # Y
    #     # Cant look to the left flag of X
    #     if self.X == 0:
    #         dont_look_down_flag = True
    #
    #     # Cant look to the right flag of X
    #     if self.X == grid_length - 1:
    #         dont_look_up_flag = True
    #
    #     # Cant look down flag of X
    #     if self.Y == 0:
    #         dont_look_left_flag = True
    #
    #     # Cant look up flag of X
    #     if self.Y == grid_length - 1:
    #         dont_look_right_flag = True
    #
    #     # In the instance where all flags are False
    #
    #     # In a Corner Cases
    #     # Top Left Corner     - No down,  No left
    #     if dont_look_left_flag and dont_look_down_flag:
    #         self.RX = 0
    #         self.RY = 0
    #         possible_moves = temperature_grid[self.X:self.X + 2, self.Y:self.Y + 2]
    #
    #     # Top Right Corner    - No Right, No Down
    #     elif dont_look_right_flag and dont_look_down_flag:
    #         self.RX = 0
    #         self.RY = 1
    #         possible_moves = temperature_grid[self.X:self.X + 2, self.Y - 1:self.Y]
    #
    #     # Bottom Left Corner  - No left,  No Up
    #     elif dont_look_left_flag and dont_look_up_flag:
    #         self.RX = 1
    #         self.RY = 0
    #         possible_moves = temperature_grid[self.X - 1:self.X, self.Y:self.Y + 2]
    #
    #     # Bottom Right Corner - No Right, No Up
    #     elif dont_look_right_flag and dont_look_up_flag:
    #         self.RX = 1
    #         self.RY = 1
    #         possible_moves = temperature_grid[self.X - 1:self.X, self.Y - 1:self.Y]
    #
    #     # This is in the event that the cell is in the left column but not on the top and bottom
    #     elif dont_look_left_flag and not dont_look_down_flag and not dont_look_right_flag and not dont_look_up_flag:
    #         self.RX = 1
    #         self.RY = 0
    #         possible_moves = temperature_grid[self.X - 1:self.X + 2, self.Y:self.Y + 2]
    #
    #     # This is in the event that the cell is in the right column but not on the top and bottom
    #     elif dont_look_right_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_up_flag:
    #         self.RX = 1
    #         self.RY = 1
    #         possible_moves = temperature_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y]
    #
    #     # This is in the event that the cell is in the first row but not on the left and right corners
    #     elif dont_look_down_flag and not dont_look_up_flag and not dont_look_left_flag and not dont_look_right_flag:
    #         self.RX = 0
    #         self.RY = 1
    #         possible_moves = temperature_grid[self.X:self.X + 2, self.Y - 1:self.Y + 2]
    #
    #     # This is in the event that the cell is in the last row but not on the left and right corners
    #     elif dont_look_up_flag and not dont_look_down_flag and not dont_look_left_flag and not dont_look_right_flag:
    #         self.RX = 1
    #         self.RY = 1
    #         possible_moves = temperature_grid[self.X - 1:self.X, self.Y - 1:self.Y + 2]
    #
    #     else:
    #         self.RX = 1
    #         self.RY = 1
    #         possible_moves = temperature_grid[self.X - 1:self.X + 2, self.Y - 1:self.Y + 2]
    #
    #     if np.sum(possible_moves) < 1:
    #         # Check memory bank to see if data exists for this sense
    #         self.Check_Memory()
    #         if self.energy_memory:
    #             X, Y = self.Access_Memory('temperature')
    #             self.DX_temperature = np.sign(X - self.X)
    #             self.DY_temperature = np.sign(Y - self.Y)
    #         else:
    #             self.DX_temperature, self.DY_temperature = 0, 0
    #
    #     else:
    #         possible_moves = possible_moves / np.sum(possible_moves)
    #         weights = np.ravel(
    #             possible_moves)  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
    #         order = np.arange(0, weights.size,
    #                           1)  # This is created the array which contains the order/indexing of the values of the pos move array
    #         choice = random.choices(order,
    #                                 weights=weights)  # This selects a random value from order based off of the weight
    #         coordinates = np.unravel_index(choice, possible_moves.shape)  # returns the unraveled index of the choice.
    #         IX, IY = coordinates[0], coordinates[
    #             1]  # Assigns the first value of that tuple to the X and the second for the Y coordinate of the local array.
    #         self.DX_temperature = (
    #                 IX - self.RX)  # The change in coordinates that cell needs to make is the difference between the target location (IX) and the current location of where the cell is in the relative frame. (RX)
    #         self.DY_temperature = (
    #                 IY - self.RY)  # The change in coordinates that cell needs to make is the difference between the target location (IY) and the current location of where the cell is in the relative frame.

    def Update_Total_Position(self):
        # t_surp    t_starv     e_surp      e_starv     w_surp      w_starv
        #   T           F          F            F          F           F
        #   F           T          F            F          F           F
        #   F           F          T            F          F           F
        #   F           F          F            T          F           F
        #   T           F          T            F          F           F
        #   T           F          F            T          F           F
        #   F           T          T            F          F           F
        #   F           T          F            T          F           F..done
        #
        #   T           F          F            F          T           F
        #   F           T          F            F          T           F
        #   F           F          T            F          T           F
        #   F           F          F            T          T           F
        #   T           F          T            F          T           F
        #   T           F          F            T          T           F
        #   F           T          T            F          T           F
        #   F           T          F            T          T           F
        #
        #   T           F          F            F          F           T
        #   F           T          F            F          F           T
        #   F           F          T            F          F           T
        #   F           F          F            T          F           T
        #   T           F          T            F          F           T
        #   T           F          F            T          F           T
        #   F           T          T            F          F           T
        #   F           T          F            T          F           T

        if self.energy_starved is False and self.energy_surplus is False and self.temperature_surplus and self.temperature_experience and self.water_surplus is False and self.water_starved is False:
            # temperature surplus
            weights = weight_b_s
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            else:
                self.DX_temperature = np.sign(self.X - temperature_x_source)
                self.DY_temperature = np.sign(self.Y - temperature_y_source)
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)

        elif self.energy_starved is False and self.energy_surplus is False and self.temperature_starved and self.temperature_memory and self.water_surplus is False and self.water_starved is False:
            # temperature starved
            weights = weight_b_s
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            else:
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)

        elif self.temperature_starved is False and self.temperature_surplus is False and self.energy_surplus and self.energy_experience and self.water_surplus is False and self.water_starved is False:
            # energy surplus
            weights = weight_b_s
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            else:
                self.DX_energy = np.sign(self.X - food_x_source)
                self.DY_energy = np.sign(self.Y - food_y_source)
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)

        elif self.temperature_starved is False and self.temperature_surplus is False and self.energy_starved and self.energy_memory and self.water_surplus is False and self.water_starved is False:
            # energy starved
            weights = weight_b_s
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            else:
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)

        elif self.temperature_surplus and self.temperature_experience and self.energy_surplus and self.energy_experience and self.water_surplus is False and self.water_starved is False:
            # temperature surplus and energy surplus (rare condition, grids shouldnt overlap)
            weights = np.array([.5,.5])
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.DX_energy = np.sign(self.X - food_x_source)
                self.DY_energy = np.sign(self.Y - food_y_source)
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
            else:
                self.DX_temperature = np.sign(self.X - temperature_x_source)
                self.DY_temperature = np.sign(self.Y - temperature_y_source)
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)

        elif self.temperature_surplus and self.temperature_experience and self.energy_starved and self.energy_memory and self.water_surplus is False and self.water_starved is False:
            # temperature surplus and energy starved
            weights = np.array([.5,.5])
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
            else:
                self.DX_temperature = np.sign(self.X - temperature_x_source)
                self.DY_temperature = np.sign(self.Y - temperature_y_source)
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)

        elif self.temperature_starved and self.temperature_memory and self.energy_surplus and self.energy_experience and self.water_surplus is False and self.water_starved is False:
            # temperature starved and energy surplus
            weights = weight_b_s_s
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.DX_energy = np.sign(self.X - food_x_source)
                self.DY_energy = np.sign(self.Y - food_y_source)
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
            elif choice[0] == 1:
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)
            else:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)

        elif self.temperature_starved and self.temperature_memory and self.energy_starved and self.energy_memory and self.water_surplus is False and self.water_starved is False:
            # temperature starved and energy starved
            weights = weight_b_s_s
            order = np.arange(0, weights.size, 1)
            choice = random.choices(order, weights=weights)
            if choice[0] == 0:
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
            elif choice[0] == 1:
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)
            else:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)

        else:
            self.X = self.X + int(self.DX_background)
            self.Y = self.Y + int(self.DY_background)

        if self.X < 1:
            self.X = 1
        elif self.X > Grid_Size-1:
            self.X = Grid_Size-1
        if self.Y < 1:
            self.Y = 1
        elif self.Y > Grid_Size - 1:
            self.Y = Grid_Size - 1

        """
        #This is the logic for Starvation
        if self.energy_starved is False:
            #This is the logic to run the proabability models for suprlus, in the event that you have too much energy only
            if self.energy_surplus and self.energy_experience and self.temperature_surplus is False:
                weights = np.array([.001, .999])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                else:
                    self.DX_energy = np.sign(self.X - food_x_source)
                    self.DY_energy = np.sign(self.Y - food_y_source)
                    self.X = self.X + int(self.DX_energy)
                    self.Y = self.Y + int(self.DY_energy)
            elif self.temperature_surplus and self.temperature_experience and self.energy_surplus is False:
                weights = np.array([.001, .999])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                else:
                    self.DX_temperature = np.sign(self.X - temperature_x_source)
                    self.DY_temperature = np.sign(self.Y - temperature_y_source)
                    self.X = self.X + int(self.DX_temperature)
                    self.Y = self.Y + int(self.DY_temperature)
            elif self.energy_surplus and self.energy_experience and self.temperature_surplus and self.temperature_experience:
                weights = np.array([.002, .499, .499])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                elif choice[0] == 1:
                    self.DX_energy = np.sign(self.X - food_x_source)
                    self.DY_energy = np.sign(self.Y - food_y_source)
                    self.X = self.X + int(self.DX_energy)
                    self.Y = self.Y + int(self.DY_energy)
                elif choice[0] == 2:
                    self.DX_temperature = np.sign(self.X - temperature_x_source)
                    self.DY_temperature = np.sign(self.Y - temperature_y_source)
                    self.X = self.X + int(self.DX_temperature)
                    self.Y = self.Y + int(self.DY_temperature)
            else:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
        elif self.energy_starved and self.energy_memory and self.temperature_starved is False:
            #There is still a possiblity that the temperature may have a surplus in this feild
            # Temp surplus logic is needed.

            #This is the logic to run the proabability models for suprlus, in the event that you have too much energy only
            if self.temperature_surplus and self.temperature_experience:
                weights = np.array([.001,.99,.001])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                elif choice[0] == 1:
                    self.DX_temperature = np.sign(self.X - temperature_x_source)
                    self.DY_temperature = np.sign(self.Y - temperature_y_source)
                    self.X = self.X + int(self.DX_temperature)
                    self.Y = self.Y + int(self.DY_temperature)
                elif choice[0] == 2:
                    self.X = self.X + int(self.DX_energy)
                    self.Y = self.Y + int(self.DY_energy)

            else:
                weights = np.array([.001, .999])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                else:
                    self.X = self.X + int(self.DX_energy)
                    self.Y = self.Y + int(self.DY_energy)
        elif self.temperature_starved and self.temperature_memory and self.energy_starved is False:


            #This is the logic to run the proabability models for suprlus, in the event that you have too much energy only
            if self.energy_surplus and self.energy_experience:
                weights = np.array([.001, .45,.45])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                elif choice[0] == 1:
                    self.X = self.X + int(self.DX_temperature)
                    self.Y = self.Y + int(self.DY_temperature)
                elif choice[0] == 2:
                    self.X = self.X + int(self.DX_energy)
                    self.Y = self.Y + int(self.DY_energy)
            else:
                weights = np.array([.001, .999])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
                order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
                choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
                if choice[0] == 0:
                    self.X = self.X + int(self.DX_background)
                    self.Y = self.Y + int(self.DY_background)
                else:
                    self.X = self.X + int(self.DX_temperature)
                    self.Y = self.Y + int(self.DY_temperature)
        elif self.energy_starved and self.temperature_starved and self.temperature_memory and self.energy_memory:
            weights = np.array([.002, .49, .49])  # Ravels the posisble moves array and uses their normalized values as a weight for the probability choice.
            order = np.arange(0, weights.size, 1)  # This is created the array which contains the order/indexing of the values of the pos move array
            choice = random.choices(order, weights=weights)  # This selects a random value from order based off of the weight
            if choice[0] == 0:
                self.X = self.X + int(self.DX_background)
                self.Y = self.Y + int(self.DY_background)
            elif choice[0] == 1:
                self.X = self.X + int(self.DX_energy)
                self.Y = self.Y + int(self.DY_energy)
            elif choice[0] == 2:
                self.X = self.X + int(self.DX_temperature)
                self.Y = self.Y + int(self.DY_temperature)
        else:
            self.X = self.X + int(self.DX_background)
            self.Y = self.Y + int(self.DY_background)
        # if self.X < 0:
        #     self.X=0
        # if self.Y < 0:
        #     self.Y=0
        """

    def Add_History(self):
        self.X_History = np.hstack((self.X_History, self.X))
        self.Y_History = np.hstack((self.Y_History, self.Y))

    def Update_Death_Count_Rate(self):
        self.death_count_rate += self.death_count_rate

    def Update_Sensory_Level(self, energy_grid, temperature_grid):
        # Updating the Energy Level
        if self.energy_experience and self.energy_level<upper_sensory_cap:
            self.energy_level += energy_grid[self.X, self.Y]
            self.energy_level -= energy_grid[self.X, self.Y] / 2
        else:
            if self.energy_level>lower_sensory_cap:
                self.energy_level -= food_decay_rate
        self.energy_level=np.min([self.energy_level,upper_sensory_cap])

        # Updating the Temperature Level
        if self.temperature_experience and self.temperature_level<upper_sensory_cap:
            self.temperature_level += temperature_grid[self.X, self.Y]
            self.temperature_level -= temperature_grid[self.X, self.Y] / 2
        else:
            if self.temperature_level>lower_sensory_cap:
                self.temperature_level -= temperature_decay_rate
        self.temperature_level=np.min([self.temperature_level,upper_sensory_cap])

    def Update_Memory(self, energy_grid, temperature_grid):

        # Energy Update
        already_known = False
        if energy_grid[self.X, self.Y] > 0:
            for i in range(self.memory_bank.shape[0]):
                if self.memory_bank[i, 0, 0] > 0:
                    if self.memory_bank[i, 1, 0] == self.X and self.memory_bank[i, 2, 0] == self.Y:
                        already_known = True
                        break
            if already_known is False:
                self.memory_bank[:, :, 0] = np.roll(self.memory_bank[:, :, 0], 1, axis=0)
                self.memory_bank[0, 0, 0], self.memory_bank[0, 1, 0], self.memory_bank[0, 2, 0] = energy_grid[self.X, self.Y], self.X, self.Y
        # Temp Update
        already_known = False
        if temperature_grid[self.X, self.Y] > 0:
            for i in range(self.memory_bank.shape[0]):
                if self.memory_bank[i, 0, 1] > 0:
                    if self.memory_bank[i, 1, 1] == self.X and self.memory_bank[i, 2, 1] == self.Y:
                        already_known = True
                        break
            if already_known is False:
                self.memory_bank[:, :, 1] = np.roll(self.memory_bank[:, :, 1], 1, axis=0)
                self.memory_bank[0, 0, 1], self.memory_bank[0, 1, 1], self.memory_bank[0, 2, 1] = temperature_grid[self.X, self.Y], self.X, self.Y

    def Experience_Check(self, energy_grid, temperature_grid):
        # Check if cell is currently experiencing sensory input
        if energy_grid[self.X, self.Y] > 0:
            self.energy_experience = True
        else:
            self.energy_experience = False
        if temperature_grid[self.X, self.Y] > 0:
            self.temperature_experience = True
        else:
            self.temperature_experience = False

    def Starvation_Check(self):
        # Check if cell is currently starved for each sense
        if self.energy_level < food_lower_harm:
            self.energy_starved = True
        else:
            self.energy_starved = False

        if self.temperature_level < temperature_lower_harm:
            self.temperature_starved = True
        else:
            self.temperature_starved = False

    def Surplus_Check(self):
        # Check if cell is currently in surplus for each sense
        if self.energy_level > food_upper_harm:
            self.energy_surplus = True
        else:
            self.energy_surplus = False

        if self.temperature_level > temperature_upper_harm:
            self.temperature_surplus = True
        else:
            self.temperature_surplus = False

    def Check_Memory(self):
        # Check if cell has memory of sensory input
        if np.any(self.memory_bank):
            if np.any(self.memory_bank[:, :, 0]):
                self.energy_memory = True

            if np.any(self.memory_bank[:, :, 1]):
                self.temperature_memory = True

    def Access_Memory(self, sense):
        # Access cell's memory bank to retrieve closest sensory coordinates
        x_sense, y_sense = 0, 0
        # if the energy memory is true find the x and y of the lowest dist
        if sense == 'energy':
            dist_store = [100000 for ii in range(10)]
            for ii in range(self.memory_bank.shape[0]):
                if self.memory_bank[ii, 0, 0] > 0:
                    dist_store[ii] = np.linalg.norm(np.array([self.memory_bank[ii, 1, 0] - self.X, self.memory_bank[ii, 2, 0] - self.Y]))
            min_index = np.argmin(dist_store)
            x_sense, y_sense = self.memory_bank[min_index, 1, 0], self.memory_bank[min_index, 2, 0]

        # if the temp memory is true find the x and y of the lowest dist
        elif sense == 'temperature':
            dist_store = [100000 for ii in range(10)]
            for ii in range(self.memory_bank.shape[0]):
                if self.memory_bank[ii, 0, 1] > 0:
                    dist_store[ii] = np.linalg.norm(
                        np.array([self.memory_bank[ii, 1, 1] - self.X, self.memory_bank[ii, 2, 1] - self.Y]))
            min_index = np.argmin(dist_store)
            x_sense, y_sense = self.memory_bank[min_index, 1, 1], self.memory_bank[min_index, 2, 1]

        return x_sense, y_sense


# ________________________________________________________________________________________________________

# FUNCTIONS

# Creates an N number of cells. Takes in information about the grid which is 'SIZE'
def Create_Cells(n, grid_size):
    Cell_List = []
    for i in range(N):
        x, y = np.random.randint(0, high=grid_size), np.random.randint(0, high=grid_size)
        memory_bank = np.zeros((10, 3, 2))
        Cell_List.append(Cell(x, y, memory_bank))
    return Cell_List


# Creating the Space the Cells will move within, and the free will field.
def Create_Grid(grid_size):
    Grid = np.random.randint(0, high=10, size=(grid_size, grid_size))
    return Grid


# Creates an array of an energy grid and defines its location and its values
def Create_Sensory_Grid(grid_size, Sense):
    Sensory_Grid = np.zeros((grid_size, grid_size))
    s_grid_length = 0
    if Sense == "Energy":
        X0 = food_x_source
        Y0 = food_y_source
        s_grid_length = 3
        for i in range(s_grid_length-1, -1, -1):
            Sensory_Grid[X0 - i:X0 + i + 1, Y0 - i:Y0 + i + 1] = source_value / (i + 1)
    elif Sense == "Temperature":
        X0 = temperature_x_source
        Y0 = temperature_y_source
        s_grid_length = 7
        for i in range(s_grid_length - 1, -1, -1):
            Sensory_Grid[X0 - i:X0 + i + 1, Y0 - i:Y0 + i + 1] = source_value / (i + 1)
    elif Sense == 'Water':
        X0 = water_x_source
        Y0 = water_y_source
        s_grid_length = 7
        for i in range(s_grid_length - 1, -1, -1):
            Sensory_Grid[X0 - i:X0 + i + 1, Y0 - i:Y0 + i + 1] = source_value / (i + 1)
    return Sensory_Grid, s_grid_length


Cell_List = Create_Cells(N, Grid_Size)
Energy_Grid, e_grid_len = Create_Sensory_Grid(Grid_Size, "Energy")
Temperature_Grid, t_grid_len = Create_Sensory_Grid(Grid_Size, "Temperature")
Water_Grid, w_grid_len = Create_Sensory_Grid(Grid_Size, "Water")
Death_Count_List = np.zeros((N, TF))
Energy_List = np.zeros((N, TF))
Temp_List = np.zeros((N, TF))
Water_List = np.zeros((N, TF))
for i in range(TF):
    Grid = Create_Grid(Grid_Size)
    for j in range(N):
        if Cell_List[j].death_count > 0:
            Cell_List[j].Experience_Check(Energy_Grid, Temperature_Grid)
            Cell_List[j].Update_Memory(Energy_Grid, Temperature_Grid)
            Cell_List[j].Update_Sensory_Level(Energy_Grid, Temperature_Grid)
            Cell_List[j].Starvation_Check()
            # Movement Models
            Cell_List[j].Surplus_Check()
            Cell_List[j].Update_Background_Position(Grid)
            if Cell_List[j].energy_starved and Cell_List[j].energy_memory:
                Cell_List[j].Update_Sensory_Position(Energy_Grid, 'energy')
            if Cell_List[j].temperature_starved and Cell_List[j].temperature_memory:
                Cell_List[j].Update_Sensory_Position(Temperature_Grid, 'temperature')
            if Cell_List[j].water_starved and Cell_List[j].water_memory:
                Cell_List[j].Update_Sensory_Position(Water_Grid, 'water')
            Cell_List[j].Update_Total_Position()

            Cell_List[j].Add_History()
            Cell_List[j].Death_Tick()
        else:
            Cell_List[j].Add_History()
        Death_Count_List[j, i] = Cell_List[j].death_count
        Energy_List[j, i] = Cell_List[j].energy_level
        Temp_List[j, i] = Cell_List[j].temperature_level
        Water_List[j, i] = Cell_List[j].water_level


    print(i)

# THIS IS ALL ANIMATION AND PLOTTING

# ________________________________________________________________________________________________________
# Doing the Animation
idx_cells = np.asarray([i for i in range(N)])
fig, axd = plt.subplot_mosaic([['upper left', 'grid'],
                               ['mid1 left', 'grid'],
                               ['mid2 left', 'grid'],
                               ['lower left', 'grid']],
                              figsize=(5.5, 3.5))
plt.tight_layout(pad=2)
ax1, ax2, ax3, ax4, ax5 = axd["grid"], axd["upper left"], axd["lower left"], axd['mid1 left'], axd['mid2 left']
# Set the axis limits
ax1.set_xlim(-1, Grid_Size)
ax1.set_ylim(-1, Grid_Size)
ax2.set_xlim(-1, N)
ax2.set_ylim(0, np.max(Death_Count_List))
ax3.set_xlim(-1, N)
ax3.set_ylim(np.min(Energy_List ), np.max(Energy_List))
ax4.set_xlim(-1, N)
ax4.set_ylim(np.min(Temp_List), np.max(Temp_List))
ax5.set_xlim(-1, N)
ax5.set_ylim(np.min(Temp_List), np.max(Water_List))
ax3.set_xticks(idx_cells)
ax2.set_xticks(idx_cells)
ax4.set_xticks(idx_cells)
ax5.set_xticks(idx_cells)
prefs = []
for i in range(N):
    prefs.append(str(Cell_List[i].energy_preference))
xlabels = []
for i in range(N):
    xlabels.append(f'E pref = {prefs[i]}')
ax3.set_xticklabels(xlabels, rotation=50)

ncol = [c for c in colors.get_named_colors_mapping()]

# This creates the rectangles for the energy grid
E_rect1 = patches.Rectangle((food_x_source - 2, food_y_source - 2), 5, 5, color='gold')
ax1.add_patch(E_rect1)
E_rect2 = patches.Rectangle((food_x_source - 1, food_y_source - 1), 3, 3, color='orange')
ax1.add_patch(E_rect2)
E_rect3 = patches.Rectangle((food_x_source, food_y_source), 1, 1, color='darkorange')
ax1.add_patch(E_rect3)
# This creates the rectangles for the Temperature grid
T_rect1 = patches.Rectangle((temperature_x_source - 6, temperature_y_source - 6), 13, 13, color=(1, .2, .2))
ax1.add_patch(T_rect1)
T_rect2 = patches.Rectangle((temperature_x_source - 5, temperature_y_source - 5), 11, 11, color=(.9, .2, .2))
ax1.add_patch(T_rect2)
T_rect3 = patches.Rectangle((temperature_x_source - 4, temperature_y_source - 4), 9, 9, color=(.8, .2, .2))
ax1.add_patch(T_rect3)
T_rect4 = patches.Rectangle((temperature_x_source - 3, temperature_y_source - 3), 7, 7, color=(.7, .2, .2))
ax1.add_patch(T_rect4)
T_rect5 = patches.Rectangle((temperature_x_source - 2, temperature_y_source - 2), 5, 5, color=(.6, .2, .2))
ax1.add_patch(T_rect5)
T_rect6 = patches.Rectangle((temperature_x_source - 1, temperature_y_source - 1), 3, 3, color=(.5, .2, .2))
ax1.add_patch(T_rect6)
T_rect7 = patches.Rectangle((temperature_x_source, temperature_y_source), 1, 1, color=(.4, .2, .2))
ax1.add_patch(T_rect7)
# This creates the rectangles for the water grid
W_rect1 = patches.Rectangle((water_x_source - 2, water_y_source - 2), 5, 5, color='deepskyblue')
ax1.add_patch(W_rect1)
W_rect2 = patches.Rectangle((water_x_source - 1, water_y_source - 1), 3, 3, color='dodgerblue')
ax1.add_patch(W_rect2)
W_rect3 = patches.Rectangle((water_x_source, water_y_source), 1, 1, color='blue')
ax1.add_patch(W_rect3)

ax1.grid(which="both")
ax1.minorticks_on()
ax2.set_title('death count')
ax3.set_title('energy levels')
ax4.set_title('temp levels')
ax5.set_title('water levels')


def update(num, lines, Cell_List, bars, Bar_List, energy, Energy_List, temps, Temp_List, waters, Water_List):
    if num > 5:
        for i in range(N):
            lines[i].set_data(Cell_List[i].X_History[num - 5:num], Cell_List[i].Y_History[num - 5:num])
    else:
        for i in range(N):
            lines[i].set_data(Cell_List[i].X_History[:num], Cell_List[i].Y_History[:num])
    ax1.set_title('iteration: %d' % num)

    # This adds in the Temperature rectangles every loop so they dont disapear.
    ax1.add_patch(T_rect1)
    ax1.add_patch(T_rect2)
    ax1.add_patch(T_rect3)
    ax1.add_patch(T_rect4)
    ax1.add_patch(T_rect5)
    ax1.add_patch(T_rect6)
    ax1.add_patch(T_rect7)

    # This adds in the Food rectangle every loop so they dont disapear.
    ax1.add_patch(E_rect1)
    ax1.add_patch(E_rect2)
    ax1.add_patch(E_rect3)

    # This adds in the Water rectangle every loop so they dont disapear.
    ax1.add_patch(W_rect1)
    ax1.add_patch(W_rect2)
    ax1.add_patch(W_rect3)

    for i in range(N):
        bars[i].set_height(Bar_List[i, num])
        energy[i].set_height(Energy_List[i, num])
        temps[i].set_height(Temp_List[i, num])
        waters[i].set_height(Water_List[i, num])


# Creates an empty list of the line items that will be used to do the animation
line_list = []
energy_list = []
bar_list = []
temp_list = []
water_list = []
for i in range(N):
    g, = ax1.plot([], [], marker='*')
    line_list.append(g)
    g, = ax2.bar(i, 0)
    bar_list.append(g)
    g, = ax3.bar(i, 0)
    energy_list.append(g)
    g, = ax4.bar(i, 0)
    temp_list.append(g)
    g, = ax5.bar(i, 0)
    water_list.append(g)

# Create the animation using the update function
anim = animation.FuncAnimation(fig, update,
                               fargs=(line_list, Cell_List, bar_list, Death_Count_List, energy_list, Energy_List, temp_list, Temp_List, water_list, Water_List),
                               interval=50, frames=TF, repeat=True)
# Show the animation

# for j in range(N):
#     ax1.plot(Cell_List[j].X_History,Cell_List[j].Y_History, marker='*')

plt.show()
