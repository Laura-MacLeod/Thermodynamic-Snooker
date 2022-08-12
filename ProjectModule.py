#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:11:57 2021

@author: laura
"""

# Code written by Laura MacLeod. Project concept provided by Imperial College London. Animation and graphics code examples provided by Imperial College.

import numpy as np
import pylab as pl
import warnings
import matplotlib.pyplot as plt
import random
import scipy as sp
from scipy import optimize

'''
A module to store classes to simulate an ideal gas in a container
Laura MacLeod
01/11/2021
'''


class Ball:
    '''
    A class that produces a ball of a specified mass, radius, vector position, 
    vector velocity, face colour, edge colour, and fill.
    A circle patch is initialised to use for the animation.
    '''
    
    def __init__(self, mass=1, radius=1, position=np.array([0, 0]), velocity=np.array([0, 0]), 
                 facecolour=None, edgecolour=None, fill=True):
        self._mass = mass
        self._radius = radius
        self._position = np.array(position, dtype="float64")
        self._velocity = np.array(velocity, dtype="float64")
        self._facecolour = facecolour
        self._edgecolour = edgecolour
        self._fill = fill
        self._patch = pl.Circle(self._position, self._radius, fc=self._facecolour, ec=self._edgecolour, fill=self._fill)


    def pos(self):
        return self._position
    
    def vel(self):
        return self._velocity
    
    def rad(self):
        return self._radius
    
    def set_vel(self, v=np.array([0, 0])):
        self._velocity = v
    
    def __repr__(self):
        return "%s(mass=%s, radius=%s, position=%s, velocity=%s, facecolour=%s, fill=%s)" % ("Ball", str(self._mass), str(self._radius), str(self._position), str(self._velocity), str(self._facecolour), str(self._fill))
    
    
    def move(self, dt):
        
        '''
        This method moves the ball a distance from its original position depending on the time
        passed and its velocity vector, using dx/dt = v.
        It sets the patch to the new position so the balls in the animation move.
        '''
        
        self._position = self._position + (self._velocity * (0.99999999*dt))
        self._patch.center = self._position
        return self._position
    
    
    def time_to_collision(self, other):
        
        '''
        This method calculates the time after which the ball will collide with another ball 
        (or container) using an equation of their relative positions, velocities, and the sum
        of their radii.
        '''
        
        r = self._position - other._position
        v = self._velocity - other._velocity

        rad = abs(other._radius + self._radius)  
        
        # The container is a ball with a negative radius and so to collide with its inside,
        # the square root is positive. Balls colliding with balls have a negative square root:
        
        if self._radius < 0 or other._radius < 0:
            out = ((-2 * np.dot(r,v) + np.sqrt(4 * (np.dot(r,v))**2 - 4 * (np.dot(v,v)) * 
                                             ((np.dot(r,r)) - rad**2)))/(2 * (np.dot(v,v))))
        else:
            out = ((-2 * np.dot(r,v) - np.sqrt(4 * (np.dot(r,v))**2 - 4 * (np.dot(v,v)) * 
                                             ((np.dot(r,r)) - rad**2)))/(2 * (np.dot(v,v))))
        
            
        # The time function returns NaN for collisions that will never happen and have never
        # happened, and a negative number for collisions that happened in the past. These
        # values are set to infinity so they don't interfere with the minimum time:
        
        if np.isnan(out) == True or out <= 0:
            out = np.inf
            
        return out
   
    
    def collide(self, other):
        
        '''
        This method collides the ball with another ball (or container). The equations are the
        2D elastic collision between for two particles, and sets the velocities to the new ones.
        '''
        
        r = self._position - other._position
        v = self._velocity - other._velocity

        self._velocity = self._velocity - ((2*other._mass)/(self._mass+other._mass))*((np.dot(v, r))/(np.dot(r, r))) * (r)
        other._velocity = other._velocity - ((2*self._mass)/(self._mass+other._mass))*((np.dot(-v, -r))/(np.dot(-r, -r))) * (-r)
        
     
    def get_patch(self):
        return self._patch






class SingleSimulation:
    
    '''
    A class to simulate the collision between a single ball and the container. It contains 
    the next_collision method to compute the next collision, which is used in the run 
    method where the simulation is run (and animated if animate=True).
    '''
    
    def __init__(self, ball, container):
        self._container = container
        self._ball = ball
    
    
    def next_collision(self):
        
        '''
        This method computes the next collision using the steps:
            
            [1] Calculates the time t until ball and container next collide.
            
            [2] Moves the ball to its position after time t (container movement is 
            negligible)
            
            [3] Collides the ball with the container, setting the new velocities.
        '''
        
        time = self._ball.time_to_collision(self._container)
        self._ball.move(time)
        self._container.move(time)
        self._ball.collide(self._container)
    
    
    def run(self, num_frames, animate=False):
        
        '''
        This method collides the ball for the number of times there are num_frames input.
        Axes of 10x10 are used.
        If animate = True, pylab animates the simulation and attempts to show a new frame
        every 0.1 second.
        '''
        
        if animate:
            pl.figure()
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self._container.get_patch())
            ax.add_patch(self._ball.get_patch())
            
        for frame in range(num_frames):
            self.next_collision()
            if animate:
                pl.pause(0.1)
                
        if animate:
            pl.show()



class Simulation:
    '''
    A class to simulate the collision between multiple balls and the container.
    The container and balls must all be manually input into the class as nested arrays (or 
    nested arrays for position and velocity) containing each ball's attribute values.
    If animate = True, the simulation is animated with smooth movements.
    '''
    
    def __init__(self, mass=1, radius=1, position=[np.array([0, 0])], velocity=[np.array([0, 0])], 
                 facecolour=None, edgecolour=None):
        
        self._mass = np.array(mass, dtype="float64")
        self._radius = np.array(radius, dtype="float64")
        self._position = np.array(np.array(position, dtype="float64"), dtype="float64")
        self._velocity = np.array(np.array(velocity, dtype="float64"), dtype="float64")
        self._facecolour = np.array(facecolour, dtype="str")
        self._edgecolour = np.array(edgecolour, dtype="str")
        
        # The balls attributes are initialised as individual balls and added to a new list:
        
        balls = []
        
        for i in range(0, len(self._mass)):
            balls.append(Ball(self._mass[i], self._radius[i], self._position[i], self._velocity[i], 
                              self._facecolour[i], 'black'))
        
        self._balls = balls  # Let the list be a class attribute to make it accessible from elsewhere
        
    
    def __repr__(self):
        return "%s(mass=%s, \n radius=%s, \n position=%s, \n velocity=%s, \n facecolour=%s)" % ("Simulation", str(self._mass), str(self._radius), str(self._position), str(self._velocity), str(self._facecolour))
    
    def give_mass(self):
        print(self._mass)
    
    def give_ball(self, index):
        print(self._balls[index])
    
    def give_ball_list(self):
        print(self._balls)
        
        
    def next_collision(self):
        
        '''
        This method computes the next collision, except it has to use the minimum collision
        time out of all possible collisions between balls and the container.
        Sets were used to ensure there were no duplicates of a collision, e.g. only one of
        [4, 7] and [7, 4] can exist.
        
        The steps in the method are:
            
            [1] Add all the possible ball-ball and ball-container collision combinations (as
            frozen sets containing their indexes) to the set 'index_set'.
            
            [2] Let the set 'index_set' and the frozen sets it contains become a nested
            list 'collision_index'.
            
            [3] Add the time to collision between the two balls to the list 'times'.
            
            [4] Find the minimum time 'mintime' and its associated list index 'minindex'. 
            
            [5] Retrieve the two colliding balls' indixes from 'collision_index' and assign
            them to 'collision1' and'collision2'.
            
            [6] Move all of the balls by the 'mintime'.
            
            [7] Collide the two balls that are associated with the soonest collision.
        '''
        
        # [1]
        index_set = set()
        
        for i in range(0, len(self._mass)):
            for j in range(0, len(self._mass)):
                if i!=j:  # ensures no ball collides with itself
                    index_set.add(frozenset([i, j]))
        
        # [2]
        index_list = list(index_set)
        collision_index = []
        
        for i in index_list:
            sublist = []
            for j in i:
                sublist.append(j)
            collision_index.append(sublist)
        
        # [3]
        times = []
        
        for i in collision_index:
            times.append(self._balls[i[0]].time_to_collision(self._balls[i[1]]))
        
        # [4]
        mintime = min(times)
        minindex = times.index(mintime)
        
        # [5]
        minlist = collision_index[minindex]

        collision1 = int(minlist[0])
        collision2 = int(minlist[1])
        
        # [6]
        for i in self._balls:
            i.move(mintime)
        
        # [7]
        self._balls[collision1].collide(self._balls[collision2])

    
    
    def run(self, num_frames, animate=False):
        
        '''
        This method runs the simulation. For every frame in the number of frames, the balls
        are collided. If animate = True, the new positions for each collision are shown in 
        the animation every 0.1 seconds.
        '''
        
        if animate:
            pl.figure()
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            for i in range(0, len(self._balls)):
                ax.add_artist(self._balls[i].get_patch())

        for frame in range(num_frames):
            
            if animate:
                pl.pause(0.1)
                
            self.next_collision()
            
        if animate:
            pl.show()





class SystematicSimulation:
    '''
    A class to simulate a gas containing N automatically initialised balls. All balls
    have the attributes parameterised in the class input. The ball positions are 
    intialised systematically, and the velocity distribution is initialised randomly 
    so the mean is around zero and the mean velocity is constant between simulations.
    If animate = True, the simulation is animated with smooth movements.
    
    '''
    
    def __init__(self, mass=1, radius=1, containerradius=-10, facecolour=None, edgecolour=None, numberballs=1, velocity=1, temps = [], press = []):
        
        '''
        In the initialisation of this class, multiple steps are taken:
            
            [1] Create the container as the first ball in the 'balls' list
            
            [2] Round the input number of balls to the nearest square number. If the input 
            is not a square number, a warning will be printed in the kernel.
            
            [3] Find the velocity component of the input speed using Pythagorus' theorem
            and let each ball's velocity components equal either the positive or negative
            version of this value.
            
            [4] An old version of the velocity distribution algorithm. Sets the velocities 
            to a random normal distribution, but is no longer used because the temperature
            is not constant between simulations.
            
            [5] Use Pythagorus' theorem to find the largest width of the square by
            considering the container's radius. The bal radius is removed from this
            distance so the balls are all initialised inside the container.
            
            [6] Initialises the ball positions by equally spacing them in the square grid
            depending on the particle number.
            
            [7] Initialises and adds all the balls to the list already containing the
            container.
            [8] Turns ball list into
            class attribute, and creates counter attributes for
            the time and the momentum applied to the container.
            
        '''
        
        self._mass = mass
        self._radius = radius
        self._containerradius = containerradius
        self._facecolour = facecolour
        self._edgecolour = edgecolour
        self._numberballs = numberballs
        self._velocity = velocity  # Magnitude of velocity for every ball
        self._temps = temps  # Used to return list of temperatures
        self._press = press  # Used to return list of pressures
        
        # [1]
        
        balls = [Ball(mass=1e100, radius=self._containerradius, position=np.array([0, 0]), 
                      velocity=np.array([0, 0]), facecolour=None, fill=False)]
        
        # [2]
        
        total_num_balls = (round((self._numberballs)**(1/2)))**2
        if self._numberballs**(1/2) != int(self._numberballs**(1/2)):
            warnings.warn("A non-square number of balls was specified. Number of balls will be rounded to nearest square number.")
        
        # [3]
        
        vel_component = (0.5*(self._velocity**2))**(1/2)
        vel_comp_list = [-vel_component, vel_component]
        
        random_velocities = []
        
        for i in range(0, total_num_balls):
            random_velocities.append([np.random.choice(vel_comp_list), np.random.choice(vel_comp_list)])
        
        # [4]
        
        # normal_velocities = []
        # positions = []
        
        # for i in range(0, total_num_balls):
        #     normal_velocities.append([np.random.normal(scale=self._velocity), np.random.normal(scale=self._velocity)])
        
        # for i in range(0, self._numberballs):
        #     r = i * (10/self._numberballs)
        #     theta = i * (np.pi/4)
        #     positions.append([r*np.cos(theta), r*np.sin(theta)])
        
        # [5]
        
        number_sqrt = int(np.sqrt(total_num_balls))
        largest_width = ((((-2 * self._containerradius)**2)/2)**0.5)/2 - self._radius  # Half of maximum square width
        
        # [6]
        
        positions = []
        for i in np.linspace(-largest_width, largest_width, num=number_sqrt):
            for j in np.linspace(-largest_width, largest_width, num=number_sqrt):
                positions.append([i, j])
        
        # [7]
        
        for i in range(0, total_num_balls):
            balls.append(Ball(self._mass, self._radius, positions[i], random_velocities[i], 
                              self._facecolour, self._edgecolour))
        
        # [8]
        
        self._balls = balls
        self._time = 0
        self._mom = 0
        
        
    
    def __repr__(self):
        return "%s(mass=%s, \n radius=%s, \n position=%s, \n velocity=%s, \n facecolour=%s, \n edgecolour=%s)" % ("Simulation", str(self._mass), str(self._radius), str(self._position), str(self._velocity), str(self._facecolour), str(self.edgecolour))
    
    def give_mass(self):
        return(self._mass)
    
    def give_ball(self, index):
        return(self._balls[index])
    
    def give_ball_list(self):
        return(self._balls)
    
    def give_time(self):
        return(self._time)
    
    def give_momentum(self, n):
        return self._mass * ((self._balls[n].vel()[0])**2 + (self._balls[n].vel()[1])**2)
    
    def give_container_radius(self):
        return self._balls[0].rad()
        
    

    def smooth_move(self):
        
        '''
        This method is an extension to the next_collision function. It collides the balls as
        normal, but moves the balls in increments of 0.001 until the next collision so the
        animation appears smooth.
        Additionally, this method accounts for balls that collide simultaneously, where
        next_collision would only collide one of them.
        
        [1] Add all the possible ball-ball and ball-container collision combinations (as
        frozen sets containing their indexes) to the set 'index_set'.
        
        [2] Let the set 'index_set' and the frozen sets it contains become a nested
        list 'collision_index'.
        
        [3] Add the time to collision between the two balls to the list 'times'.
        
        [4] Finds the minimum time and then adds all combinations of collision indexes that
        have the 'mintime' to the list 'min_indices'.
        
        [5] Retrieves all of the colliding ball pairs' indexes from 'collision_index' and
        adds them to 'minlist'.
        
        [6] Generates a list of times increasing by 0.001 until it reaches the min time.
        
        [7] Moves each ball for 0.001 second until the min time is reached, while adding
        the time passed to the time counter attribute.
        * Moves balls until the penultimate value in list of times to ensure they don't
        move for longer than the min time, because it probably won't be perfectly divisible
        by 0.001.
        ** Moves the balls for a time that is the difference between the min time and the
        penultimate list time to ensure that the balls move for the exact min time
        
        [8] Assigns each value of each index index pair to 'collision1' or 'collision2'.
        These are all the collisions that occur at the minimum time. Collides each pair of
        balls.
        
        [9] Every time a ball collides with the container (index 0), the momentum it
        applies (2 * ball momentum) is added to the momentum counter attribute. This is
        only done after 0.2 seconds to give the balls time to disperse from their original
        positions in a square.
        
        '''
        
        # [1]
        
        index_set = set()
        
        for i in range(0, len(self._balls)):
            for j in range(0, len(self._balls)):
                if i!=j:  # ensures no ball collides with itself
                    index_set.add(frozenset([i, j]))
        
        # [2]
        
        index_list = list(index_set)
        collision_index = []
        
        for i in index_list:
            sublist = []
            for j in i:
                sublist.append(j)
            collision_index.append(sublist)
      
        # [3]
        
        times = []

        for i in collision_index:
            times.append(self._balls[i[0]].time_to_collision(self._balls[i[1]]))

        # [4]

        mintime = min(times)

        min_indices = []
        
        for i in range(0, len(times)):
            if times[i] == mintime:
                min_indices.append(i)
        
        # [5]
        
        minlist = []
        
        for i in min_indices:
            minlist.append(collision_index[i])
        
        # [6]
        
        timerange = np.arange(0, mintime, step=0.001)
        
        # [7]
        
        for i in self._balls:
                for j in timerange[0:len(timerange)-1:]: # *
                    i.move(0.001)
                    self._time += 0.001
                i.move(mintime-timerange[len(timerange)-1])  # **
                self._time += mintime-timerange[len(timerange)-1]
        
        # [8]
        
        for i in range(0, len(minlist)):
            collision1 = int(minlist[i][0])
            collision2 = int(minlist[i][1])
            self._balls[collision1].collide(self._balls[collision2])
        
        # [9]
        
        if self._time > (0.2):
            if collision1 == 0:
                self._mom += (2 * self.give_momentum(collision2))
            elif collision2 == 0:
                self._mom += (2 * self.give_momentum(collision1))



    
    def run(self, num_frames, animate=False):
        
        '''
        This method runs the simulation. For every frame in the number of frames, the balls
        are moved for 0.001 seconds. If animate = True, the new positions for each 
        movement are shown in the animation every 0.001 seconds.
        
        This method also produces the following data:
            
            [1] Plots a histogram for the distance from the container's centre for every ball
            and for every frame, although does not include the first 500 frames to allow for the
            dispersion of the balls.
            
            [2] Plots a histogram for the separation between every ball and every other ball
            and for every frame, although does not include the first 500 frames to allow for the
            dispersion of the balls.
            
            [3] Plots a scatter graph of the total kinetic energy over every incrememt of
            time in the simulation
            
            [4] Assuming the kinetic energy is constant as it should be, the first value in
            the list of kinetic energies is printed.
            
            [5] The temperature is found from the kinetic energy using T = (2/3 * KE) / k_B
            and printed.
            
            [6] Plots a scatter graph of the total momentum over every incrememt of
            time in the simulation
            
            [7] Assuming the momentum is constant as it should be, the first value in
            the list of momentums is printed.
            
            [8] Plots a histogram for the speed of every ball for every frame, although does 
            not include the first 100 frames to allow for the dispersion of the balls. A 
            Maxwell-Boltzmann distribution is plot against the histogram for the same
            temperature as the simulation, and normalised by multiplying by the histogram's
            area.
            
            [9] The ratio between the total balls' area and the container's area is 
            calculated and printed.
            
            [10] The total momentum applied to the container is printed.
            
            [11] The total time passed is printed.
            
            [12] The force on the container is calculated and printed.
            
            [13] The pressure is calculated using the force and the container's circumference,
            and printed.
            
        '''
        
        # Animation axes are set
        
        if animate:
            pl.figure(0)
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            for i in range(0, len(self._balls)):
                ax.add_artist(self._balls[i].get_patch())
        
        
        # Setting up a nested list for the balls' displacements
        
        hist_values1 = []
        for i in range(0, len(self._balls)):
            hist_values1.append([])
        
        
        # Finding the indexes of the collision of every ball with every other ball, 
        # excluding the container, setting up a nested list for the ball pairs for 
        # inter-ball separation
        
        pair_index = []
        
        for i in range(1, len(self._balls)):
            for j in range(1, len(self._balls)):
                if i!=j:
                    pair_index.append([i, j])
        
        hist_pairs = []
        for i in range(0, len(pair_index)):
            hist_pairs.append([])
        
        
        
        # Setting up lists for kinetic energy, momentum and velocity:
            
        kinetic_energy = []
        momentum = []
        velocity = []
  
        
        for frame in range(num_frames):
            
            # Running the simulation
            
            if animate:
                pl.pause(0.001)
            self.smooth_move()
    
            
            # Generating data for displacement histogram by finding the magnitude of each
            # particle's position vector, and adding to 'hist_values':
            
            for i in range(1, len(self._balls)):
                displace = ((self._balls[i].pos()[0])**2 + (self._balls[i].pos()[1])**2)**(1/2)
                hist_values1[i].append(displace)
                
            
            # Generating data for pairs histogram by finding the magnitude of each pair's
            # relative position, and adding to 'hist_pairs':
            
            for i in range(0, len(hist_pairs)):
                relative_position = self._balls[pair_index[i][0]].pos() - self._balls[pair_index[i][1]].pos()
                hist_pairs[i].append(((relative_position[0])**2 + (relative_position[1])**2)**(1/2))
            
            
            # Generating the velocities and kinetic energy for each frame by finding the
            # magnitude of each particle's velocity and adding to velocities.
            # These speeds were used to find the root mean square of the velocity, which
            # was then used to calculate the kinetic energy.
            
            velocities = []
            velocities_square = []
            
            for i in range(0, len(self._balls)):
                vel = ((self._balls[i].vel()[0])**2 + (self._balls[i].vel()[1])**2)**(1/2)
                velocities.append(vel)
                velocities_square.append(vel**2)
        
            velocities_meansquare = np.mean(velocities_square)
            rms = np.sqrt(velocities_meansquare)
            KE = 1/2 * self._mass * rms**2
            kinetic_energy.append(KE)
            
            velocity.append(velocities)
                    
            
            # Generating the mean momentum for each frame

            moms = []
            
            for i in range(0, len(self._balls)):
                moms.append(self.give_momentum(i))
        
            moms_mean = sum(moms) / len(moms)

            momentum.append(moms_mean)
            
        
        # [1]
        # The data is all combined to provide one histogram over all of the frames.
        # Plotting average displacement histogram
        
        ave_displacement_hist = []
        
        for i in hist_values1:
            i = i[500:]
            for j in i:
                ave_displacement_hist.append(j)
            
        plt.figure(1)
        plt.hist(ave_displacement_hist, bins=15, color='#9491BE')
        plt.title("Average distance histogram", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xlabel("Distance [m]", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        
        # [2]
        # The data is all combined to provide one histogram over all of the frames.
        # Plotting average pairs histogram
        
        ave_pairs_hist = []
        
        for i in hist_pairs:
            i = i[500:]
            for j in i:
                ave_pairs_hist.append(j)
                
        plt.figure(2)
        plt.hist(i, bins=10, color='#7AB356')
        plt.title("Average separation histogram", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xlabel("Inter-Ball Separation [m]", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        
        # [3]
        # Kinetic energy is plot over the whole time period
        
        time_range = np.arange(0, self._time, step=((self._time/num_frames)))
        
        if len(time_range) > num_frames:
            time_range = time_range[0:num_frames:]
        
        plt.figure(3)
        plt.scatter(time_range, kinetic_energy, color='orange')
        plt.ylim(0, 2*kinetic_energy[0])
        plt.ylabel("Kinetic Energy [J]", fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.title("Total kinetic energy of balls over time", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        # [4]
        print("The kinetic energy is constant, at a value of ", kinetic_energy[0], "J")
        
        temperature = (2/3 * kinetic_energy[0]) / (1.38*10**(-23))
        
        # [5]
        print("The temperature is ", temperature, "K")
        
        
        # [6]
        # Momentum is plot over the whole time period
        
        plt.figure(4)
        plt.scatter(time_range, momentum, color='green')
        plt.ylim(0, 2*momentum[0])
        plt.ylabel("Momentum $[kg m s^{-1}]$", fontsize=14)
        plt.xlabel("Time [s]", fontsize=14)
        plt.title("Total momentum of balls over time", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        # [7]
        print("The average momentum is constant, at a value of ", momentum[0], "kg m s^-1")
        
        
        # [8]
        # Plot velocity distributions
        
        ave_vel = []
        for i in velocity[100:]:
            for j in i:
                ave_vel.append(j)

        plt.figure(20)
        vals = plt.hist(ave_vel, bins=15, label="Speed", color='#3E8ADE')
        plt.ylabel("Frequency", fontsize=14)
        plt.xlabel("Speed $[m s^{-1}]$", fontsize=14)
        plt.title("Complete speed distribution histogram", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        
        
        # Normalised Maxwell-Boltzmann distribution, where a is the normalisation factor
        
        def Maxwell_Boltzmann(a, v, m, T):
            y = a * (m / (2 * np.pi * 1.38*10**(-23) * T))**(3/2) * 4 * np.pi * v**2 * np.exp((-m * (v)**2)/(2 * 1.38*10**(-23) * T))
            return y
        
        max_vel = max(ave_vel)
        
        area = sum(vals[0] * np.diff(vals[1]))
        x_vals = np.linspace(0, max_vel, 100)
        y_vals = Maxwell_Boltzmann(area, x_vals, self._mass, temperature)
        plt.plot(x_vals, y_vals, label="M-B Distribution", color='orange', linewidth=3)
        plt.legend(fontsize=12)
        
            
        if animate:
            pl.show()
        
        
        # Calculates and prints various useful values.
        
        area_ratio = ((len(self._balls)-1) * self._radius**2) / -(self._containerradius**2)
        force = self._mom / (self._time-(200*0.01))
        circumf = 2 * np.pi * -self.give_container_radius()
        pressure = force / circumf
        # [9]
        print("The ratio between the total balls' area and the container's area is: ", area_ratio)
        # [10]
        print("The total momentum applied to the container was: ", self._mom, "kg m^-1")
        # [11]
        print("The total time passed is: ", self._time, "s")
        # [12]
        print("The force on the container is ", force, "N")
        # [13]
        print("The pressure is: ", pressure, "Pa")
        
        
        # Adds all the temperature and pressure values to the inputted lists.
        
        self._temps.append(temperature)
        self._press.append(pressure)
        
    




