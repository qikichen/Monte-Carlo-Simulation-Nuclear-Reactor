# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 18:10:07 2022

@author: Andreas Hadjichristou and Qi Nohr Chen
"""
import numpy.random as rnd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TRIALS = 1
iterations = 10
radius = 10 
height = 20
number_of_neutrons = 100
standard_temp = 293.15
eV = 1.6*10**(-19)
barn = 1*10**(-24)

percentage_of_neutrons_leaked = 0.033
parts_U235 = 0.5
parts_U238 = 0.5
parts_Graphite = 5
parts_total = parts_Graphite + parts_U235 + parts_U238
percentage_B10 = 0
percentage_U235 = parts_U235/parts_total
percentage_H20 = 0
percentage_D20 = 0
percentage_Graphite = parts_Graphite/parts_total
percentage_U238 = parts_U238/parts_total

th_scattering_cs_U235 = 10 * 10**(-24)
fa_scattering_cs_U235 = 4 * 10**(-24)
th_fission_cs_U235 = 579 * 10**(-24)
fa_fission_cs_U235 = 1* 10**(-24)
th_capture_cs_U235 = 101 * 10**(-24)
fa_capture_cs_U235 = 0.09 * 10 ** (-24)


th_capture_cs_B10 = 200 * 10**(-24)
th_scattering_cs_B10 = 2 * 10**(-24)
fa_capture_cs_B10 = 0.4 * 10**(-24)
fa_scattering_cs_B10 = 2  * 10**(-24)
th_scattering_cs_U238 = 8.3 *10**(-24)
fa_scattering_cs_U238 = 5 *10**(-24)
th_scattering_cs_H20 = 49.2 *10**(-24)
th_absorbtion_cs_H20 = 0.66 *10**(-24)
th_scattering_cs_D20 = 10.6 *10**(-24)
th_absorbtion_cs_D20 = 0.001  *10**(-24)
th_scattering_cs_Graphite = 4.7 *10**(-24)
th_absorbtion_cs_Graphite = 0.0045 *10**(-24)
th_fission_cs_U238 = 0.00002 *10**(-24)
fa_fission_cs_U238 = 0.3 * 10**(-24)
th_capture_cs_U238 = 2.72 * 10**(-24)
fa_capture_cs_U238 = 0.07 * 10**(-24)
density_U235 = 18.7
density_U238 = 18.9
density_H20 = 1.0
density_D20 = 1.1
density_Graphite = 1.6
density_B10 = 2.34

FISSION_ENERGY = 202.79 * (10 ** 6) * eV

def if_file_exists_read_data(filename):
    """
    The function checks if the specified data file exists and if it does,
    it reads the data from the file and stores it inside an array, otherwise
    it outputs an error message.

    Parameters
    ----------
    filename = csv file

    Returns
    -------
    data_array = array


    """
    file_open = False
    try:
        open(filename, 'r')
        file_open = True

    except FileNotFoundError:
        print('Unable to find ', filename)
    if file_open:
        data_array = np.genfromtxt(filename, delimiter=',', comments='%')
        return data_array

def prob_distribution_neutrons():
    #distribution to generate an average of 2.4 neutrons
    number_of_neutrons_released = [1,2,3]
    probabilities = [0.1,0.4,0.5]
    return np.random.choice(number_of_neutrons_released, p=probabilities)

def energy_decrement(atomic_number):
    A = (atomic_number-1)**2/(2*atomic_number)
    B = np.log((atomic_number+1)/(atomic_number-1))
    return 1-(A*B)

def generate_neutron_positions(n):
    z_array = np.array([])
    xy_array=np.array([])
    energy_array=np.array([])
    N = 0
    while n > N:
        
        positionx_y=rnd.uniform(-radius,radius,size=2)
        if np.linalg.norm(positionx_y)<=radius:
            xy_array = np.append(xy_array,  positionx_y)
            N = N + 1
    xy_array = np.reshape(xy_array, (-1, 2))
    for i in range(number_of_neutrons):
      z_array = np.append(z_array, rnd.uniform(0,height, size = 1))  
    z_array = z_array.reshape((-1, 1))
    xyz_array = np.hstack((xy_array,z_array))
    for i in range(number_of_neutrons):
        energy_array = np.append(energy_array, 2*10**6*eV)
    energy_array = energy_array.reshape((-1, 1))
    position_and_energy_array = np.hstack((xyz_array,energy_array))
    
    return position_and_energy_array 

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid 


def formula_for_adjusted_U235cross_section_fission(neutron_energy):
    neutron_energy = neutron_energy / eV
    
    if neutron_energy <2300:
         
         data_array = if_file_exists_read_data('U235_resonance_fission.csv')
         difference = abs(data_array[:,0]-neutron_energy)
         
         
         min_index = np.argmin(difference)
         return data_array[min_index,1]
    elif 2300<=neutron_energy<830000:
        return (1.112 + (6.457*np.exp(-neutron_energy/2260))+
                 1.895*np.exp(-neutron_energy/13133)+
                 0.935*np.exp(-neutron_energy/150132))
    else :
        return ((-0.74842 + 4.93377*10**(-6)*neutron_energy) 
                  -(4.9117*10**(-12)*(neutron_energy**2)) +
                  (2.56743*10**(-18)*(neutron_energy**3)) +
                   (-7.6643*10**(-25)*(neutron_energy**4)) +
                   (1.33429*10**(-31)*(neutron_energy**5)) +
                   (-1.33057*10**(-38)*(neutron_energy**6)) +
                   (7.02335*10**(-46)*(neutron_energy**7)) +
                   (-1.51932*10**(-53)*(neutron_energy**8)))

def formula_for_adjusted_U235cross_section_scattering(neutron_energy):
    neutron_energy = neutron_energy / eV
    if neutron_energy < 1090000:
        return 2.16625 + (11.14182-2.16625)/(1+(neutron_energy/349185.81852)**1.37786)
    elif 1090000<=neutron_energy<=3921371:
        return (5.72508-(3.39*10**(-6)*neutron_energy) +
                (1.63765*10**(-12)* neutron_energy**2)
                -(2.16331*10**(-19)*neutron_energy**3))
    else:
        return (3.90259 + 7.33673*10**(-7)*neutron_energy +
                -1.76776*10**(-13)*neutron_energy**2 +
                9.14875*10**(-21)*neutron_energy**3)
   
def formula_for_adjusted_U235cross_section_absorbtion(neutron_energy):
     neutron_energy = neutron_energy / eV
     
     if neutron_energy < 2000:
         data_array = if_file_exists_read_data('U235_resonance_capture.csv')
         difference = abs(data_array[:,0]-neutron_energy)
         
         
         min_index = np.argmin(difference)
         return data_array[min_index,1] 
          
     elif  2000<= neutron_energy < 761000:
         
         return 0.09427 + (0.71639-0.09427)/(1+(neutron_energy/122272.81077)**1.57674)
     else:
         return  (0.17247-(7.60782*10**(-8)*neutron_energy) +
                (1.10557*10**(-14)* neutron_energy**2)
                -(5.19723*10**(-22)*neutron_energy**3))
         
def formula_for_adjusted_U238cross_section_scattering(neutron_energy):
     neutron_energy = neutron_energy / eV
    
     if neutron_energy <= 6.497:
         return 8.55276 - 9.22238*10**(-5)*(np.exp(1.84672*neutron_energy)-1)/1.84672
     
     elif 6.497 < neutron_energy <= 19000:
         data_array = if_file_exists_read_data('U238_resonance_scattering.csv')
         difference = abs(data_array[:,0]-neutron_energy)
         
         
         min_index = np.argmin(difference)
         return data_array[min_index,1] 
     else:  
         return  ((12.66086 + -1.61993*10**(-5)*neutron_energy) 
                  +(1.02884*10**(-11)*(neutron_energy**2)) -
                  (3.00306*10**(-18)*(neutron_energy**3)) +
                   (4.44771*10**(-25)*(neutron_energy**4)) -
                   (3.26516*10**(-32)*(neutron_energy**5)) +
                   (9.43884*10**(-40)*(neutron_energy**6)))
     
def formula_for_adjusted_U238cross_section_absorbtion(neutron_energy):
     neutron_energy = neutron_energy/eV 
     
         
        
         
     if  neutron_energy <= 21000:
         data_array = if_file_exists_read_data('U238_resonance_capture.csv')
         difference = abs(data_array[:,0]-neutron_energy)
         
         
         min_index = np.argmin(difference)
         return data_array[min_index,1] 
     else:  
         return  ((0.3183*np.exp(-neutron_energy/1048870.54535) )
                  -0.00136*2.1489*10**(-10)*neutron_energy)
    
     
def formula_for_adjusted_Graphite_section_scattering(neutron_energy):
     neutron_energy = neutron_energy / eV
    
     
     data_array = if_file_exists_read_data('Graphite_scattering.csv')
     
     difference = abs(data_array[:,0]-neutron_energy)
         
         
     min_index = np.argmin(difference)
     return data_array[min_index,1] 
    

   
def generate_comparison_points(spherical, cylindrical):
    """
    The spherical and cylindrical inputs are True or False statements. This
    function generates sphere sample points or cylinder sample points depending
    on the boolean that you input. These are used to compare
    """
    sample_cycles = 10000
    theta = rnd.uniform(0, np.pi, sample_cycles).T
    phi = rnd.uniform(0, 2*np.pi, sample_cycles).T
    sample_sphere = np.empty((0,3))
    sample_cylinder = np.empty((0,3))
    
    #Spherical sample points that are generated
    if spherical == True: 
        for index in range(len(theta)):
            x_values = radius * np.cos(phi[index]) * np.sin(theta[index])
            y_values = radius * np.sin(phi[index]) * np.sin(theta[index])
            z_values = radius * np.cos(theta[index])
            collect = np.column_stack((x_values, y_values, z_values))
            sample_sphere = np.vstack((sample_sphere, collect))
            
        return sample_sphere
    
    #Cylindrical sample points that are generated
    elif cylindrical == True:
        z_values = rnd.uniform(0, height, sample_cycles).T
        RADII = rnd.uniform(-radius, radius, sample_cycles).T
        for index in range(len(theta)):
                x_circle_top = RADII[index] * np.cos(phi[index])
                y_circle_top = RADII[index] * np.sin(phi[index])
                z_circle_top = height 
                x_circle_bot = RADII[index] * np.cos(phi[index])
                y_circle_bot = RADII[index] * np.sin(phi[index])
                z_circle_bot = 0
                x_values = radius * np.cos(phi[index]) 
                y_values = radius * np.sin(phi[index]) 
                z = z_values[index]
                x_values = np.vstack((x_values, x_circle_top, x_circle_bot))
                y_values = np.vstack((y_values, y_circle_top, y_circle_bot))
                z = np.vstack((z, z_circle_top, z_circle_bot))
                collect = np.column_stack((x_values, y_values, z))
                sample_cylinder = np.vstack((sample_cylinder, collect))
                
        return sample_cylinder
    
def respawn_neutrons(outside_neutron, sample_shape):
    """
    It takes in a neutron with coordinates outside and the array for the sample
    shape which would be a data array full of points that define the sphere
    or cylinder
    """
    
    difference = np.empty((0,1))
    
    for point in range(len(sample_shape)):
        x_coord_diff = sample_shape[point, 0] - outside_neutron[0] 
        y_coord_diff = sample_shape[point, 1] - outside_neutron[1] 
        z_coord_diff = sample_shape[point, 2] - outside_neutron[2] 
        modulus = np.sqrt(x_coord_diff**2 + y_coord_diff**2 + z_coord_diff**2)
        difference = np.vstack((difference, modulus))
    
    minimum = np.argmin(difference)
    new_position = np.reshape(sample_shape[minimum],(1,3))
    return new_position

    
            
def Watt_probability_distribution():
    mn=0 # Lowest value of domain
    mx=10 # Highest value of domain
    bound=0.3574# Upper bound of PDF value
    while True: # Do the following until a value is returned
       # Choose an X inside the desired sampling domain.
       x=rnd.uniform(mn,mx)
       # Choose a Y between 0 and the maximum PDF value.
       y=rnd.uniform(0,bound)
       # Calculate PDF
       pdf=0.4865*np.sinh(np.sqrt(2*x))*np.exp(-x)
       # Does (x,y) fall in the PDF?
       if y<pdf:
           # Yes, so return x in J
           return x * 10**6 * eV
       # No, so loop           
       
            
    


    
initial_positions = generate_neutron_positions(number_of_neutrons)


def final_position_and_energy(initial_position):
    position = []
    
    fission_counter = 0
    neutrons_leaked = 0
    neutrons = len(initial_position)
    condition = True
    average_energy_per_neutron = np.average(np.array(initial_position[:,3]))
    
    for i in range(neutrons):
        dnew =rnd.uniform(-1,1, size=3)
        while condition:
            dnew=rnd.uniform(-1,1, size=3)
            
            if np.linalg.norm(dnew)<= 1:
                condition = False
                
        unit_vector = dnew/np.linalg.norm(dnew)
        unit_vector = np.array(unit_vector)
        
        # Decide cross section and therefore distance travelled by using neutron energy
        random_distance, process = distance_travelled(initial_position[i][3])
        print(process)
        print(np.linalg.norm(random_distance))
        
        
        
       
        extra_distance = random_distance * unit_vector
        extra_distance = np.append(extra_distance,0)
        
        
        move_position = initial_position[i]+ extra_distance
        
        
    
        
    
        radius_from_origin = np.sqrt(move_position[1]**2+move_position[0]**2)
    
          
        if (radius_from_origin < radius) and (0 < move_position[2] < height):
            move_position = move_position   
            escape = False
            if process =='Fission':
                new_neutrons_added = prob_distribution_neutrons()
                fission_counter = fission_counter +1
                
                for i in range(new_neutrons_added):
                    energy_of_neutron = Watt_probability_distribution()
                    
                    position = np.hstack((position, move_position[0:3], energy_of_neutron))
                           
                    
            elif process =='Scattering':
                number_density_U235 = ((density_U235 * 6.022 * 10 ** 23)/235)*(percentage_U235)
                number_density_U238 = ((density_U238 * 6.022 * 10 ** 23)/238)*(percentage_U238)
                number_density_Graphite = number_density_moderator()[2]
                energy_of_neutron = move_position[3]
                scattering_U235 = number_density_U235 * formula_for_adjusted_U235cross_section_scattering(energy_of_neutron)*barn
                scattering_U238 = number_density_U238 * formula_for_adjusted_U238cross_section_scattering(energy_of_neutron)*barn
                scattering_Graphite = number_density_Graphite * formula_for_adjusted_Graphite_section_scattering(energy_of_neutron)*barn
                total_scattering = scattering_U235 + scattering_U238 + scattering_Graphite 
                prob_U235 = scattering_U235/total_scattering
                prob_U238 = scattering_U238/total_scattering
                prob_Graphite = scattering_Graphite/total_scattering
                #throw dice to decide material by which neutron was scattered
                #according to percentage of scattering material out of total scattering material
                scattering_materials = ['U235','U238','H20','D20', 'Graphite', 'Boron']
                probabilities = [prob_U235, prob_U238, percentage_H20, percentage_D20, prob_Graphite, percentage_B10]
                scattering_material=np.random.choice(scattering_materials, p=probabilities)
                if scattering_material == 'U235':
                    energy_of_neutron = move_position[3]/np.exp(energy_decrement(235))
                    print('Scattered by U235')
                elif scattering_material == 'U238': 
                    energy_of_neutron= move_position[3]/np.exp(energy_decrement(238))
                    print('Scattered by U238')
                elif scattering_material == 'H20':
                    energy_of_neutron= move_position[3]/np.exp(energy_decrement(18))
                elif scattering_material == 'D20':
                    energy_of_neutron = move_position[3]/np.exp(energy_decrement(20))
                elif scattering_material == 'Graphite':
                    energy_of_neutron = move_position[3]/np.exp(energy_decrement(12))
                    print('Scattered by Graphite')
                else:
                    energy_of_neutron = move_position[3]/np.exp(energy_decrement(10))
                position=np.hstack((position, move_position[0:3], energy_of_neutron))
            
        else:
            #checking if the number of neutrons leaked is correct
            #print('neutron leaked')
            escape = True
            
            if  rnd.uniform(0,1,size=1) > percentage_of_neutrons_leaked:
                escape = False
                if not escape:
                    
                   
                    # print(new_position)
                    print('Neutron reflected')
                    
                    final_position = move_position[0:3]
                    while not ((radius_from_origin <= radius) and (0 <= final_position[2] <= height)):
                        boundary_position = respawn_neutrons(final_position, 
                                    generate_comparison_points(
                                        spherical=False, cylindrical=True))
                        boundary_position = np.array(boundary_position[0,:])
                        
                        distance = np.linalg.norm(final_position-boundary_position)
                        
                        x = boundary_position[0]
                        y = boundary_position[1]
                        z= boundary_position[2]
                        if (z == height) or (z == 0):
                            #unit vector to surface
                            transformation = [1,1,-1]
                            new_unit_vector = np.multiply(unit_vector,transformation)
                            move_distance = new_unit_vector * distance
                            final_position = boundary_position + move_distance 
                            unit_vector = new_unit_vector
                            energy_of_neutron = initial_position[i][3]/np.exp(energy_decrement(9))
                            
                            
                        else:
                           n = [x/(np.sqrt(x**2+y**2)),y/(np.sqrt(x**2+y**2)),0]
                           n = np.array(n)
                           reflected_unit_vector = np.subtract(unit_vector,np.multiply(2*np.dot(-n,unit_vector),-n))
                           
                           move_distance = reflected_unit_vector * distance
                           final_position = boundary_position + move_distance 
                           unit_vector = reflected_unit_vector
                           energy_of_neutron = initial_position[i][3]/np.exp(energy_decrement(9))
                           
                           
                        
                        radius_from_origin = np.sqrt(final_position[1]**2+final_position[0]**2)
                        
                    
                    position = np.hstack((position, final_position, energy_of_neutron))
               
        if escape:
            
            neutrons_leaked = neutrons_leaked + 1
        
            
        
           
                
           
         
      
    position = np.reshape(position,[-1,4])
    print('Number of neutrons inside reactor is: ', len(position))
    
    return position , neutrons_leaked, fission_counter, (average_energy_per_neutron/(eV*10**6))

def k_factor_against_iteration(k_fact):
    """
    Calculates the average energy per neutron per iteration
    """
    
    number_of_iterations = np.array((list(range(0,iterations+1))))
   
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='Iteration', ylabel='k factor', 
            title='Iteration against k factor') 
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(number_of_iterations[1:], k_fact[0:])
    plt.show()
    

def metropolis_random_walk():
    global k
    ytotal = 0
    k = 1
    start = initial_positions
    FISSIONS = [0]
    average_per_iteration = [0]
    k_data = []
    for i in range(iterations):
       
       old_gen_neutrons = len(start) 
       
       x,y,c, aver = final_position_and_energy(start)
       FISSIONS.append(c)
       average_per_iteration.append(aver)
       
       start = x
       
       new_gen_neutrons = len(x)
      
       ytotal = ytotal + y
       
       
       try:
           
           k = new_gen_neutrons/old_gen_neutrons
           
       except ZeroDivisionError:
             print('No neutrons inside the reactor')
         
       print('Multiplicity factor k: ' ,k,'for iteration ', i+1)
       k_data.append(k)
    print('The total number of neutrons leaked is: ', ytotal)  
    return x,c, FISSIONS, average_per_iteration, k_data

def control_rod_material(k_factor):
     if k_factor > 1.5 :
         number_density = (density_B10 * 6.022 * 10 ** 23)/10*percentage_B10
     else:
         number_density = 0
     return number_density
 
def number_density_moderator():
    number_density_H20 = ((density_H20 * 6.022 * 10 ** 23)/18) * (percentage_H20)
    number_density_D20 = ((density_D20 * 6.022 * 10 ** 23)/20) * (percentage_D20)
    number_density_Graphite = ((density_Graphite * 6.022 * 10 ** 23)/12) * percentage_Graphite
    return number_density_H20, number_density_D20, number_density_Graphite


    # new method for finding distance travelled:
    
def macroscopic_cs(neutron_energy):
    number_density_U235 = ((density_U235 * 6.022 * 10 ** 23)/235)*(percentage_U235)
    number_density_U238 = ((density_U238 * 6.022 * 10 ** 23)/238)*(percentage_U238)
    number_density_H20 = number_density_moderator()[0]
    number_density_D20 = number_density_moderator()[1]
    number_density_Graphite = number_density_moderator()[2]
    number_density_B10 = control_rod_material(k)
    macroscopic_cs_scattering = (number_density_U235 * (formula_for_adjusted_U235cross_section_scattering(neutron_energy)*barn)+
                                 number_density_U238 * (formula_for_adjusted_U238cross_section_scattering(neutron_energy)*barn)+
                                 number_density_D20 * th_scattering_cs_D20+
                                 number_density_H20 * th_scattering_cs_H20+
                                 number_density_Graphite * formula_for_adjusted_Graphite_section_scattering(neutron_energy)*barn)
                                 
    macroscopic_cs_fission =    (number_density_U235 * (formula_for_adjusted_U235cross_section_fission(neutron_energy)*barn)+
                                 number_density_U238 * th_fission_cs_U238)
    macroscopic_cs_capture =     (number_density_U235 * (formula_for_adjusted_U235cross_section_absorbtion(neutron_energy)*barn)+
                                 number_density_U238 * (formula_for_adjusted_U235cross_section_absorbtion(neutron_energy)*barn)+
                                 number_density_D20 * th_absorbtion_cs_D20+
                                 number_density_H20 * th_absorbtion_cs_H20+
                                 number_density_Graphite * th_absorbtion_cs_Graphite+
                                 number_density_B10 * th_capture_cs_B10)
    
    total_macroscopic_cs = (macroscopic_cs_capture + macroscopic_cs_fission +  
                            macroscopic_cs_scattering) 
    gamma = rnd.uniform(0,1,size=1)
    if gamma < (macroscopic_cs_scattering/total_macroscopic_cs):
        result = 'Scattering'
    elif (macroscopic_cs_scattering/total_macroscopic_cs) < gamma < ((macroscopic_cs_scattering+macroscopic_cs_capture)/total_macroscopic_cs):
        result = 'Capture'
    elif gamma > ((macroscopic_cs_scattering+macroscopic_cs_capture)/total_macroscopic_cs):
        result = 'Fission'
    return total_macroscopic_cs, result

def distance_travelled(neutron_energy): 
     macroscopic_cross_section, mode = macroscopic_cs(neutron_energy)
     mean_free_path = 1/(macroscopic_cross_section * 100)
     random_distance = np.random.exponential(mean_free_path)
     return random_distance, mode
 


def energy_grapher_per_iteration(fissions):
    """
    Takes in a list of fissions and multiplies it to make a suitable graph
    """
    
    energy_values = np.array((fissions)) * FISSION_ENERGY #in MeV
    
    number_of_iterations = np.array((list(range(0,iterations+1))))
   
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='Iteration', ylabel='Energy in MeV', title='Iteration vs Energy') 
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.step(number_of_iterations, energy_values, 'r')
    plt.grid()
    plt.show()
    
def cumulative_energy(fissions):
    """
    Calculates the cumulative aka total energy
    """
    energy_values = np.cumsum(np.array((fissions)) * (FISSION_ENERGY)) #in J
    
    
    number_of_iterations = np.array((list(range(0,iterations+1))))
   
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='Iteration', ylabel='Energy in J', title='Cumulative Energy')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(number_of_iterations, energy_values, 'r')
    plt.grid()
    plt.show()
    
def average_energy_per_neutron(average_energy_per_ite):
    """
    Calculates the average energy per neutron per iteration
    """
    
    
    number_of_iterations = np.array((list(range(0,iterations+1))))
   
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='Iteration', ylabel='Average Energy of Neutron / MeV', 
            title='Average Energy of Neutron in each iteration') 
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(number_of_iterations[1:], average_energy_per_ite[1:], 'r')
    plt.grid()
    plt.show()
    
def graph_for_average(uncertainty_array, k_array):
    
    """
    Calculates the average energy per neutron per iteration
    """
    
    
    number_of_iterations = np.array((list(range(0,iterations+1))))
   
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set(xlabel='Iteration', ylabel='Average k of each iteration', 
            title='Average k value in each iteration') 
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.errorbar(number_of_iterations[1:], k_array, yerr=uncertainty_array, color='red', fmt='o', ecolor='black')
    plt.grid()
    plt.show()
    
def experiment():      
    fig1 = plt.figure() 
    ax1 = fig1.add_subplot(111, projection='3d')
    Xc,Yc,Zc = data_for_cylinder_along_z(0,0,radius,height)
    ax1.plot_surface(Xc, Yc, Zc, alpha=0.1)
    
    # Data for three-dimensional for initial scattered points
    zdata = initial_positions[:,2]
    xdata = initial_positions[:,0]
    ydata = initial_positions[:,1]
    ax1.scatter3D(xdata, ydata, zdata, color='black', alpha = 1);
    plt.show()
    
                              
    fig2 = plt.figure() 
    ax2 = fig2.add_subplot(111, projection='3d')
    Xc,Yc,Zc = data_for_cylinder_along_z(0,0,radius,height)
    
    ax2.plot_surface(Xc, Yc, Zc, alpha=0.1)
    
    
        
    
    end_positions,c, FIS, average, k_data = metropolis_random_walk()
    
    final_number_neutrons = len(end_positions)
    
    
    
    newzdata = end_positions[:,2]
    newxdata = end_positions[:,0]
    newydata = end_positions[:,1]
    ax2.scatter3D(newxdata, newydata, newzdata,  color='red', alpha = 1);
    plt.show()  
    
    fig3 = plt.figure() 
    
    ax3 = fig3.add_subplot(111)
    theta = np.linspace(0, 2*np.pi, 50)
    x_grid = radius*np.cos(theta) 
    y_grid = radius*np.sin(theta) 
    ax3.plot(x_grid,y_grid, c='black')
    ax3.scatter(initial_positions[:,0], initial_positions[:,1], c='r')
    ax3.set(xlabel='x-position', ylabel='y-position', title='Confirmation Plot of Uniform Distribution') 
    plt.show()                             
    
    #Plot histogram to view distribution in r
    square_radius_array=[]
    for j in range(final_number_neutrons):
        square_radius = (np.sqrt(end_positions[j][0]**2+end_positions[j][1]**2))**2
        square_radius_array= np.append(square_radius_array, square_radius)
        
    plt.hist(square_radius_array, bins = 10)
    plt.xlabel('r^2 position of neutron from origin')
    plt.ylabel('Frequency')         
    plt.title('Confirmation Histogram of Uniform Distribution (r^2 vs Frequency)')
    plt.axhline()
    plt.show()
    
    #Plot histogram to view distribution in r
    z_array=[]
    for j in range(final_number_neutrons):
        z_position = end_positions[j][2]
        z_array= np.append(z_array, z_position)
    plt.hist(z_array, bins = 100)
    plt.xlabel('Z position of neutron')
    plt.show()
    
    energy_grapher_per_iteration(FIS)
    cumulative_energy(FIS)
    average_energy_per_neutron(average)
    k_factor_against_iteration(k_data)
    
    
    return k_data

def main():
    k_data_array = np.empty((0,iterations))
    standard_deviations = np.empty((0,1))
    averages = np.empty((0,1))
    
    for x in range(TRIALS):
        k_data = experiment()
        k_data_array = np.vstack((k_data_array, k_data))
        
    for x in range(iterations):
        stdv_iteration_x = np.std(k_data_array[:,x])
        standard_deviations = np.vstack((standard_deviations,(stdv_iteration_x)))
        mean_iterations_x = np.mean(k_data_array[:,x])
        averages = np.vstack((averages,(mean_iterations_x)))
        
    final_average = np.mean(averages[15:iterations])
    final_standard_deviation = (np.linalg.norm(standard_deviations[15:iterations]))/np.sqrt(24)
    print("standard_deviation_array ", standard_deviations)
    print("average_array ", averages)
    print("final average ", final_average)
    print("final stdv ", final_standard_deviation)
    
    
    return standard_deviations, averages, final_average, final_standard_deviation
        
array_std_v, array_average, result_average, result_std_v = main()
graph_for_average(array_std_v, array_average)

