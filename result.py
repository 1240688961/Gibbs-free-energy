# coding=utf-8
import math
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, threshold='nan')


def calculation_gibbs(temperature, pressure, adsorption_energy):
    elementary_charge = 1.602e-19
    avogadro_constant = 6.02e23
    standard_pressure = 101325
    entropy_c2h4 = calculate_entropy(temperature, pressure, "c2h4")
    entropy_h2 = calculate_entropy(temperature, pressure, "h2")
    entropy_c2 = unit_conversion_c2(temperature)
    gbs = -52.26 - adsorption_energy*elementary_charge*avogadro_constant/1000 - temperature*(entropy_h2*2 + entropy_c2 - entropy_c2h4)/1000
    return gbs


def calculate_entropy(temperature, pressure, molecule):
    c_pm = 0.0
    entropy_0 = 0.0
    gas_constant = 8.314
    standard_pressure = 101325
    standard_temperature = 273.15
    if molecule == 'c2h4':
        entropy_0 = 219.5  # calculate the C2H4_entropy
        c_pm = (((3.919 - 3.416) / (1473.15 - 1033.15)) * (temperature - 1033.15) + 3.416) * (12.011 * 2 + 1.008 * 4)
    elif molecule == 'h2':
        entropy_0 = 130.6  # calculate the H2_entropy
        c_pm = (((16.25 - 15.02) / (1473.15 - 1033.15)) * (temperature - 1033.15) + 15.02) * (1.008 * 2)

    entropy = c_pm * math.log(temperature, math.e) - gas_constant * math.log(pressure, math.e) + (
            entropy_0 - c_pm * math.log(standard_temperature, math.e) + gas_constant * math.log(standard_pressure,
                                                                                                math.e))

    return entropy


def read_grafreqgp():
    # Read the data from the gra.freq.gp
    data = []
    with open('gra.freq.gp', 'r') as rf:
        line = rf.readline()
        while line:
            data.append([float(x) for x in line.split()])
            line = rf.readline()
    # Unit conversion
    data = np.array(data)
    for i in range(1, data.shape[1]):
        data[:, i] = (data[:, i] * 299792458 * 100) / 1e12
    return data


def calculate_entropy_c2(frequence,temperature):
    planck = 4.13566743e-15  # Planck constant (electron volt)
    boltzmann_charge = 1.3806505e-23 / 1.60217653e-19  # Boltzmann constant / Basic charge
    Thz = 1e12
    data = -boltzmann_charge * math.log(1 - math.exp(-planck * frequence * Thz / (boltzmann_charge * temperature)))
    return data


def make_calculation(data,temperature):
    # Calculation G-M，skip the G-point
    data_GM_total = 0.0
    for i in range(1, 50):
        for j in range(1, data.shape[1]):
            data_GM_total = data_GM_total + calculate_entropy_c2(data[i, j],temperature)
    # Calculation M-K
    data_MK_total = 0.0
    for i in range(50, 100):
        for j in range(1, data.shape[1]):
            data_MK_total = data_MK_total + calculate_entropy_c2(data[i, j],temperature)*2
    # Calculation K-G,skip the G
    data_KG_total = 0.0
    for i in range(100, 150):
        for j in range(1, data.shape[1]):
            data_KG_total = data_KG_total + calculate_entropy_c2(data[i, j],temperature)
    # Total weightings
    total_entropy = (data_GM_total + data_MK_total + data_KG_total) / 199
    return total_entropy


def unit_conversion_c2(temperature):
    elementary_charge = 1.602e-19
    avogadro_constant = 6.02e23
    data_read = read_grafreqgp()
    data_cal = make_calculation(data_read,temperature)
    data1 = data_cal * elementary_charge * avogadro_constant
    return data1


if __name__ == '__main__':
    fig=plt.figure()
    axis_x = np.arange(760+273, 1200+273, 1)
    #Plot the Pt_1L
    axis_y = []
    for i in axis_x:
        axis_y.append(calculation_gibbs(i, 25, 0.226))
    plt.plot(axis_x, axis_y, color='blue', linewidth=1.2, linestyle='-', label='Pt_1L_1273K')
    #Plt the Pt_2L
    axis_y = []
    for i in axis_x:
        axis_y.append(calculation_gibbs(i, 25, 0.106))
    plt.plot(axis_x, axis_y, color='blue', linewidth=1.2, linestyle='--', label='Pt_2L_1273K')
    #Plt the Cu_1L
    axis_y = []
    for i in axis_x:
        axis_y.append(calculation_gibbs(i, 25, 0.190))
    plt.plot(axis_x, axis_y, color='red', linewidth=1.2, linestyle='-', label='Cu_1L_1273K')
    #Plt the Cu_2L
    axis_y = []
    for i in axis_x:
        axis_y.append(calculation_gibbs(i, 25, 0.100))
    plt.plot(axis_x, axis_y, color='red', linewidth=1.2, linestyle='--', label='Cu_2L_1273K')

    plt.legend(loc = 'upper right')
    #Set the xlable and ylable and title
    plt.ylabel("Gibbs Free Energy (KJ/mol)", size=15)
    plt.xlabel("Temperature (K)", size=15)

    plt.show()
