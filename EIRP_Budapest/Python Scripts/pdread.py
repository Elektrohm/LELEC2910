import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plotgeo(filename1, filename2, label):
    """filename1 : file containing values for angles from 0 to 50
       filename2 : file containing values for angles from 55 to 90
       label     : plot label

    """

    df1 = pd.read_csv(filename1, delimiter=",", skiprows=6)
    df2 = pd.read_csv(filename2, delimiter=",", skiprows=6)

    pdf = pd.read_csv('BudapestPDF.csv', delimiter=",")

    # number of point for an elevation_angle
    N1 = df1[' Satellite number'].value_counts().to_numpy()
    N_elev1 = df1[' Elevation angle (deg)'].nunique()
    N2 = df2[' Satellite number'].value_counts().to_numpy()
    N_elev2 = df2[' Elevation angle (deg)'].nunique()

    size1 = df1.shape[0]
    size2 = df2.shape[0]

    ff = []

    for i in range(N_elev1):
        att = df1[df1.keys()[-1]][i * N1[i]: (i + 1) * N1[i]]
        prob = df1[' Time exceeded (%)'][i * N1[i]: (i + 1) * N1[i]]
        ff.append(interp1d(att, prob))
        # plt.plot(ff[i](att), att)

    for i in range(N_elev2):
        att = df2[df2.keys()[-1]][i * N2[i]: (i + 1) * N2[i]]
        prob = df2[' Time exceeded (%)'][i * N2[i]: (i + 1) * N2[i]]
        ff.append(interp1d(att, prob))

    K = 100  # number of attenuations

    att_0_1 = np.min(df1.loc[df1[' Time exceeded (%)'] == 0.01][df1.keys()[-1]])
    att_max_1 = np.max(df1.loc[df1[' Time exceeded (%)'] == 99.99][df1.keys()[-1]])

    att_0_2 = np.min(df2.loc[df1[' Time exceeded (%)'] == 0.01][df2.keys()[-1]])
    att_max_2 = np.max(df2.loc[df1[' Time exceeded (%)'] == 99.99][df2.keys()[-1]])

    att = np.linspace(np.max([att_max_1, att_max_2]), np.min([att_0_1, att_0_2]), K)
    prob = np.zeros(K)

    for j in range(K):
        for i in range(N_elev1):
            index = np.where(pdf["theta"] == df1[' Elevation angle (deg)'][i * N1[i]])
            prob[j] += ff[i](att[j]) * 0.01 * pdf["PDF"].iloc[index]

    for j in range(K):
        for i in range(N_elev2):
            index = np.where(pdf["theta"] == df2[' Elevation angle (deg)'][i * N2[i]])
            prob[j] += ff[i](att[j]) * 0.01 * pdf["PDF"].iloc[index]

    maxi = np.amax(prob)
    plt.plot(prob / maxi * 100, att, label=label)


def Plot_effect(filename1, filename2, label):
    df1 = pd.read_csv(filename1, delimiter=",", skiprows=6)
    df2 = pd.read_csv(filename2, delimiter=",", skiprows=6)

    N1 = df1[' Satellite number'].value_counts().to_numpy()
    N_elev1 = df1[' Elevation angle (deg)'].nunique()
    N2 = df2[' Satellite number'].value_counts().to_numpy()
    N_elev2 = df2[' Elevation angle (deg)'].nunique()

    N1_time_exc = df1[' Time exceeded (%)'].nunique()
    N2_time_exc = df2[' Time exceeded (%)'].nunique()

    time_exc1 = df1[' Time exceeded (%)'].to_numpy()[:N1_time_exc]
    time_exc2 = df2[' Time exceeded (%)'].to_numpy()[:N2_time_exc]

    keys1 = df1.keys()
    keys2 = df2.keys()

    for i in range(N_elev1):
        att = df1[keys1[-1]][i * N1[i]: (i + 1) * N1[i]]
        plt.semilogx(time_exc1, att, label='{}°'.format(5 * (i + 1)))

    for i in range(N_elev2):
        att = df1[keys2[-1]][i * N2[i]: (i + 1) * N2[i]]
        plt.semilogx(time_exc1, att, label='{}°'.format(55 + 5 * i))

    plt.xlabel('Time exceeded [%]')
    plt.ylabel('Attenuation [dB]')
    plt.tight_layout()
    plt.legend(loc='right')
    # plt.title(label)
    plt.grid(True, which='both')
    plt.show()


def Plot_Budapest(frequency):
    filename1 = 'Budapest/ascii_' + str(frequency) + '_1/attenuation_total.csv'
    filename2 = 'Budapest/ascii_' + str(frequency) + '_2/attenuation_total.csv'
    plotgeo(filename1, filename2, 'total attenuation')

    plt.xscale('log')
    plt.ylabel('Attenuation [dB]')
    plt.xlabel('Availibity [%]')
    plt.legend()
    plt.grid(which='both')
    plt.tight_layout()
    plt.show()


def Plot_vapor(filename1, filename2, label):
    df1 = pd.read_csv(filename1, delimiter=",", skiprows=6)
    df2 = pd.read_csv(filename2, delimiter=",", skiprows=6)


def Plot_all_effects(frequency):
    filename1 = 'Budapest/ascii_' + str(frequency) + '_1/attenuation_scintillation.csv'
    filename2 = 'Budapest/ascii_' + str(frequency) + '_2/attenuation_scintillation.csv'
    Plot_effect(filename1, filename2, 'scintillation attenuation')

    filename1 = 'Budapest/ascii_' + str(frequency) + '_1/attenuation_cloud.csv'
    filename2 = 'Budapest/ascii_' + str(frequency) + '_2/attenuation_cloud.csv'
    Plot_effect(filename1, filename2, 'cloud attenuation')

    filename1 = 'Budapest/ascii_' + str(frequency) + '_1/attenuation_rain.csv'
    filename2 = 'Budapest/ascii_' + str(frequency) + '_2/attenuation_rain.csv'
    Plot_effect(filename1, filename2, 'rain attenuation')

    filename1 = 'Budapest/ascii_' + str(frequency) + '_1/attenuation_vapor.csv'
    filename2 = 'Budapest/ascii_' + str(frequency) + '_2/attenuation_vapor.csv'
    Plot_vapor(filename1, filename2, 'vapor')


if __name__ == '__main__':
    Plot_Budapest(75)
    # Plot_all_effects(75)
