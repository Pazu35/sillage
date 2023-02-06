import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg


import gpxpy
import gpxpy.gpx

# Parsing an existing file:
# -------------------------

def convert(d):
    d = abs(d)
    D = int(d)
    M = (d - D) * 60
    return float("{0:2d}{1:7.4f}".format(D, M))

def get_latlon_as_in_nmea_from_gpx(filename):
    gpx_file = open(filename, 'r')

    gpx = gpxpy.parse(gpx_file)
    lat = []
    lon = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                lat.append(convert(point.latitude))
                lon.append(convert(point.longitude))
    return np.array(lat), np.array(lon)



def butter_passe_bande(fc_inf,fc_sup, fs, order):
    nyq = 0.5 * fs
    b, a = sg.butter(order, (fc_inf/nyq, fc_sup/nyq), btype='bandpass', analog=False)
    return b, a


def freq_filter(l):
    order = 2
    fs = 50
    fc_inf, fc_sup = 0.2 , 0.6

    #Calcul de la réponse et du filtre
    b, a = butter_passe_bande(fc_inf, fc_sup, fs, order)
    y = sg.lfilter(b, a , l)
    return y



def detect_points(data, seuil, delta_p):
    detect = [i for i in range(len(data)) if data[i] > seuil]
    i = 0
    res = []
    while i < len(detect) - 1:
        data_i = detect[i]
        while detect[i] - data_i < delta_p:
            i+= 1
            if i == len(detect):
                i -= 1
                break
        data_t = detect[i-1]
        i+=1
        res += [((data_i + data_t)//2, [data_i, data_t])]
    return res


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
 
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])

    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
 
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
 
    R = np.dot(R_z, np.dot( R_y, R_x ))
 
    return R


def eulermat2angles(R):
    φ=np.arctan2(R[2,1],R[2,2])
    θ=-np.arcsin(R[2,0])
    ψ=np.arctan2(R[1,0],R[0,0])
    return φ,θ,ψ


def tolist(w): return list(w.flatten())

def adjoint(w):
    if isinstance(w, (float, int)): return array([[0,-w] , [w,0]])
    w=tolist(w)
    return np.array([[0,-w[2],w[1]] , [w[2],0,-w[0]] , [-w[1],w[0],0]])


def expw(w): return expm(adjoint(w))




def log_gps(file_name, data = True):
    '''
    Reading a GPS log file GPGGA frame
    '''

    # https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm
    # Format : GPGGA, UTC time, Lat DDmm.mm, North/South, Lon DDmm.mm, East/West, Quality, Nb sats, Hdop, Altitude, Alt unit(M), Undulation, Und unit(M), Age (s)

    # https://docs.novatel.com/OEM7/Content/Logs/GPGGA.htm#GPSQualityIndicators
    # Quality Indicator : 0 invalid, 1 Single point + Converging PPP, 2 Pseudorange differential + Converged and converging PPP, 
    #                     4 RTK fixed solution, 5 RTK floating solution + Converged PPP, 6 Dead Reckoning, 7 Manual Input
    #                     8 Simulator mode, 9 WAAS (SBAS) 


    time = []
    gps_time = []
    latitudes = []
    longitudes = []
    qualities = []
    satelites = []
    hdops = []
    altitudes = []


    f = open(file_name, 'r')

    count = 0
    for line in f:
        # print("\n")
        l = line.strip()
        l = l.split(';') # Split along the log tags
        # print("l1 : ", l)
        date = l[0].split(' ') # Split the date from the header
        # print("Date : ", date)
        l = l[2].split(', ') # Split the GPGGA gps frame


        # cnt = 0
        # for i in l:
        #   print('Element', cnt, ' : ', i)
        #   cnt +=1
        # print("l2 : ", l)


        try :
            # Computer time
            intermediate = date[1].split(':')
            t_i = float(intermediate[0])*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", "."))
            time += [t_i]

            # GPS time
            t_gpst = (l[1][1:-1])  # l[1] is the time hhmmss.ms, and [1:-1] is to get rid of the '' around it
            t_gps = float(t_gpst[:2])*3600 + float(t_gpst[2:4])*60 + float(t_gpst[4:])
            gps_time += [t_gps]

            # Latitude
            lat = l[2][1:-1]
            if lat != '':
                lat = float(lat)
            else:
                lat = None
            latitudes += [lat]


            # Longitude
            Lon = l[4][1:-1]
            if Lon != '':
                Lon = float(Lon)
            else:
                Lon = None
            longitudes += [Lon]

            # Quality
            Qual = l[6][1:-1]
            if Qual != '':
                Qual = float(Qual)
            else:
                Qual = None
            qualities += [Qual]

            # Number of satellites
            Nb_sat = l[7][1:-1]
            if Nb_sat != '':
                Nb_sat = float(Nb_sat)
            else:
                Nb_sat = None
            satelites += [Nb_sat]

            Hdop = l[8][1:-1]
            if Hdop != '':
                Hdop = float(Hdop)
            else:
                Hdop = None

            hdops += [Hdop]

            # Altitude
            Altitude = l[9][1:-1]
            if Altitude != '':
                Altitude = float(Altitude)
            else:
                Altitude = None
            altitudes += [Altitude]

            # print('time : ', t_i, ', ', intermediate)
            # print('t_gps : ', t_gps, ', ', t_gpst)    
            # print('Lat : ', lat)
            # print('Lon : ', Lon)
            # print('Qual : ', Qual)
            # print('Nb_sat : ', Nb_sat)
            # print('Hdop : ', Hdop)
            # print('Altitude : ', Altitude)

            # print("Delta time : ", t_i - t_gps)


        # print('t_gps : ', float(l[1]))
        # time.append(t_i)
        # detect_times.append(float(l[0]))

        except:
            print("Log error in file " + file_name)
            print('Line : ', line)

    if data :
        return np.array(time), np.array(gps_time), np.array(latitudes),np.array(longitudes), np.array(altitudes)
    else:
        return np.array(time), np.array(gps_time), np.array(latitudes),np.array(longitudes), np.array(altitudes), np.array(qualities), np.array(satelites), np.array(hdops)



def log_file(file_name, alignement = 0.):

    time = []
    acc = []
    gyr = []
    mag = []
    temp = []
    rate = []

    f = open(file_name, 'r')

    count = 0
    for line in f:
        l = line.strip()
        l = l.split(';')
        date = l[0].split(' ')
        l = l[2].split(',')
        if count < 30:
            count +=1
        else :
            try :
                intermediate = date[1].split(':')
                t_i = float(intermediate[0])*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", ".")) + alignement
                time.append(t_i)
                acc.append([float(l[0]), float(l[1]), float(l[2])])
                gyr.append([float(l[3]), float(l[4]), float(l[5])])
                mag.append([float(l[6]), float(l[7]), float(l[8])])
                temp.append(float(l[9]))
                rate.append(float(l[10]))
            except:

                print("Log error in file " + file_name)
                print('Line : ', line)

    return np.array(time), np.array(acc)*0.01, np.array(gyr)*np.pi/180, np.array(mag), np.array(temp), np.array(rate)


def log_file_detect(file_name, alignement = 0.):

    time = []
    detect_times = []

    f = open(file_name, 'r')

    count = 0
    for line in f:

        l = line.strip()
        l = l.split(';')
        date = l[0].split(' ')
        l = l[2].split(',')
        try :
            intermediate = date[1].split(':')
            t_i = float(intermediate[0])*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", ".")) + alignement
            time.append(t_i)
            detect_times.append(float(l[0]))

        except:
            print("Log error in file " + file_name)
            print('Line : ', line)

    return np.array(time), np.array(detect_times)



def RK(p, vr, R , ar, wr, dt):
    b1 = R @ vr
    b2 = R.T @ np.array([[0, 0, -9.80985]]).T + ar - np.cross(wr.flatten(), vr.flatten()).reshape(3,1)
    v1 = R @ expw(2 * dt / 3 * wr)
    v2 = R @ expw(dt * wr)
    Raux = R
    R = R @ expw(2 * dt / 3 * wr)
    b3 = R @ (b2 * 2 * dt / 3 + vr)
    b4 = R.T @ np.array([[0, 0, -9.80985]]).T + ar - np.cross(wr.flatten(), (b2 * 2 * dt / 3 + vr).flatten()).reshape(3,1)
    p = p + b1 * dt / 4 + b3 * 3 * dt / 4
    vr = b2 * dt / 4 + vr + b4 * 3 * dt / 4
    R = Raux @ expw(dt * wr)
    return (p, vr, R)


def plot_files(f1, gps, df1 = '', time = '', gpx = '', f_point=0, l_point=-1):

    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)

    t_0 = t[f_point]
    t_1 = t[l_point]
    print('Times : ', t_0, ', ', t_1)
    i, = np.where(times >= t_0)
    print('Check : ', i[0], ', ', times[i[0]])
    j, = np.where(times <= t_1)
    print('Check : ', j[-1], ', ', times[j[-1]])
    i = i[0]
    j = j[-1]

    lat = lat[i: j+1]
    lon = lon[i: j+1]
    t = t[f_point:l_point]
    acc1 = acc1[f_point:l_point]




    # plt.figure()
    # plt.plot([i for i in range(len(t))], t)

    acc_z = (acc1[:,2])
    print("Moyenne : ", np.mean(acc_z))
    print("Ecrat type : ", np.std(acc_z))
    acc_z = np.abs(acc_z - np.mean(acc_z))/np.std(acc_z)

    filtered_z = np.abs(freq_filter(acc_z))
    filtered_z = (filtered_z - np.mean(filtered_z)) * 2.3




    detect_times = []
    if df1 != '':
        detect_times, _ = log_file_detect(df1)

    detect_times = [detect_times[i] for i in range(len(detect_times)) if (t_0 < detect_times[i] < t_1) ]
    lat_x, lon_x = [], []

    if gpx != '':
        lat_x, lon_x = get_latlon_as_in_nmea_from_gpx(gpx)

    # dilat = 1.
    # time_off = 0.

    # For the last Guerledan measures
    time_off = 180 + alignement
    dilat = 1.35

    real_time_detect = time.split(',')
    lt_time = []
    # print("Times : ", real_time_detect)
    if real_time_detect != ['']:
        for i in real_time_detect:
            inter1 = i.split(':')
            value = float(inter1[0])*3600 + (float(inter1[1]))*60 + float(inter1[2]) + time_off
            if lt_time!= []:
                test = np.abs(lt_time[-1] - value)
                value = dilat*test + lt_time[-1]

                time_off += (dilat - 1)*test
            lt_time.append(value)

    seuil = 1.5
    delta_p = 5.

    filtered_detect = detect_points(filtered_z, seuil, delta_p)
    # print(filtered_detect)


    graph_offset = 6
    a = len(t)/len(lat)
    b = len(lat_x)/len(lat)
    print('Coef for correl : ', a, ', ', b)



    f1 = plt.figure(figsize=(15,10))
    f1.subplots_adjust(hspace=0.5)
    f2 = plt.figure()

    ax1 = f1.add_subplot(311)
    ax2 = f1.add_subplot(312)
    ax4 = f1.add_subplot(313)
    ax3 = f2.add_subplot(111)

    ax1.set_title('GPS Altitude')
    ax2.set_title('Accelerometers, x, y, z')
    ax2.legend(['X', 'Y', 'Z'])
    ax3.set_title('GPS track')
    ax3.legend(['Buoy', 'Boat'])
    ax4.set_title('Z accel centered')
    ax4.set_ylim([0, 10])

    for i in range(len(lat)):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

        new_i_x = int(i*b)
        # print("Check new_i_x : ", new_i_x)

        ax3.plot(lat, lon, color='blue')
        # ax3.plot(lat_x, lon_x, color='green')
        if len(lat_x)!= 0:
            ax3.scatter(lat_x[new_i_x], lon_x[new_i_x], color='purple')
        ax3.scatter(lat[i], lon[i], color='red')

        ax1.plot(times, altitudes)
        ax1.scatter(times[i], altitudes[i], color='red')

        ax2.plot(t, acc1)
        new_i = int(np.round(i*a))
        ax2.scatter(t[new_i], acc1[:,0][new_i], color='red')
        ax2.scatter(t[new_i], acc1[:,1][new_i], color='red')
        ax2.scatter(t[new_i], acc1[:,2][new_i], color='red')

        ax4.plot(t, acc_z)
        ax4.plot(t, filtered_z)
        ax4.scatter(t[new_i], acc_z[new_i], color='red')
        ax4.scatter(detect_times, [graph_offset + 6 for i in range(len(detect_times))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect)):
            # print(filtered_detect[j][0])
            # print(filtered_z[filtered_detect[j][0]])
            ax4.scatter(t[filtered_detect[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude')
        ax2.set_title('Accelerometers, x, y, z')
        ax2.legend(['X', 'Y', 'Z'])
        ax3.set_title('GPS track')
        if len(lat_x) != 0 :
            ax3.legend(['Buoy', 'Boat at time t'])
        else:
            ax3.legend(['Buoy', 'Pos at time t'])

        ax4.set_title('Z accel centered')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered'])
        ax4.set_ylim([-1, graph_offset + 7])



        plt.pause(0.001)

        

