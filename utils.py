import numpy as np
import matplotlib.pyplot as plt



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


def plot_files(f1, gps, df1 = '', time = ''):

    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
    acc_z = (acc[:,2])
    print("Moyenne : ", np.mean(acc_z))
    print("Ecrat type : ", np.std(acc_z))
    acc_z = np.abs(acc_z - np.mean(acc_z))
    detect_times = []
    if df1 != '':
        detect_times, _ = log_file_detect(df1)


    dilat = 1.
    time_off = 0.
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

    graph_offset = 15
    a = len(t)/len(lat)

    f1 = plt.figure()
    f2 = plt.figure()

    ax1 = f1.add_subplot(311)
    ax1.plot(times, altitudes)
    ax2 = f1.add_subplot(312)
    ax2.plot(t,acc1)
    ax2.scatter(detect_times, [graph_offset for i in range(len(detect_times))], color='blue')
    ax2.scatter(lt_time, [graph_offset for i in range(len(lt_time))], color='green')
    ax4 = f1.add_subplot(313)
    ax4.plot(t, acc_z)
    ax3 = f2.add_subplot(111)

    ax1.set_title('GPS Altitude')
    ax2.set_title('Accelerometers, x, y, z')
    ax2.legend(['X', 'Y', 'Z'])
    ax3.set_title('Buoy GPS track')
    ax4.set_title('Z accel centered')

    for i in range(len(lat)):
        ax1.cla()
        ax2.cla()
        ax3.cla()

        ax3.plot(lat, lon, color='blue')
        ax3.scatter(lat[i], lon[i], color='red')

        ax1.plot(times, altitudes)
        ax1.scatter(times[i], altitudes[i], color='red')

        ax2.plot(t, acc1)
        ax2.scatter(detect_times, [graph_offset for i in range(len(detect_times))], color='blue')
        ax2.scatter(lt_time, [graph_offset for i in range(len(lt_time))], color='green')
        new_i = int(np.round(i*a))
        ax2.scatter(t[new_i], acc1[:,0][new_i], color='red')
        ax2.scatter(t[new_i], acc1[:,1][new_i], color='red')
        ax2.scatter(t[new_i], acc1[:,2][new_i], color='red')

        ax4.plot(t, acc_z)


        ax1.set_title('GPS Altitude')
        ax2.set_title('Accelerometers, x, y, z')
        ax2.legend(['X', 'Y', 'Z'])
        ax3.set_title('Buoy GPS track')
        ax4.set_title('Z accel centered')


        plt.pause(0.001)