import numpy as np


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