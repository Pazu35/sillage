import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import datetime
# For gpx files
import gpxpy
import gpxpy.gpx

#  https://matplotlib.org/stable/tutorials/introductory/lifecycle.html
# Parsing an existing gpx file:
def convert(d):
    d = abs(d)
    D = int(d)
    M = (d - D) * 60
    # print("Convert : ", d, D, M)
    return float("{0:2d}{1:7.4f}".format(D, M).replace(" ", "0"))

def get_latlon_as_in_nmea_from_gpx(filename):
    gpx_file = open(filename, 'r')

    gpx = gpxpy.parse(gpx_file)
    lat = []
    lon = []
    times = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                # print(point.latitude)
                # print(point.longitude)
                lat.append(convert(point.latitude))
                lon.append(convert(point.longitude))
                date = point.time.strftime("%H:%M:%S.%s")
                # print(date)
                t_i, _ = cvt_time(date, date)
                # print(t_i)
                times.append(t_i + 3600)
                # print(type(point.time))
    return np.array(lat), np.array(lon), np.array(times)

def get_time_from_point(index):
    hours = index//3600
    minutes = (index - 3600*hours)//60
    seconds = index - 3600*hours - 60*minutes
    return str(hours)+':'+str(minutes)+':'+str(seconds)

def cvt_time(t1, t2, alignement=0.):


    intermediate = t1.split(':')
    t_1 = (float(intermediate[0]) - 1)*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", ".")) + alignement


    intermediate = t2.split(':')
    t_2 = (float(intermediate[0])- 1)*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", ".")) + alignement

    return t_1, t_2

def find_index(L, t1):
    i, = np.where(L  >= t1)
    if len(i) != 0:
        i = i[0]
    else :
        i = -1
    return i

def write_in_csv(file_name, detect_times, gps_pos):
    f = open(file_name, "w")

    # print("Length of both lists : ", len(detect_times), 'and ', len(gps_pos))
    L = len(detect_times)
    for i in range(L):
        string = str(detect_times[i][0]) + "," + str(detect_times[i][1]) + "," + str(gps_pos[i][0]) + "," + str(gps_pos[i][1])+ "," + str(gps_pos[i][2]) + "," + str(gps_pos[i][3]) +"\n"
        f.write(string)
    f.close()

def gps_to_csv(file_name, lat_x, lon_x, t_x):
    f = open(file_name, "w")

    # print("Length of both lists : ", len(detect_times), 'and ', len(gps_pos))
    L = len(lat_x)
    for i in range(L):
        string = str(t_x[i]) + "," + str(t_x[i])+ "," + str(lat_x[i])+ "," + str(lat_x[i]) + "," + str(lon_x[i]) + "," + str(lon_x[i]) + "\n"
        f.write(string)
    f.close()

# ====================

# Frequency filter :
def butter_passe_bande(fc_inf,fc_sup, fs, order):
    nyq = 0.5 * fs
    b, a = sg.butter(order, (fc_inf/nyq, fc_sup/nyq), btype='bandpass', analog=False)
    return b, a


def freq_filter(l):
    order = 2
    fs = 50
    fc_inf, fc_sup = 0.2 , 0.6
    gain = 4.

    #Calcul de la réponse et du filtre
    b, a = butter_passe_bande(fc_inf, fc_sup, fs, order)
    y = sg.lfilter(b, a , l)
    y = np.abs(y - np.mean(y)) * gain
    return y

# ====================

# Detection function with a set of data, a threshold and a delta_p for grouping
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

# ====================
# 3D Rotation functions if needed

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

# Calculate Euler angles from rotation matrix 
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


# ====================
# Read log file functions

# Read gps log
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

# Read AHRS log 
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
        # Whole raw datas :
        # rtcDate,rtcTime,aX,aY,aZ,gX,gY,gZ,mX,mY,mZ,imu_degC,output_Hz

        l = line.strip()
        l = l.split(';')
        date = l[0].split(' ')
        # print('l : ', l)
        l = l[2].split(',')
        # Skip the first 30 useless lines 
        if count < 30:
            count +=1
        else :
            try :
                intermediate = date[1].split(':')
                t_i = float(intermediate[0])*3600 + float(intermediate[1])*60 + float(intermediate[2].replace(",", ".")) + alignement
                acc_i = [float(l[0]), float(l[1]), float(l[2])]
                gyr_i = [float(l[3]), float(l[4]), float(l[5])]
                mag_i = [float(l[6]), float(l[7]), float(l[8])]
                temp_i = float(l[9])
                rate_i = float(l[10])

                time.append(t_i)
                acc.append(acc_i)
                gyr.append(gyr_i)
                mag.append(mag_i)
                temp.append(temp_i)
                rate.append(rate_i)

            except:

                print("Log error in file " + file_name)
                print('Line : ', line)

    return np.array(time), np.array(acc)*0.01, np.array(gyr)*np.pi/180, np.array(mag), np.array(temp), np.array(rate)

# Read detetion log
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

# Get the appropriate time vectors from a string of times
def real_time_read(time, dilat=1., time_off = 0.):
    real_time_detect = time.split(',')
    lt_time = []
    # print("Times : ", real_time_detect)
    if real_time_detect != ['']:
        for i in real_time_detect:
            inter1 = i.split(':')
            value = (float(inter1[0]) - 1)*3600 + (float(inter1[1]))*60 + float(inter1[2]) + time_off
            if lt_time!= []:
                test = np.abs(lt_time[-1] - value)
                value = dilat*test + lt_time[-1]

                time_off += (dilat - 1)*test
            lt_time.append(value)
    return lt_time

# ====================
# Main post process to plot the log files and detections


def plot_files(f1, gps, seuil, delta_p, df1 = '', time = '', gpx = '', f_point=0, l_point=-1, animate=False):
    
    # Time dilatation prameters if delay in acquisition :
    time_off = 180
    dilat = 1.35

    # Detection parameters for waves detection
    # dilat = 1.
    # time_off = 0.

    # Graph parameter fo a better view of the data
    graph_offset = 6
    step = 10


    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)

    t_0 = t[f_point]
    t_1 = t[l_point]
    print('Times : ', t_0, ', ', t_1)
    i, = np.where(times >= t_0)
    print('Check i : ', i[0], ', ', times[i[0]])
    j, = np.where(times <= t_1)
    print('Check j : ',j[0], ', ', j[-1], ', ', times[j[-1]])
    i = i[0]
    j = j[-1]

    lat = lat[i: j+1]
    lon = lon[i: j+1]
    altitudes = altitudes[i: j+1]
    times = times[i: j+1]
    t = t[f_point:l_point]
    acc1 = acc1[f_point:l_point]


    # plt.figure()
    # plt.plot([i for i in range(len(t))], t)

    acc_z = (acc1[:,2])
    # print("Moyenne : ", np.mean(acc_z))
    # print("Ecrat type : ", np.std(acc_z))
    acc_z = np.abs(acc_z - np.mean(acc_z))/np.std(acc_z)

    filtered_z = freq_filter(acc_z)

    detect_times = []
    if df1 != '':
        detect_times, _ = log_file_detect(df1)

    detect_times = [detect_times[i] for i in range(len(detect_times)) if (t_0 < detect_times[i] < t_1) ]
    lat_x, lon_x = [], []

    if gpx != '':
        lat_x, lon_x = get_latlon_as_in_nmea_from_gpx(gpx)

    time_off += alignement
    lt_time = real_time_read(time, dilat, time_off)

    filtered_detect = detect_points(filtered_z, seuil, delta_p)
    # print(filtered_detect)


    a = len(t)/len(lat)
    b = len(lat_x)/len(lat)
    # print('Coef for correl : ', a, ', ', b)



    f1 = plt.figure(figsize=(20,10))
    f1.subplots_adjust(hspace=0.5)
    f2 = plt.figure()

    ax1 = f1.add_subplot(311)
    ax2 = f1.add_subplot(312)
    ax4 = f1.add_subplot(313)
    ax3 = f2.add_subplot(111)

    ax1.set_title('GPS Altitude')
    ax2.set_title('Accelerometers, x, y, z')
    ax3.set_title('GPS track')
    ax4.set_title('Z accel centered')

    if animate:
        for i in range(0, len(lat), step):
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
            ax1.set_xlabel("Time (s)")
            ax2.set_title('Accelerometers, x, y, z')
            ax2.legend(['X', 'Y', 'Z'])
            ax2.set_ylim([-1, graph_offset + 7])
            ax2.set_xlabel("Time (s)")
            ax3.set_title('GPS track')
            if len(lat_x) != 0 :
                ax3.legend(['Buoy', 'Boat at time t'])
            else:
                ax3.legend(['Buoy', 'Pos at time t'])

            ax4.set_title('Z accel centered')
            ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered'])
            ax4.set_ylim([-1, graph_offset + 7])
            ax4.set_xlabel("Time (s)")



            plt.pause(0.001)
    else:

        ax3.plot(lat, lon, color='blue')
        # ax3.plot(lat_x, lon_x, color='green')

        ax1.plot(times, altitudes)

        ax2.plot(t, acc1)

        ax4.plot(t, acc_z)
        ax4.plot(t, filtered_z)
        ax4.scatter(detect_times, [graph_offset + 6 for i in range(len(detect_times))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect)):
            # print(filtered_detect[j][0])
            # print(filtered_z[filtered_detect[j][0]])
            ax4.scatter(t[filtered_detect[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude')
        ax1.set_xlabel("Time (s)")
        ax2.set_title('Accelerometers, x, y, z')
        ax2.legend(['X', 'Y', 'Z'])
        ax2.set_ylim([-1, graph_offset + 7])
        ax2.set_xlabel("Time (s)")
        ax3.set_title('GPS track')
        if len(lat_x) != 0 :
            ax3.legend(['Buoy', 'Boat at time t'])
        else:
            ax3.legend(['Buoy', 'Pos at time t'])

        ax4.set_title('Z accel centered')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered'])
        ax4.set_ylim([-1, graph_offset + 7])
        ax4.set_xlabel("Time (s)")


def plot_files2(f1, f2, gps, seuil, delta_p, df1='', df2='', time = '', gpx = '', f_point=0, l_point=-1, animate=False):
    
    # Time dilatation prameters if delay in acquisition :
    time_off = 180
    dilat = 1.35

    # Detection parameters for waves detection
    # dilat = 1.
    # time_off = 0.

    # Graph parameter fo a better view of the data
    graph_offset = 6
    step = 10



    # Read gps log
    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    # Read ahrs logs
    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
    t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)

    # Slicing according to the f_point and l_point, for all arrays
    t_02 = t2[f_point]
    t_12 = t2[l_point]
    print('Times : ', t_02, ', ', t_12)
    t_0 = t[f_point]
    t_1 = t[l_point]
    print('Times : ', t_0, ', ', t_1)
    # delta_t_ahrs = np.mean([t_0 - t_02, t_1 - t_12])
    delta_t_ahrs = t_0 - t_02
    print('Delta t AHRS : ', delta_t_ahrs)
    i, = np.where(times  >= t_0 - alignement)
    print('Check gps slicing first : ', i[0], ', ', times[i[0]])
    j, = np.where(times  <= t_1 - alignement)
    print('Check gps slicing last : ',j[0], ', ', j[-1], ', ', times[j[-1]])
    i = i[0]
    j = j[-1]

    lat = lat[i: j+1]
    lon = lon[i: j+1]
    altitudes = altitudes[i: j+1]
    times = times[i: j+1]
    t = t[f_point:l_point]
    acc1 = acc1[f_point:l_point]
    t2 = t2[f_point:l_point] + delta_t_ahrs
    acc2 = acc2[f_point:l_point]

    # Getting Z-axis accelerations and normalisation
    acc_z1 = (acc1[:,2])
    acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

    acc_z2 = (acc2[:,2])
    acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)

    # Harmonisation between the two files
    # std1 = np.std(acc_z1)
    # std2 = np.std(acc_z2)
    # coef = std1/std2
    # acc_z2 *= (1+coef)

    filtered_z1 = freq_filter(acc_z1)
    filtered_z2 = freq_filter(acc_z2)

    # Getting detection times
    detect_times1 = []
    if df1 != '':
        detect_times1, _ = log_file_detect(df1)
        detect_times1 = [detect_times1[i] for i in range(len(detect_times1)) if (t_0 < detect_times1[i] < t_1) ]
    else:
        detect_times1 = detect_points(acc_z1, seuil, delta_p)
        res = []
        for j in range(len(detect_times1)):
            res.append(t2[detect_times1[j][0]])
        detect_times1 = res
    
    detect_times2 = []
    if df2 != '':
        detect_times2, _ = log_file_detect(df2)
        detect_times2 = [detect_times2[i] for i in range(len(detect_times2)) if (t_0 < detect_times2[i] < t_1) ]
    else:
        detect_times2 = detect_points(acc_z2, seuil, delta_p)
        res = []
        for j in range(len(detect_times2)):
            res.append(t2[detect_times2[j][0]])
        detect_times2 = res
        # print('D : ', (detect_times2))
    
    # Getting gps track of gpx file
    lat_x, lon_x = [], []
    if gpx != '':
        lat_x, lon_x = get_latlon_as_in_nmea_from_gpx(gpx)

    # Getting the timestamps
    time_off += alignement
    lt_time = real_time_read(time, dilat, time_off)

    # Getting frequency filtered detections
    filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
    filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

    # Coefs for graph plotting the different sizes arrays
    a = len(t)/len(lat)
    b = len(lat_x)/len(lat)


    # Plt stuff for plotting 
    f1 = plt.figure(figsize=(20,10))
    f1.subplots_adjust(hspace=0.5)
    f2 = plt.figure()

    ax1 = f1.add_subplot(311)
    ax2 = f1.add_subplot(323)
    ax4 = f1.add_subplot(325)
    ax6 = f1.add_subplot(324)
    ax7 = f1.add_subplot(326)

    ax3 = f2.add_subplot(111)

    ax1.set_title('GPS Altitude')
    ax2.set_title('Accelerometers, x, y, z AHRS1')
    ax4.set_title('Z accel centered AHRS1')
    ax6.set_title('Accelerometers, x, y, z AHRS2')
    ax7.set_title('Z accel centered AHRS2')

    ax3.set_title('GPS track')

    if animate:
        for i in range(0, len(lat), step):
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            ax6.cla()
            ax7.cla()

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

            ax6.plot(t2, acc2)
            new_i = int(np.round(i*a))
            ax6.scatter(t2[new_i], acc2[:,0][new_i], color='red')
            ax6.scatter(t2[new_i], acc2[:,1][new_i], color='red')
            ax6.scatter(t2[new_i], acc2[:,2][new_i], color='red')


            ax4.plot(t, acc_z1)
            ax4.plot(t, filtered_z1)
            ax4.plot([t_0, t_1], [seuil, seuil], color='red')
            ax4.scatter(t[new_i], acc_z1[new_i], color='red')
            ax4.scatter(detect_times1, [graph_offset + 6 for i in range(len(detect_times1))], color='blue')
            ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
            for j in range(len(filtered_detect1)):
                # print(filtered_detect1[j][0])
                # print(filtered_z1[filtered_detect1[j][0]])
                ax4.scatter(t[filtered_detect1[j][0]], graph_offset+2, color='purple')

            ax7.plot(t2, acc_z2)
            ax7.plot(t2, filtered_z2)
            ax7.plot([t_0, t_1], [seuil, seuil], color='red')
            ax7.scatter(t2[new_i], acc_z2[new_i], color='red')
            ax7.scatter(detect_times2, [graph_offset + 6 for i in range(len(detect_times2))], color='blue')
            ax7.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
            for j in range(len(filtered_detect2)):
                # print(filtered_detect2[j][0])
                # print(filtered_z2[filtered_detect2[j][0]])
                ax7.scatter(t[filtered_detect2[j][0]], graph_offset+2, color='purple')



            ax1.set_title('GPS Altitude')
            ax1.set_xlabel("Time (s)")
            ax2.set_title('Accelerometers, x, y, z AHRS1')
            ax2.legend(['X', 'Y', 'Z'])
            ax2.set_ylim([-1, graph_offset + 7])
            ax2.set_xlabel("Time (s)")
            ax6.set_title('Accelerometers, x, y, z AHRS2')
            ax6.legend(['X', 'Y', 'Z'])
            ax6.set_ylim([-1, graph_offset + 7])
            ax6.set_xlabel("Time (s)")
            ax3.set_title('GPS track')
            if len(lat_x) != 0 :
                ax3.legend(['Buoy', 'Boat at time t'])
            else:
                ax3.legend(['Buoy', 'Pos at time t'])

            ax4.set_title('Z accel centered AHRS1')
            ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
            ax4.set_ylim([-1, graph_offset + 7])
            ax4.set_xlabel("Time (s)")
            ax7.set_title('Z accel centered AHRS2')
            ax7.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
            ax7.set_ylim([-1, graph_offset + 7])
            ax7.set_xlabel("Time (s)")



            plt.pause(0.001)
    else:

        ax3.plot(lat, lon, color='blue')
        # ax3.plot(lat_x, lon_x, color='green')

        ax1.plot(times, altitudes)

        ax2.plot(t, acc1)
        ax6.plot(t2, acc2)

        ax4.plot(t, acc_z1)
        ax4.plot(t, filtered_z1)
        ax4.plot([t_0, t_1], [seuil, seuil], color='red')
        ax4.scatter(detect_times1, [graph_offset + 6 for i in range(len(detect_times1))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect1)):
            # print(filtered_detect1[j][0])
            # print(filtered_z1[filtered_detect1[j][0]])
            ax4.scatter(t[filtered_detect1[j][0]], graph_offset+2, color='purple')


        ax7.plot(t2, acc_z2)
        ax7.plot(t2, filtered_z2)
        ax7.plot([t_0, t_1], [seuil, seuil], color='red')
        ax7.scatter(detect_times2, [graph_offset + 6 for i in range(len(detect_times2))], color='blue')
        ax7.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect2)):
            # print(filtered_detect2[j][0])
            # print(filtered_z2[filtered_detect2[j][0]])
            ax7.scatter(t[filtered_detect2[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude')
        ax1.set_xlabel("Time (s)")
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax2.legend(['X', 'Y', 'Z'])
        ax2.set_ylim([-1, graph_offset + 7])
        ax2.set_xlabel("Time (s)")
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax6.legend(['X', 'Y', 'Z'])
        ax6.set_ylim([-1, graph_offset + 7])
        ax6.set_xlabel("Time (s)")
        ax3.set_title('GPS track')
        if len(lat_x) != 0 :
            ax3.legend(['Buoy', 'Boat at time t'])
        else:
            ax3.legend(['Buoy', 'Pos at time t'])

        ax4.set_title('Z accel centered AHRS1')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax4.set_ylim([-1, graph_offset + 7])
        ax4.set_xlabel("Time (s)")
        ax7.set_title('Z accel centered AHRS2')
        ax7.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax7.set_ylim([-1, graph_offset + 7])
        ax7.set_xlabel("Time (s)")


def plot_buoys( seuil, delta_p, file1 = [], file2=[], gps_files = [], df1='', df2='', time = '', gpx = '', f_point=0, l_point=-1):
    # Time dilatation prameters if delay in acquisition :
    time_off = 0.
    dilat = 1.

    # Detection parameters for waves detection
    # dilat = 1.
    # time_off = 0.

    # Graph parameter fo a better view of the data
    graph_offset = 6
    step = 10


    if len(file1) != 0 and len(file2) != 0 and len(gps_files)!= 0:
        print("\n====================")
        print("Buoy 1")
        print("====================\n")
        f1, f2 = file1[0], file1[1]
        gps = gps_files[0]    
        # Read gps log
        times, gps_times, lat, lon , altitudes = log_gps(gps)
        alignement = np.mean(gps_times - times)
        print('GPS time offset : ', alignement)

        # Read ahrs logs
        t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
        t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)

        # Slicing according to the f_point and l_point, for all arrays
        t_02 = t2[f_point]
        t_12 = t2[l_point]
        print('Times : ', t_02, ', ', t_12)
        t_0 = t[f_point]
        t_1 = t[l_point]
        print('Times : ', t_0, ', ', t_1)
        # delta_t_ahrs = np.mean([t_0 - t_02, t_1 - t_12])
        delta_t_ahrs = t_0 - t_02
        print('Delta t AHRS : ', delta_t_ahrs)
        i, = np.where(times  >= t_0 - alignement)
        print('Check gps slicing first : ', i[0], ', ', times[i[0]])
        j, = np.where(times  <= t_1 - alignement)
        print('Check gps slicing last : ',j[0], ', ', j[-1], ', ', times[j[-1]])
        i = i[0]
        j = j[-1]

        lat = lat[i: j+1]
        lon = lon[i: j+1]
        altitudes = altitudes[i: j+1]
        times = times[i: j+1]
        t = t[f_point:l_point]
        acc1 = acc1[f_point:l_point]
        t2 = t2[f_point:l_point] + delta_t_ahrs
        acc2 = acc2[f_point:l_point]

        # Getting Z-axis accelerations and normalisation
        acc_z1 = (acc1[:,2])
        acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

        acc_z2 = (acc2[:,2])
        acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)

        # Harmonisation between the two files
        # std1 = np.std(acc_z1)
        # std2 = np.std(acc_z2)
        # coef = std1/std2
        # acc_z2 *= (1+coef)

        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)

        # Getting detection times
        detect_times1 = []
        if df1 != '':
            detect_times1, _ = log_file_detect(df1)
            detect_times1 = [detect_times1[i] for i in range(len(detect_times1)) if (t_0 < detect_times1[i] < t_1) ]
        else:
            detect_times1 = detect_points(acc_z1, seuil, delta_p)
            res = []
            for j in range(len(detect_times1)):
                res.append(t2[detect_times1[j][0]])
            detect_times1 = res
        
        detect_times2 = []
        if df2 != '':
            detect_times2, _ = log_file_detect(df2)
            detect_times2 = [detect_times2[i] for i in range(len(detect_times2)) if (t_0 < detect_times2[i] < t_1) ]
        else:
            detect_times2 = detect_points(acc_z2, seuil, delta_p)
            res = []
            for j in range(len(detect_times2)):
                res.append(t2[detect_times2[j][0]])
            detect_times2 = res
            # print('D : ', (detect_times2))
        
        # Getting gps track of gpx file
        lat_x, lon_x = [], []
        if gpx != '':
            lat_x, lon_x = get_latlon_as_in_nmea_from_gpx(gpx)

        # Getting the timestamps
        time_off += alignement
        lt_time = real_time_read(time, dilat, time_off)

        # Getting frequency filtered detections
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        # Coefs for graph plotting the different sizes arrays
        a = len(t)/len(lat)
        b = len(lat_x)/len(lat)


        # Plt stuff for plotting 
        f1 = plt.figure(figsize=(20,10))
        f1.subplots_adjust(hspace=0.5)
        f2 = plt.figure()

        ax1 = f1.add_subplot(311)
        ax2 = f1.add_subplot(323)
        ax4 = f1.add_subplot(325)
        ax6 = f1.add_subplot(324)
        ax7 = f1.add_subplot(326)

        ax3 = f2.add_subplot(111)

        ax1.set_title('GPS Altitude buoy 1')
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax4.set_title('Z accel centered AHRS1')
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax7.set_title('Z accel centered AHRS2')

        ax3.set_title('GPS track')

        ax3.plot(lat, lon, color='blue')
        # ax3.plot(lat_x, lon_x, color='green')

        ax1.plot(times, altitudes)

        ax2.plot(t, acc1)
        ax6.plot(t2, acc2)

        ax4.plot(t, acc_z1)
        ax4.plot(t, filtered_z1)
        ax4.plot([t_0, t_1], [seuil, seuil], color='red')
        ax4.scatter(detect_times1, [graph_offset + 6 for i in range(len(detect_times1))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect1)):
            # print(filtered_detect1[j][0])
            # print(filtered_z1[filtered_detect1[j][0]])
            ax4.scatter(t[filtered_detect1[j][0]], graph_offset+2, color='purple')


        ax7.plot(t2, acc_z2)
        ax7.plot(t2, filtered_z2)
        ax7.plot([t_0, t_1], [seuil, seuil], color='red')
        ax7.scatter(detect_times2, [graph_offset + 6 for i in range(len(detect_times2))], color='blue')
        ax7.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect2)):
            # print(filtered_detect2[j][0])
            # print(filtered_z2[filtered_detect2[j][0]])
            ax7.scatter(t[filtered_detect2[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude buoy 1')
        ax1.set_xlabel("Time (s)")
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax2.legend(['X', 'Y', 'Z'])
        ax2.set_ylim([-1, graph_offset + 7])
        ax2.set_xlabel("Time (s)")
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax6.legend(['X', 'Y', 'Z'])
        ax6.set_ylim([-1, graph_offset + 7])
        ax6.set_xlabel("Time (s)")
        ax3.set_title('GPS track')
        if len(lat_x) != 0 :
            ax3.legend(['Buoy', 'Boat at time t'])
        else:
            ax3.legend(['Buoy', 'Pos at time t'])

        ax4.set_title('Z accel centered AHRS1')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax4.set_ylim([-1, graph_offset + 7])
        ax4.set_xlabel("Time (s)")
        ax7.set_title('Z accel centered AHRS2')
        ax7.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax7.set_ylim([-1, graph_offset + 7])
        ax7.set_xlabel("Time (s)")


        print("\n====================")
        print("Buoy 2")
        print("====================\n")


        f1, f2 = file2[0], file2[1]
        gps = gps_files[1]    
        # Read gps log
        times, gps_times, lat, lon , altitudes = log_gps(gps)
        alignement = np.mean(gps_times - times)
        print('GPS time offset : ', alignement)

        # Read ahrs logs
        t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
        t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)

        if f_point > len(t) or f_point > len(t2):
            f_point = -1

        if l_point > len(t) or l_point > len(t2):
            l_point = -1


        # Slicing according to the f_point and l_point, for all arrays
        t_02 = t2[f_point]
        t_12 = t2[l_point]
        print('Times : ', t_02, ', ', t_12)
        t_0 = t[f_point]
        t_1 = t[l_point]
        print('Times : ', t_0, ', ', t_1)
        # delta_t_ahrs = np.mean([t_0 - t_02, t_1 - t_12])
        delta_t_ahrs = t_0 - t_02
        print('Delta t AHRS : ', delta_t_ahrs)
        i, = np.where(times  >= t_0 - alignement)
        j, = np.where(times  <= t_1 - alignement)
        # print(i)
        # print(j)
        if len(i) == 0:
            i = [-1]
            print("Warning ! log of buoy 2 stopped first ! ")

        # print(i)
        # print(j)
        print('Check gps slicing first : ', i[0], ', ', times[i[0]])
        print('Check gps slicing last : ',j[0], ', ', j[-1], ', ', times[j[-1]])
        i = i[0]
        j = j[-1]

        lat = lat[i: j+1]
        lon = lon[i: j+1]
        altitudes = altitudes[i: j+1]
        times = times[i: j+1]
        t = t[f_point:l_point]
        acc1 = acc1[f_point:l_point]
        t2 = t2[f_point:l_point] + delta_t_ahrs
        acc2 = acc2[f_point:l_point]

        # Getting Z-axis accelerations and normalisation
        acc_z1 = (acc1[:,2])
        acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

        acc_z2 = (acc2[:,2])
        acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)

        # Harmonisation between the two files
        # std1 = np.std(acc_z1)
        # std2 = np.std(acc_z2)
        # coef = std1/std2
        # acc_z2 *= (1+coef)

        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)

        # Getting detection times
        detect_times1 = []
        if df1 != '':
            detect_times1, _ = log_file_detect(df1)
            detect_times1 = [detect_times1[i] for i in range(len(detect_times1)) if (t_0 < detect_times1[i] < t_1) ]
        else:
            detect_times1 = detect_points(acc_z1, seuil, delta_p)
            res = []
            for j in range(len(detect_times1)):
                res.append(t[detect_times1[j][0]])
            detect_times1 = res
        
        detect_times2 = []
        if df2 != '':
            detect_times2, _ = log_file_detect(df2)
            detect_times2 = [detect_times2[i] for i in range(len(detect_times2)) if (t_0 < detect_times2[i] < t_1) ]
        else:
            detect_times2 = detect_points(acc_z2, seuil, delta_p)
            res = []
            for j in range(len(detect_times2)):
                res.append(t2[detect_times2[j][0]])
            detect_times2 = res
            # print('D : ', (detect_times2))
        
        # Getting gps track of gpx file
        lat_x, lon_x = [], []
        if gpx != '':
            lat_x, lon_x = get_latlon_as_in_nmea_from_gpx(gpx)

        # Getting the timestamps
        time_off += alignement
        lt_time = real_time_read(time, dilat, time_off)

        # Getting frequency filtered detections
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        # Coefs for graph plotting the different sizes arrays
        a = len(t)/len(lat)
        b = len(lat_x)/len(lat)


        # Plt stuff for plotting 
        f3 = plt.figure(figsize=(20,10))
        f3.subplots_adjust(hspace=0.5)

        ax1 = f3.add_subplot(311)
        ax2 = f3.add_subplot(323)
        ax4 = f3.add_subplot(325)
        ax6 = f3.add_subplot(324)
        ax7 = f3.add_subplot(326)


        ax1.set_title('GPS Altitude buoy 2')
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax4.set_title('Z accel centered AHRS1')
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax7.set_title('Z accel centered AHRS2')

        ax3.plot(lat, lon, color='purple')
        # ax3.plot(lat_x, lon_x, color='green')

        ax1.plot(times, altitudes)

        ax2.plot(t, acc1)
        ax6.plot(t2, acc2)

        ax4.plot(t, acc_z1)
        ax4.plot(t, filtered_z1)
        ax4.plot([t_0, t_1], [seuil, seuil], color='red')
        ax4.scatter(detect_times1, [graph_offset + 6 for i in range(len(detect_times1))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect1)):
            # print(filtered_detect1[j][0])
            # print(filtered_z1[filtered_detect1[j][0]])
            ax4.scatter(t[filtered_detect1[j][0]], graph_offset+2, color='purple')


        ax7.plot(t2, acc_z2)
        ax7.plot(t2, filtered_z2)
        ax7.plot([t_0, t_1], [seuil, seuil], color='red')
        ax7.scatter(detect_times2, [graph_offset + 6 for i in range(len(detect_times2))], color='blue')
        ax7.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect2)):
            # print(filtered_detect2[j][0])
            # print(filtered_z2[filtered_detect2[j][0]])
            ax7.scatter(t[filtered_detect2[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude buoy 2')
        ax1.set_xlabel("Time (s)")
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax2.legend(['X', 'Y', 'Z'])
        ax2.set_ylim([-1, graph_offset + 7])
        ax2.set_xlabel("Time (s)")
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax6.legend(['X', 'Y', 'Z'])
        ax6.set_ylim([-1, graph_offset + 7])
        ax6.set_xlabel("Time (s)")
        ax3.set_title('GPS track')
        ax3.legend(['Buoy 1', 'Buoy 2'])

        ax4.set_title('Z accel centered AHRS1')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax4.set_ylim([-1, graph_offset + 7])
        ax4.set_xlabel("Time (s)")
        ax7.set_title('Z accel centered AHRS2')
        ax7.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax7.set_ylim([-1, graph_offset + 7])
        ax7.set_xlabel("Time (s)")

    else:
        print("Wrong data type ! Please put in lists !")

def plot_detections(t=[], intervals=[], t0 = '', t1=''):

    assert(len(t) == len(intervals))

    if t0 == '':
        t0 = t[0]
    if t1 == '':
        t1 = t[1]
    f1 = plt.figure(figsize=(20,10))
    f1.subplots_adjust(hspace=0.5)

    l_t = len(t)
    for i in range(l_t):
        subplot = 100*l_t + 10 + i + 1
        # print(subplot)
        ax = f1.add_subplot(subplot)
        ax.scatter(t[i], [0 for i in range(len(t[i]))], color='red')
        for j in intervals[i]:
            ax.plot([j[0], j[1]], [0,0], color='blue')
        ax.set_title("Detection times and intervals of " + str(i+1))
        ax.set_xlim([t0, t1])
        ax.set_xlabel("Time (s)")



def plot_buoys2( seuil, delta_p, file1 = [], file2=[], gps_files = [], df1='', df2='', time = '', gpx = '', t_begin=0, t_last=-1):
    # Time dilatation prameters if delay in acquisition :
    time_off = 0.
    dilat = 1.

    # Detection parameters for waves detection
    # dilat = 1.
    # time_off = 0.

    # Graph parameter fo a better view of the data
    graph_offset = 6
    step = 10


    if len(file1) != 0 and len(file2) != 0 and len(gps_files)!= 0:
        print("\n====================")
        print("Buoy 1")
        print("====================\n")
        f1, f2 = file1[0], file1[1]
        gps = gps_files[0]    
        # Read gps log
        times, gps_times, lat, lon , altitudes = log_gps(gps)
        alignement = np.mean(gps_times - times)
        print('GPS time offset : ', alignement)

        # Read ahrs logs
        t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
        t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)



        # Slicing according to the f_point and l_point, for all arrays
        if t_begin != 0 and t_last != -1:
            t_0, t_1 = cvt_time(t_begin, t_last)
            # print('Times in if : ', t_0, ', ', t_1)

        else:
            t_0 = t[0]
            t_1 = t[-1]

        f_point = find_index(t, t_0)
        l_point = find_index(t, t_1)
        # print('Indexes : ', f_point, ',', l_point)
        print('Times : ', t_0, ', ', t_1)
        t = t[f_point:l_point]
        acc1 = acc1[f_point:l_point]

        f_point = find_index(t2, t_0)
        l_point = find_index(t2, t_1)
        # print('Slicing t2 : ', f_point, ', ', l_point)
        t_02 = t2[f_point]
        t_12 = t2[l_point]
        delta_t_ahrs = t_0 - t_02
        print('Delta t AHRS : ', delta_t_ahrs)
        t2 = t2[f_point:l_point]
        acc2 = acc2[f_point:l_point]

        # print('Times : ', t_02, ', ', t_12)

        f_point = find_index(times, t_0 - alignement)
        l_point = find_index(times, t_1 - alignement)

        lat = lat[f_point: l_point]
        lon = lon[f_point: l_point]
        altitudes = altitudes[f_point: l_point]
        times = times[f_point: l_point]


        # Getting Z-axis accelerations and normalisation
        acc_z1 = (acc1[:,2])
        acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

        acc_z2 = (acc2[:,2])
        acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)

        # Harmonisation between the two files
        # std1 = np.std(acc_z1)
        # std2 = np.std(acc_z2)
        # coef = std1/std2
        # acc_z2 *= (1+coef)

        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)

        # Getting detection times
        detect_times1 = []
        if df1 != '':
            detect_times1, _ = log_file_detect(df1)
            detect_times1 = [detect_times1[i] for i in range(len(detect_times1)) if (t_0 < detect_times1[i] < t_1) ]
        else:
            detect_times1 = detect_points(acc_z1, seuil, delta_p)
            res = []
            for j in range(len(detect_times1)):
                res.append(t[detect_times1[j][0]])
            detect_times1 = res
        
        detect_times2 = []
        if df2 != '':
            detect_times2, _ = log_file_detect(df2)
            detect_times2 = [detect_times2[i] for i in range(len(detect_times2)) if (t_0 < detect_times2[i] < t_1) ]
        else:
            detect_times2 = detect_points(acc_z2, seuil, delta_p)
            res = []
            for j in range(len(detect_times2)):
                res.append(t2[detect_times2[j][0]])
            detect_times2 = res
            # print('D : ', (detect_times2))
        
        # Getting gps track of gpx file
        lat_x, lon_x, t_x = [], [], []
        if gpx != '':
            lat_x, lon_x, t_x = get_latlon_as_in_nmea_from_gpx(gpx)

            f_point = find_index(t_x, t_0 - alignement)
            l_point = find_index(t_x, t_1 - alignement)
            lat_x = lat_x[f_point:l_point]
            lon_x = lon_x[f_point:l_point]
            print("Temps gpx : ", t_x[0], t_x[-1])
            t_x = t_x[f_point:l_point]
            print("Temps gpx : ", t_x[0], t_x[-1])
            gps_to_csv("cropped_gps_track", lat_x, lon_x, t_x)


        # Getting the timestamps
        time_off += alignement
        lt_time = real_time_read(time, dilat, time_off)

        # Getting frequency filtered detections
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        # # Coefs for graph plotting the different sizes arrays
        # a = len(t)/len(lat)
        # b = len(lat_x)/len(lat)


        # Plt stuff for plotting 
        f1 = plt.figure(figsize=(20,10))
        f1.subplots_adjust(hspace=0.5)
        f1.canvas.set_window_title('Buoy 1')
        f2 = plt.figure()
        f2.canvas.set_window_title('GPS tracks')

        ax1 = f1.add_subplot(311)
        ax2 = f1.add_subplot(323)
        ax4 = f1.add_subplot(325)
        ax6 = f1.add_subplot(324)
        ax7 = f1.add_subplot(326)

        ax3 = f2.add_subplot(111)

        ax1.set_title('GPS Altitude buoy 1')
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax4.set_title('Z accel centered AHRS1')
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax7.set_title('Z accel centered AHRS2')

        ax3.set_title('GPS track')

        ax3.plot(lat, lon, color='blue')
        if len(lat_x) != 0 :
            ax3.plot(lat_x, lon_x, color='green')

        ax1.plot(times, altitudes)

        ax2.plot(t, acc1)
        ax6.plot(t2, acc2)

        ax4.plot(t, acc_z1)
        ax4.plot(t, filtered_z1)
        ax4.plot([t_0, t_1], [seuil, seuil], color='red')
        ax4.scatter(detect_times1, [graph_offset + 6 for i in range(len(detect_times1))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect1)):
            # print(filtered_detect1[j][0])
            # print(filtered_z1[filtered_detect1[j][0]])
            ax4.scatter(t[filtered_detect1[j][0]], graph_offset+2, color='purple')


        ax7.plot(t2, acc_z2)
        ax7.plot(t2, filtered_z2)
        ax7.plot([t_0, t_1], [seuil, seuil], color='red')
        ax7.scatter(detect_times2, [graph_offset + 6 for i in range(len(detect_times2))], color='blue')
        ax7.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect2)):
            # print(filtered_detect2[j][0])
            # print(filtered_z2[filtered_detect2[j][0]])
            ax7.scatter(t2[filtered_detect2[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude buoy 1')
        ax1.set_xlabel("Time (s)")
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax2.legend(['X', 'Y', 'Z'])
        ax2.set_ylim([-1, graph_offset + 7])
        ax2.set_xlabel("Time (s)")
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax6.legend(['X', 'Y', 'Z'])
        ax6.set_ylim([-1, graph_offset + 7])
        ax6.set_xlabel("Time (s)")
        ax3.set_title('GPS track')
        if len(lat_x) != 0 :
            ax3.legend(['Buoy', 'Boat at time t'])
        else:
            ax3.legend(['Buoy', 'Pos at time t'])

        ax4.set_title('Z accel centered AHRS1')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax4.set_ylim([-1, graph_offset + 7])
        ax4.set_xlabel("Time (s)")
        ax7.set_title('Z accel centered AHRS2')
        ax7.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax7.set_ylim([-1, graph_offset + 7])
        ax7.set_xlabel("Time (s)")


        print("\n====================")
        print("Buoy 2")
        print("====================\n")


        f1, f2 = file2[0], file2[1]
        gps = gps_files[1]    
        # Read gps log
        times, gps_times, lat, lon , altitudes = log_gps(gps)
        alignement = np.mean(gps_times - times)
        print('GPS time offset : ', alignement)

        # Read ahrs logs
        t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
        t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)


        # Slicing according to the f_point and l_point, for all arrays

        f_point = find_index(t, t_0)
        l_point = find_index(t, t_1)
        print('Times : ', t_0, ', ', t_1)
        t = t[f_point:l_point]
        acc1 = acc1[f_point:l_point]
        f_point = find_index(t2, t_0)
        l_point = find_index(t2, t_1)
        t_02 = t2[f_point]
        t_12 = t2[l_point]
        delta_t_ahrs = t_0 - t_02
        print('Delta t AHRS : ', delta_t_ahrs)
        t2 = t2[f_point:l_point]
        acc2 = acc2[f_point:l_point]

        print('Times : ', t_02, ', ', t_12)

        f_point = find_index(times, t_0 - alignement)
        l_point = find_index(times, t_1 - alignement)

        lat = lat[f_point: l_point]
        lon = lon[f_point: l_point]
        altitudes = altitudes[f_point: l_point]
        times = times[f_point: l_point]



        # Getting Z-axis accelerations and normalisation
        acc_z1 = (acc1[:,2])
        acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

        acc_z2 = (acc2[:,2])
        acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)


        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)

        # Getting detection times
        detect_times1 = []
        if df1 != '':
            detect_times1, _ = log_file_detect(df1)
            detect_times1 = [detect_times1[i] for i in range(len(detect_times1)) if (t_0 < detect_times1[i] < t_1) ]
        else:
            detect_times1 = detect_points(acc_z1, seuil, delta_p)
            res = []
            for j in range(len(detect_times1)):
                res.append(t[detect_times1[j][0]])
            detect_times1 = res
        
        detect_times2 = []
        if df2 != '':
            detect_times2, _ = log_file_detect(df2)
            detect_times2 = [detect_times2[i] for i in range(len(detect_times2)) if (t_0 < detect_times2[i] < t_1) ]
        else:
            detect_times2 = detect_points(acc_z2, seuil, delta_p)
            res = []
            for j in range(len(detect_times2)):
                res.append(t2[detect_times2[j][0]])
            detect_times2 = res
            # print('D : ', (detect_times2))
        

        # Getting the timestamps
        time_off += alignement
        lt_time = real_time_read(time, dilat, time_off)

        # Getting frequency filtered detections
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        # Coefs for graph plotting the different sizes arrays
        a = len(t)/len(lat)
        b = len(lat_x)/len(lat)


        # Plt stuff for plotting 
        f3 = plt.figure(figsize=(20,10))
        f3.subplots_adjust(hspace=0.5)
        f3.canvas.set_window_title('Buoy 2')

        ax1 = f3.add_subplot(311)
        ax2 = f3.add_subplot(323)
        ax4 = f3.add_subplot(325)
        ax6 = f3.add_subplot(324)
        ax7 = f3.add_subplot(326)


        ax1.set_title('GPS Altitude buoy 2')
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax4.set_title('Z accel centered AHRS1')
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax7.set_title('Z accel centered AHRS2')

        ax3.plot(lat, lon, color='purple')
        # ax3.plot(lat_x, lon_x, color='green')

        ax1.plot(times, altitudes)

        ax2.plot(t, acc1)
        ax6.plot(t2, acc2)

        ax4.plot(t, acc_z1)
        ax4.plot(t, filtered_z1)
        ax4.plot([t_0, t_1], [seuil, seuil], color='red')
        ax4.scatter(detect_times1, [graph_offset + 6 for i in range(len(detect_times1))], color='blue')
        ax4.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect1)):
            # print(filtered_detect1[j][0])
            # print(filtered_z1[filtered_detect1[j][0]])
            ax4.scatter(t[filtered_detect1[j][0]], graph_offset+2, color='purple')


        ax7.plot(t2, acc_z2)
        ax7.plot(t2, filtered_z2)
        ax7.plot([t_0, t_1], [seuil, seuil], color='red')
        ax7.scatter(detect_times2, [graph_offset + 6 for i in range(len(detect_times2))], color='blue')
        ax7.scatter(lt_time, [graph_offset+4 for i in range(len(lt_time))], color='green')
        for j in range(len(filtered_detect2)):
            # print(filtered_detect2[j][0])
            # print(filtered_z2[filtered_detect2[j][0]])
            ax7.scatter(t2[filtered_detect2[j][0]], graph_offset+2, color='purple')



        ax1.set_title('GPS Altitude buoy 1')
        ax1.set_xlabel("Time (s)")
        ax2.set_title('Accelerometers, x, y, z AHRS1')
        ax2.legend(['X', 'Y', 'Z'])
        ax2.set_ylim([-1, graph_offset + 7])
        ax2.set_xlabel("Time (s)")
        ax6.set_title('Accelerometers, x, y, z AHRS2')
        ax6.legend(['X', 'Y', 'Z'])
        ax6.set_ylim([-1, graph_offset + 7])
        ax6.set_xlabel("Time (s)")
        ax3.set_title('GPS track')
        if len(lat_x) != 0 :
            ax3.legend(['Buoy', 'Boat at time t'])
        else:
            ax3.legend(['Buoy', 'Pos at time t'])

        ax4.set_title('Z accel centered AHRS1')
        ax4.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax4.set_ylim([-1, graph_offset + 7])
        ax4.set_xlabel("Time (s)")
        ax7.set_title('Z accel centered AHRS2')
        ax7.legend(['Raw z accel (z - m)/s', 'Band Pass filtered', 'Detection Threshold'])
        ax7.set_ylim([-1, graph_offset + 7])
        ax7.set_xlabel("Time (s)")

    else:
        print("Wrong data type ! Please put in lists !")

# ====================     
# Main functions for detections :

def detection(f1, f2, gps, seuil, delta_p, f_point =0, l_point=-1, output='Standard'):

    # Read gps log
    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    # Read ahrs logs
    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
    t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)

    # Slicing according to the f_point and l_point, for all arrays
    t_0 = t[f_point]
    t_1 = t[l_point]

    t = t[f_point:l_point]
    acc1 = acc1[f_point:l_point]
    t2 = t[f_point:l_point]
    acc2 = acc2[f_point:l_point]

    # Getting Z-axis accelerations and normalisation
    acc_z1 = (acc1[:,2])
    acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

    acc_z2 = (acc2[:,2])
    acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)



    res1, res2 = [], []
    interval1, interval2 = [], []
    if output == 'Standard':
        # Getting detection times
        detect_times1 = detect_points(acc_z1, seuil, delta_p)
        for j in range(len(detect_times1)):
            res1.append(t[detect_times1[j][0]])
            interval1.append([t[detect_times1[j][1][0]], t[detect_times1[j][1][1]]])

        detect_times2 = detect_points(acc_z2, seuil, delta_p)
        for j in range(len(detect_times2)):
            res2.append(t[detect_times2[j][0]])
            interval2.append([t[detect_times2[j][1][0]], t[detect_times2[j][1][1]]])

        return res1, res2, interval1, interval2
    
    elif output == 'Frequency':
        # Getting frequency filtered detections
        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        for j in range(len(filtered_detect1)):
            res1.append(t[filtered_detect1[j][0]])
            interval1.append([t[filtered_detect1[j][1][0]], t[filtered_detect1[j][1][1]]])


        for j in range(len(filtered_detect2)):
            res2.append(t[filtered_detect2[j][0]])
            interval2.append([t[filtered_detect2[j][1][0]], t[filtered_detect2[j][1][1]]])


        return res1, res2, interval1, interval2
    else:
        print('Output parameter invalid !')
        return [], [], [], []



def detection2(f1, f2, gps, seuil, delta_p, t_begin =0, t_last=-1, output='Standard'):

    # Read gps log
    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    # Read ahrs logs
    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
    t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)

    # Slicing according to the f_point and l_point, for all arrays
    if t_begin != 0 and t_last != -1:
        t_0, t_1 = cvt_time(t_begin, t_last)
        # print('Times in if : ', t_0, ', ', t_1)

    else:
        t_0 = t[0]
        t_1 = t[-1]

    f_point = find_index(t, t_0)
    l_point = find_index(t, t_1)
    # print('Indexes : ', f_point, ',', l_point)
    print('Times : ', t_0, ', ', t_1)
    t = t[f_point:l_point]
    acc1 = acc1[f_point:l_point]

    f_point = find_index(t2, t_0)
    l_point = find_index(t2, t_1)
    # print('Slicing t2 : ', f_point, ', ', l_point)
    t_02 = t2[f_point]
    t_12 = t2[l_point]
    delta_t_ahrs = t_0 - t_02
    print('Delta t AHRS : ', delta_t_ahrs)
    t2 = t2[f_point:l_point]
    acc2 = acc2[f_point:l_point]


    # Getting Z-axis accelerations and normalisation
    acc_z1 = (acc1[:,2])
    acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

    acc_z2 = (acc2[:,2])
    acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)



    res1, res2 = [], []
    interval1, interval2 = [], []
    if output == 'Standard':
        # Getting detection times
        detect_times1 = detect_points(acc_z1, seuil, delta_p)
        for j in range(len(detect_times1)):
            res1.append(t[detect_times1[j][0]])
            interval1.append([t[detect_times1[j][1][0]], t[detect_times1[j][1][1]]])

        detect_times2 = detect_points(acc_z2, seuil, delta_p)
        for j in range(len(detect_times2)):
            res2.append(t[detect_times2[j][0]])
            interval2.append([t[detect_times2[j][1][0]], t[detect_times2[j][1][1]]])

        return res1, res2, interval1, interval2
    
    elif output == 'Frequency':
        # Getting frequency filtered detections
        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        for j in range(len(filtered_detect1)):
            res1.append(t[filtered_detect1[j][0]])
            interval1.append([t[filtered_detect1[j][1][0]], t[filtered_detect1[j][1][1]]])


        for j in range(len(filtered_detect2)):
            res2.append(t[filtered_detect2[j][0]])
            interval2.append([t[filtered_detect2[j][1][0]], t[filtered_detect2[j][1][1]]])


        return res1, res2, interval1, interval2
    else:
        print('Output parameter invalid !')
        return [], [], [], []



def get_gps_intervals(f1, f2, gps, seuil, delta_p, t_begin =0, t_last=-1, output='Standard'):

    # Read gps log
    times, gps_times, lat, lon , altitudes = log_gps(gps)
    alignement = np.mean(gps_times - times)
    print('GPS time offset : ', alignement)

    # Read ahrs logs
    t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
    t2, acc2, gyr, mag, temp, rate = log_file(f2, alignement)

    # Slicing according to the f_point and l_point, for all arrays
    if t_begin != 0 and t_last != -1:
        t_0, t_1 = cvt_time(t_begin, t_last)
        # print('Times in if : ', t_0, ', ', t_1)

    else:
        t_0 = t[0]
        t_1 = t[-1]

    f_point = find_index(t, t_0)
    l_point = find_index(t, t_1)
    # print('Indexes : ', f_point, ',', l_point)
    print('Times : ', t_0, ', ', t_1)
    t = t[f_point:l_point]
    acc1 = acc1[f_point:l_point]

    f_point = find_index(t2, t_0)
    l_point = find_index(t2, t_1)
    # print('Slicing t2 : ', f_point, ', ', l_point)
    t_02 = t2[f_point]
    t_12 = t2[l_point]
    delta_t_ahrs = t_0 - t_02
    print('Delta t AHRS : ', delta_t_ahrs)
    t2 = t2[f_point:l_point]
    acc2 = acc2[f_point:l_point]


    # Getting Z-axis accelerations and normalisation
    acc_z1 = (acc1[:,2])
    acc_z1 = np.abs(acc_z1 - np.mean(acc_z1))/np.std(acc_z1)

    acc_z2 = (acc2[:,2])
    acc_z2 = np.abs(acc_z2 - np.mean(acc_z2))/np.std(acc_z2)



    res1, res2 = [], []
    interval1, interval2 = [], []
    if output == 'Standard':
        # Getting detection times
        detect_times1 = detect_points(acc_z1, seuil, delta_p)
        for j in range(len(detect_times1)):
            res1.append(t[detect_times1[j][0]])
            interval1.append([t[detect_times1[j][1][0]], t[detect_times1[j][1][1]]])

        detect_times2 = detect_points(acc_z2, seuil, delta_p)
        for j in range(len(detect_times2)):
            res2.append(t[detect_times2[j][0]])
            interval2.append([t[detect_times2[j][1][0]], t[detect_times2[j][1][1]]])
    
    elif output == 'Frequency':
        # Getting frequency filtered detections
        filtered_z1 = freq_filter(acc_z1)
        filtered_z2 = freq_filter(acc_z2)
        filtered_detect1 = detect_points(filtered_z1, seuil, delta_p)
        filtered_detect2 = detect_points(filtered_z2, seuil, delta_p)

        for j in range(len(filtered_detect1)):
            res1.append(t[filtered_detect1[j][0]])
            interval1.append([t[filtered_detect1[j][1][0]], t[filtered_detect1[j][1][1]]])


        for j in range(len(filtered_detect2)):
            res2.append(t[filtered_detect2[j][0]])
            interval2.append([t[filtered_detect2[j][1][0]], t[filtered_detect2[j][1][1]]])

    else:
        print('Output parameter invalid !')
        return [], [], [], []



    gps_int1 = []
    for i in interval1:
        t1, t2 = i[0], i[1]

        f_point = find_index(times, t1 - alignement)
        l_point = find_index(times, t2 - alignement)
        # print('Indexes : ', f_point, l_point)
        if f_point != l_point:
            lat_i = lat[f_point: l_point]
            lon_i = lon[f_point: l_point]
            minlat, maxlat, minlon, maxlon = np.min(lat_i), np.max(lat_i), np.min(lon_i), np.max(lon_i)
            gps_int1.append([minlat, maxlat, minlon, maxlon])
        else :
            lat_i = lat[f_point]
            lon_i = lon[f_point]
            minlat, maxlat, minlon, maxlon = np.min(lat_i), np.max(lat_i), np.min(lon_i), np.max(lon_i)
            gps_int1.append([minlat, maxlat, minlon, maxlon])


    gps_int2 = []
    for i in interval2:
        t1, t2 = i[0], i[1]

        f_point = find_index(times, t1 - alignement)
        l_point = find_index(times, t2 - alignement)

        if f_point != l_point:
            lat_i = lat[f_point: l_point]
            lon_i = lon[f_point: l_point]
            minlat, maxlat, minlon, maxlon = np.min(lat_i), np.max(lat_i), np.min(lon_i), np.max(lon_i)
            gps_int2.append([minlat, maxlat, minlon, maxlon])
        else :
            lat_i = lat[f_point]
            lon_i = lon[f_point]
            minlat, maxlat, minlon, maxlon = np.min(lat_i), np.max(lat_i), np.min(lon_i), np.max(lon_i)
            gps_int2.append([minlat, maxlat, minlon, maxlon])


    return gps_int1, gps_int2