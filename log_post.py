import numpy as np
import matplotlib.pyplot as plt
import filt

# from utils import log_gps, log_file, log_file_detect, plot_files, plot_files2, freq_filter, detection, plot_buoys, plot_detections, get_time_from_point, cvt_time, find_index, plot_buoys2, detection2
from utils import *

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


def group_list_elements(l, proximity):
	i = 0
	res = []
	while i < len(l) - 1:
		data_i = l[i]
		while l[i] - data_i < proximity:
			i+= 1
			if i == len(l):
				i -= 1
				break
		data_t = l[i-1]
		i+=1
		res += [((data_i + data_t)//2, [data_i, data_t])]
	return res


def delta_time(detect1, detect2, detect3, proximity):
	n1 = len(detect1)
	n2 = len(detect2)
	n3 = len(detect3)
	# print(n1, n2, n3)
	ct1 = 0
	ct2 = 0
	ct3 = 0
	detection_time = []
	times = []
	for i in range(max([n1, n2, n3])):
		tr1, tr2, tr3 = 0, 0, 0
		try :
			dt1 = detect1[i+ct1][0] - detect2[i+ct2][0]
			dt2 = detect2[i+ct2][0] - detect3[i+ct3][0]
			dt3 = detect1[i+ct1][0] - detect3[i+ct3][0]

			l_t = []
			temps = []


			if dt1 > proximity*2:
				tr1 += 1
			elif dt1 < -2*proximity:
				tr2 += 1
			else:
				l_t.append(dt1)
				temps.append(detect1[i+ct1][0])
				temps.append(detect2[i+ct2][0])

			if dt2 > proximity*2:
				tr2 += 1
			elif dt2 < -2*proximity:
				tr3 += 1
			else:
				l_t.append(dt2)
				temps.append(detect2[i+ct2][0])
				temps.append(detect3[i+ct3][0])

			if dt3 > proximity*2:
				tr1 += 1
			elif dt3 < -2*proximity:
				tr3 += 1
			else:
				l_t.append(dt3)
				temps.append(detect1[i+ct1][0])
				temps.append(detect3[i+ct3][0])

			if tr1 > 0:
				ct1 -= 1

			if tr2 > 0:
				ct2 -= 1

			if tr3 > 0:
				ct3 -= 1

			# print("Data : ", detect1[i+ct1][0], detect2[i+ct2][0], detect3[i+ct3][0])
			# print("Dt : ", dt1, dt2, dt3)
			# print("Trigger : ", tr1, tr2, tr3, (tr1%2)+(tr2%2)+(tr3%2))
			# print('Cnters : ', i, ct1, ct2, ct3)

			if (tr1%2)+(tr2%2)+(tr3%2) <= 1:
				detection_time.append(l_t)
				# print("Tmps :", temps)
				if temps != []:
					times.append(int(np.round(np.mean(np.array(temps)))))
			else :
				print("Detected default at ", i)

			# print("==========\n")
		except:
			print("Reached end of common values")
			break

	return detection_time, times


def delta_time_list(detects, proximity):
	n_list = []
	counters = []
	for i in detects:
		n_list.append(len(i))
		counters.append(0)
	print("N : ", n_list)
	nb_elem = len(detects)
	detection_time = []
	times = []

	for i in range(max(n_list)):
		tr = [0 for i in range(nb_elem)]
		dt = []
		try:
			# Creating a list of the difference of detect time between each measurements
			for j in range(nb_elem): # For every element of detects
				for k in range(1,nb_elem- j): # For every next detections 
					print("Step : ", j, ", ", k + j)
					if k +j >j:
						dt.append(detects[j][i+counters[j]][0] - detects[k+j][i+counters[k+j]][0]) # Add the time difference between the two

			print("Dt : ", dt)
			l_t = []
			temps = []

			k = 0 # Memory of the index of the second element used for the dt list
			track = 0 # Memory of the current element to check in the detect list
			
			for j in range(len(dt)):
				# print("Check k :", k+1)
				k += 1 # index of the second element of the dt list to check with
				
				if k == nb_elem	 - track :
					track += 1
					k = 0
				# print("Step : ", track, ", ", k)

				if dt[j]>proximity*2:
					tr[track] +=1
				elif dt[j]<proximity*-2:
					tr[k]

				else:
					# print("Counters : ", counters)
					# print("Datas : ", i+counters[k])
					l_t.append(dt[j])
					temps.append(detects[track][i+counters[track]][0])
					temps.append(detects[k][i+counters[k]][0])

			S = 0
			for l in range(len(tr)):
				if tr[l] > 0:
					counters[l] -= 1
					S += (tr[l]%2)
			if S <=1:
				detection_time.append(l_t)
				if temps != []:
					times.append(int(np.round(np.mean(np.array(temps)))))
			else :
				print("Detected default at ", i)
		except:
			print("Reached end of common values")
			break

	return detection_time, times


def angles(delta_t, f):
	for i in delta_t:
		if len(i)>1:
			dt1 = i[0]/f
			dt2 = i[1]/f
			dt3 = i[2]/f
			at1 = np.round(np.arctan2(dt2, dt1)*180/np.pi)
			at2 = np.round(np.arctan2(dt3, dt2)*180/np.pi)
			at3 = np.round(np.arctan2(dt3, dt1)*180/np.pi)
			at4 = np.round(np.arctan2(dt1, dt2)*180/np.pi)
			at5 = np.round(np.arctan2(dt2, dt3)*180/np.pi)
			at6 = np.round(np.arctan2(dt1, dt3)*180/np.pi)
			angles = np.array([at1, at2, at3, at4, at5, at6]) - 90

			print("Angles trouves : ", angles)

# t, acc, gyr, mag, temp, rate = log_file('log/ahrs1_log_2022-10-11_09_07_23.txt')
# Heure de fin : 11:44:00


def main_post(str_temps, str_f1, str_f2, str_f3, str_f4, f_point =0, l_point = -1):
	# Parametres
	# Parametres de detection
	seuil = 1.2
	proximity = 250 
	# seuil = 1.5
	# proximity = 50 
	axe = 2

	real_time_detect = str_temps.split(',')
	lt_time = []
	for i in real_time_detect:
		inter1 = i.split(':')
		lt_time.append(float(inter1[0])*3600 + (float(inter1[1]) +  4)*60 + float(inter1[2]))


	print("\n---------------")
	print("Premiere figure ")
	print("---------------\n")



	plt.figure(figsize=(15,10))
	plt.subplots_adjust(hspace=0.5)
	print("AHRS 1")
	print("================")

	t, acc, gyr, mag, temp, rate = log_file(str_f1)
	acc_z = (acc[:,axe][f_point:l_point])
	print("Moyenne : ", np.mean(acc_z))
	print("Ecrat type : ", np.std(acc_z))
	acc_z = np.abs(acc_z - np.mean(acc_z))
	t = t[f_point:l_point]

	detect = [i for i in range(len(acc_z)) if acc_z[i] > seuil]
	print('Nb de detections : ', len(detect))
	detect1 = detect_points(acc_z, seuil, proximity)
	print('Nb de detections retenues : ', len(detect1))
	# print(detect1)

	plt.subplot(4,1,1)
	plt.plot(t, acc_z, color ='green')
	plt.title('Measured acceleration towards z axis (AHRS 1)', color='red')

	plt.subplot(4,1,2)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect1)):
		plt.scatter(t[detect1[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Waves detected (AHRS 1)', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')

	# # Parce que 2 fois moins de points
	# f_point = f_point//2
	# l_point = l_point//2

	print("AHRS 2 ")
	print("================")

	t, acc, gyr, mag, temp, rate = log_file(str_f2)

	acc_z = (acc[:,axe][f_point:l_point])
	print("Moyenne : ", np.mean(acc_z))
	print("Ecrat type : ", np.std(acc_z))

	acc_z = np.abs(acc_z - np.mean(acc_z))
	t = t[f_point:l_point]

	detect = [i for i in range(len(acc_z)) if acc_z[i] > seuil]
	print('Nb de detections : ', len(detect))
	detect2 = detect_points(acc_z, seuil, proximity)
	print('Nb de detections retenues : ', len(detect2))
	# print(detect2)


	plt.subplot(4,1,3)

	plt.plot(t, acc_z, color ='green')
	plt.title('Measured acceleration towards z axis (AHRS 2)', color='red')


	plt.subplot(4,1,4)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect2)):
		plt.scatter(t[detect2[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Waves detected (AHRS 2)', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')


	print("\n---------------")
	print("Deuxieme figure ")
	print("---------------\n")

	plt.figure(figsize=(15,10))
	plt.subplots_adjust(hspace=0.5)
	print("AHRS 3 ")
	print("================")
	t, acc, gyr, mag, temp, rate = log_file(str_f3)

	acc_z = (acc[:,axe][f_point:l_point])
	print("Moyenne : ", np.mean(acc_z))
	print("Ecrat type : ", np.std(acc_z))

	acc_z = np.abs(acc_z - np.mean(acc_z))
	t = t[f_point:l_point]

	detect = [i for i in range(len(acc_z)) if acc_z[i] > seuil]
	print('Nb de detections : ', len(detect))

	detect3 = detect_points(acc_z, seuil, proximity)
	print('Nb de detections retenues : ', len(detect3))
	# print(detect3)

	plt.subplot(4,1,1)
	plt.plot(t, acc_z, color ='green')
	plt.title('Measured acceleration towards z axis (AHRS 3)', color='red')

	plt.subplot(4,1,2)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect3)):
		plt.scatter(t[detect3[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Waves detected (AHRS 3)', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')


	print("AHRS 4 ")
	print("================")
	t, acc, gyr, mag, temp, rate = log_file(str_f4)


	acc_z = (acc[:,axe][f_point:l_point])
	print("Moyenne : ", np.mean(acc_z))
	print("Ecrat type : ", np.std(acc_z))

	acc_z = np.abs(acc_z - np.mean(acc_z))
	t = t[f_point:l_point]


	detect = [i for i in range(len(acc_z)) if acc_z[i] > seuil]
	print('Nb de detections : ', len(detect))
	detect4 = detect_points(acc_z, seuil, proximity)
	print('Nb de detections retenues : ', len(detect4))
	# print(detect4)


	plt.subplot(4,1,3)

	plt.plot(t, acc_z, color ='green')
	plt.title('Measured acceleration towards z axis (AHRS 4)', color='red')

	plt.subplot(4,1,4)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect4)):
		plt.scatter(t[detect4[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Waves detected (AHRS 4)', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')



	''' 

	Fusion of datas
	'''


	time_detections, tps = delta_time(detect2, detect3, detect4, proximity)
	print("Delta t : ", time_detections)
	print("Nb detections : ", len(time_detections))
	print("Temps : ", tps)
	angles(time_detections, 10.)


	plt.figure(figsize=(15,10))
	plt.subplots_adjust(hspace=0.5)
	plt.subplot(4,1,1)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect2)):
		plt.scatter(t[detect2[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Detections AHRS 2', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')
	plt.subplot(4,1,2)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect3)):
		plt.scatter(t[detect3[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Detections AHRS 3', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')

	plt.subplot(4,1,3)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in range(len(detect)):
		plt.scatter(t[detect[i]], acc_z[detect[i]], marker = 'o', color = 'blue')
	for i in range(len(detect4)):
		plt.scatter(t[detect4[i][0]], 0, marker = 'o', color = 'green')
	plt.title('Detections AHRS 4', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')

	plt.subplot(4,1,4)
	plt.scatter(t[0], 0)
	plt.scatter(t[-1], 0)
	for i in lt_time:
		plt.axvline(i, color='red')
	for i in range(len(tps)):
		plt.scatter(t[tps[i]], 0, marker = 'o', color = 'green')
	plt.title('Fusion of detection datas', color='red')


	plt.show()


def real_time_test(f1, time, f_point = 0, l_point = -1):



	t, acc, gyr, mag, temp, rate = log_file(f1)
	Hz = int(np.round(np.mean(rate)))
	print("Rate : ", Hz)
	# Parametres de detection
	proximity = 5*Hz
	param_seuil = 3.6


	axe = 2

	# if time!='':
	real_time_detect = time.split(',')
	lt_time = []
	# print("Lt_time : ", lt_time)
	if lt_time != []:
		for i in real_time_detect:
			inter1 = i.split(':')
			lt_time.append(float(inter1[0])*3600 + (float(inter1[1]))*60 + float(inter1[2]))


	acc_z = (acc[:,axe][f_point:l_point])
	t = t[f_point:l_point]

	filt_size1 = 40*Hz
	param_flt = filt.MovingAverageFilter(filt_size1)


	res1 = []
	res2 = []
	plt.figure()
	plt.title('Detection "real time"', color='red')
	plt.plot(t, acc_z)

	for i in range(len(acc_z)):
		value_i = acc_z[i]
		mean = param_flt.fir_radar(value_i)
		seuil = np.std(np.array(param_flt.data))*param_seuil

		if np.abs(value_i - mean) > seuil:
			if res2 == []:
				res2.append(i)
			elif i - res2[-1] < proximity:
				res2.append(i)
			else :
				res1.append([int(np.round(np.mean(np.array(res2)))), [res2[0], res2[-1]]])
				# print("res2 : ", res2)
				res2 = [i]
	if res2 != []:
		res1.append([int(np.round(np.mean(np.array(res2)))), [res2[0], res2[-1]]])
		# print("res2 : ", res2)
		res2 = [i]


			# plt.scatter(t[i], 0, color='red', marker='x' )
		# detect_filt.ajouter(value_i)
		# liste = detect_points(np.abs(np.array(detect_filt) - mean), seuil, proximity)
		# if len(liste) >0:
			# print('===========\n')
			# print("Seuil :", seuil)
			# print("Detections : \n", liste)
			# for j in liste:
				# plt.scatter(t[i - filt_size2 -1 + j[0]], 0, color="red", marker = "x")
				# res1.append(i - filt_size2 -1 + j[0])

	print("Detections : ", res1)
	# 	if len(res1) >= 40:
	# 		lt = group_list_elements(res1, proximity*4)
	# 		for j in lt:
	# 			res2.append(j)	
	# if len(res1) >= 1:
	# 	lt = group_list_elements(res1, proximity)
	# 	for j in lt:
	# 		res2.append(j)
	# print("Releved points : ", res2)			
	for i in res1:
		plt.scatter(t[i[0]], 0, color="red", marker = "x")

	for i in lt_time:
		plt.axvline(i, color='red')
	plt.show()

def real_time_detect_check(f1, detect_f1, time = '', f_point = 0, l_point = -1):
	t, acc, gyr, mag, temp, rate = log_file(f1)
	Hz = int(np.round(np.mean(rate)))
	print("Rate : ", Hz)
	# Parametres de detection
	proximity = 5*Hz
	param_seuil = 3.6

	axe = 2

	time_off = 180
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


	acc_z = (acc[:,axe][f_point:l_point])
	t = t[f_point:l_point]

	filt_size1 = 40*Hz
	param_flt = filt.MovingAverageFilter(filt_size1)


	res1 = []
	res2 = []
	plt.figure()
	plt.title('Detection "real time"', color='red')
	plt.plot(t, acc_z)
	plt.ylim([8, 12])

	for i in range(len(acc_z)):
		value_i = acc_z[i]
		mean = param_flt.fir_radar(value_i)
		seuil = np.std(np.array(param_flt.data))*param_seuil

		if np.abs(value_i - mean) > seuil:
			if res2 == []:
				res2.append(i)
			elif i - res2[-1] < proximity:
				res2.append(i)
			else :
				res1.append([int(np.round(np.mean(np.array(res2)))), [res2[0], res2[-1]]])
				res2 = [i]
	if res2 != []:
		res1.append([int(np.round(np.mean(np.array(res2)))), [res2[0], res2[-1]]])
		res2 = [i]


	offset = 8.5 # For graph purpose
	print("Detections : ", res1)		
	for i in res1:
		plt.scatter(t[i[0]], offset, color="red", marker="x")

	detect_times, __ = log_file_detect(detect_f1) 
	for i in detect_times:
		if i < t[-1] and i > t[0]:
			plt.scatter(i, offset, color="green", marker="o")


	for i in lt_time:
		plt.axvline(i, color='red')


	# plt.show()


def real_time_detect_fusion(f1, f2, f3, f4, detect_f1, detect_f2, detect_f3, detect_f4, time = '',	f_point = 0, l_point = -1, gps = ''):
	

	# GPS time log alignement
	alignement = 0.
	if gps != '':
		times, gps_times, lat, lon , altitudes = log_gps(gps)
		alignement = np.mean(gps_times - times)
		print('GPS time offset : ', alignement)



	detect_times1, __ = log_file_detect(detect_f1, alignement)
	detect_times2, __ = log_file_detect(detect_f2, alignement) 
	detect_times3, __ = log_file_detect(detect_f3, alignement) 
	detect_times4, __ = log_file_detect(detect_f4, alignement)



	offset = 6.5 # For graph purpose

	proximity = 30
	axe = 2
	detect_times1 = group_list_elements(detect_times1, proximity)
	detect_times2 = group_list_elements(detect_times2, proximity) 
	detect_times3 = group_list_elements(detect_times3, proximity) 
	detect_times4 = group_list_elements(detect_times4, proximity)



	t, acc1, gyr, mag, temp, rate = log_file(f1, alignement)
	acc_z1 = (acc1[:,axe][f_point:l_point])
	t1 = t[f_point:l_point]

	t, acc2, gyr, mag, temp, rate = log_file(f2, alignement)
	acc_z2 = (acc2[:,axe][f_point:l_point])
	t2 = t[f_point:l_point]

	t, acc3, gyr, mag, temp, rate = log_file(f3, alignement)
	acc_z3 = (acc3[:,axe][f_point:l_point])
	t3 = t[f_point:l_point]

	t, acc4, gyr, mag, temp, rate = log_file(f4, alignement)
	acc_z4 = (acc4[:,axe][f_point:l_point]) 
	t4 = t[f_point:l_point]



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

	proximity = 10
	time_detections, tps = delta_time(detect_times2, detect_times3, detect_times4, proximity)
	time_detections1, tps1 = delta_time_list([detect_times2, detect_times3, detect_times4], proximity)
	print("Delta t : ", time_detections)
	print("==========")
	print("Delta t list : ", time_detections1)
	# print("Delta t : ", time_detections)
	cnt = 0
	for i in tps:
		if i < t3[-1] and i > t3[0]:
			cnt+=1

	print("Nb detections : ", cnt)
	# print("Temps : ", tps)
	# angles(time_detections, 50.)


	plt.figure(figsize=(15,10))
	plt.subplots_adjust(hspace=0.5)


	plt.subplot(5,1,1)
	plt.ylim(-0.5,13.5)
	plt.scatter(t1[0], offset)
	plt.scatter(t1[-1], offset)
	plt.plot(t1, acc_z1, marker = 'o', color = 'blue')

	for i in detect_times1:
		if i[0] < t1[-1] and i[0] > t1[0]:
			plt.scatter(i[0], offset, color="green", marker="o")

	plt.title('Detections AHRS 1', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')

	# plt.show()
	plt.subplot(5,1,2)
	plt.ylim(-0.5,13.5)
	plt.scatter(t2[0], offset)
	plt.scatter(t2[-1], offset)
	plt.plot(t2, acc_z2, marker = 'o', color = 'blue')

	for i in detect_times2:
		if i[0] < t2[-1] and i[0] > t2[0]:
			plt.scatter(i[0], offset, color="green", marker="o")

	plt.title('Detections AHRS 2', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')


	plt.subplot(5,1,3)
	plt.ylim(-0.5,13.5)
	plt.scatter(t3[0], offset)
	plt.scatter(t3[-1], offset)
	plt.plot(t3, acc_z3, marker = 'o', color = 'blue')

	for i in detect_times3:
		if i[0] < t3[-1] and i[0] > t3[0]:
			plt.scatter(i[0], offset, color="green", marker="o")

	plt.title('Detections AHRS 3', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')

	plt.subplot(5,1,4)
	plt.ylim(-0.5,13.5)
	plt.scatter(t4[0], offset)
	plt.scatter(t4[-1], offset)
	plt.plot(t4, acc_z1, marker = 'o', color = 'blue')

	for i in detect_times4:
		if i[0] < t4[-1] and i[0] > t4[0]:
			plt.scatter(i[0], offset, color="green", marker="o")

	plt.title('Detections AHRS 4', color='red')
	for i in lt_time:
		plt.axvline(i, color='red')

	plt.subplot(5,1,5)
	plt.scatter(t1[0], 0)
	plt.scatter(t1[-1], 0)

	for i in lt_time:
		plt.axvline(i, color='red')
	for i in range(len(tps)):
		if tps[i] < t4[-1] and tps[i] > t4[0]:
			plt.scatter(tps[i], 0, marker = 'o', color = 'green')
	plt.title('Fusion of detection datas', color='red')



def guerledan1():
	time = '07:54:54.82, 07:55:38.65, 07:56:20.48, 07:56:59.63, 07:57:30.27, 07:58:10.93, 07:58:48.31, 07:59:20.84, 08:00:00.39, 08:00:35.97, 08:01:06.07, 08:01:40.59, 08:02:14.72, 08:02:55.85, 08:03:29.77, 08:04:05.97, 08:04:36.77, 08:05:12.87'
	f1 = 'log/log_14_10/ahrs1_log_2022-10-14_07_44_50.log'
	df1 = 'log/log_14_10/detect_ahrs1_log_2022-10-14_07_44_50.log'

	f2 = 'log/log_14_10/ahrs2_log_2022-10-14_07_44_50.log'
	df2 = 'log/log_14_10/detect_ahrs2_log_2022-10-14_07_44_50.log'	

	gps = 'log/log_14_10/gps1_log_2022-10-14_07_44_50.log'

	f_point = 25500
	l_point = 63000
	# Detection parameters
	# For the last Guerledan measures
	seuil = 1.5
	delta_p = 180.


	# Time dilatation parameters for this set of acquisition (Change them in the plot_files fctn:
	# # time_off = 180 + alignement
	# # dilat = 1.35
	plot_files2(f1, f2, gps, seuil, delta_p, df1=df1, df2=df2, time=time, f_point=f_point, l_point=l_point, animate=False)

	# real_time_detect_check(f1, df1, time=time, f_point=f_point, l_point=l_point)
	# real_time_detect_check(f2, df2, time=time, f_point=f_point, l_point=l_point)
	# real_time_detect_check(f3, df3, time=time, f_point=f_point, l_point=l_point)
	# real_time_detect_check(f4, df4, time=time, f_point=f_point, l_point=l_point)

	# main_post(time, f1, f2, f3, f4, f_point, l_point)
	# real_time_detect_fusion(f1, f2, f3, f4, df1, df2, df3, df4, time = time, f_point=f_point, l_point=l_point, gps = gps)



	res1, res2, interval1, interval2 = detection(f1, f2, gps, seuil, delta_p, f_point=f_point, l_point=l_point, output='Frequency')
	# print(res1)
	# print(res2)
	# print(interval1)
	# print(interval2)
	plot_detections([res1, res2], [interval1, interval2])


def rade1():
	time = '13:40:00.00, 13:52:00.00, 13:58:00.00, 14:05:00.00'

	f1 = 'log/log_rade/ahrs1_log_2023-01-19_13_34_14.log'
	gps1 = 'log/log_rade/gps1_log_2023-01-19_13_34_14.log'
	df1 = 'log/log_rade/detect_ahrs1_log_2023-01-19_13_34_14.log'
	gpx = 'log/log_rade/2023-01-19 14_39_15.gpx'

	# For the last Guerledan measures
	seuil = 1.5
	delta_p = 5.
	plot_files(f1, gps1, seuil, delta_p, df1=df1, time=time, gpx=gpx)

def guerledan2():
	# time = '11:09:00.00, 11:10:00.00, 11:12:00.00, 11:13:00.00, 11:14:00.00, 11:15:00.00, 13:43:00.00, 16:11:00.00, 16:12:00.00, 16:13:00.00, 16:14:00.00, 16:15:00.00'
	time = '11:09:00.00, 11:10:00.00, 11:12:00.00, 11:13:00.00, 11:14:00.00, 11:15:00.00'
	f1 = 'log/guerledan2/sillage1_ahrs1_log_2023-02-07_10_45_00.log'

	f2 = 'log/guerledan2/sillage1_ahrs2_log_2023-02-07_10_45_01.log'

	gps = 'log/guerledan2/sillage1_gps1_log_2023-02-07_10_45_00.log'

	f3 = 'log/guerledan2/sillage2_ahrs1_log_2023-02-07_10_45_03.log'

	f4 = 'log/guerledan2/sillage2_ahrs2_log_2023-02-07_10_45_03.log'

	gps2 = 'log/guerledan2/sillage2_gps1_log_2023-02-07_10_45_03.log'

	f_point = 2900
	l_point = 5000
	f_point = 0
	# For the last Guerledan measures
	seuil = 1.5
	delta_p = 5.
	# l_point = -1
	# plot_files(f1, gps, seuil, delta_p, time=time)
	# plot_files(f2, gps, seuil, delta_p, time=time)

	# plot_files2(f1, f2, gps, seuil, delta_p, time=time, f_point=f_point, l_point=l_point)
	# plot_files2(f3, f4, gps2, seuil, delta_p, time=time, f_point=f_point, l_point=l_point)

	plot_buoys(file1=[f1, f2], file2=[f3, f4], gps_files=[gps, gps2], time=time, f_point=f_point, l_point=l_point)


def real_test1():
	# time = '11:09:00.00, 11:10:00.00, 11:12:00.00, 11:13:00.00, 11:14:00.00, 11:15:00.00'
	time = '15:30:30, 15:31:30, 15:32:40, 15:33:38,15:33:42, 15:35:00,15:36:10,15:36:20,15:37:17,15:37:22,15:38:35,15:38:45,15:41:20'
	f1 = 'log/sillage1_ahrs1_log_2023-02-08_14_29_14.log'

	f2 = 'log/sillage1_ahrs2_log_2023-02-08_14_29_14.log'

	gps = 'log/sillage1_gps1_log_2023-02-08_14_29_14.log'

	f3 = 'log/sillage2_ahrs1_log_2023-02-08_14_27_50.log'

	f4 = 'log/sillage2_ahrs2_log_2023-02-08_14_27_50.log'

	gps2 = 'log/sillage2_gps1_log_2023-02-08_14_27_50.log'

	# f_point = 500
	# l_point = 68000
	# l_point = 10000
	# f_point = 0
	# l_point = -1
	t_begin = '15:30:20'
	t_last='17:20:00'
	# For the last Guerledan measures
	seuil = 1.5
	delta_p = 180.
	# plot_files(f1, gps, seuil, delta_p, time=time)
	# plot_files(f2, gps, seuil, delta_p, time=time)

	# plot_files2(f1, f2, gps, seuil, delta_p, time=time, f_point=f_point, l_point=l_point)
	# plot_files2(f3, f4, gps2, seuil, delta_p, time=time, f_point=f_point, l_point=l_point)

	plot_buoys2(seuil, delta_p, file1=[f1, f2], file2=[f3, f4], gps_files=[gps, gps2], time=time, t_begin=t_begin, t_last=t_last)


	t0, t1 = cvt_time(t_begin, t_last)
	print('\nDetection buoy 1')
	res1, res2, interval1, interval2 = detection2(f1, f2, gps, seuil, delta_p,t_begin=t_begin, t_last=t_last, output='Frequency')
	plot_detections([res1, res2], [interval1, interval2], t0, t1)
	gps_box1, gps_box2 = get_gps_intervals(f1, f2, gps, seuil, delta_p,t_begin=t_begin, t_last=t_last, output='Frequency')
	# print(gps_box1)
	write_in_csv('test_complet_ahrs1', interval1, gps_box1)
	write_in_csv('test_complet_ahrs2', interval2, gps_box2)


	print('\nDetection buoy 2')
	res3, res4, interval3, interval4 = detection2(f3, f4, gps2, seuil, delta_p,t_begin=t_begin, t_last=t_last, output='Frequency')
	plot_detections([res3, res4], [interval3, interval4], t0, t1)
	gps_box3, gps_box4 = get_gps_intervals(f3, f4, gps2, seuil, delta_p,t_begin=t_begin, t_last=t_last, output='Frequency')
	# print(gps_box3)
	write_in_csv('test_complet_ahrs3', interval3, gps_box3)
	write_in_csv('test_complet_ahrs4', interval4, gps_box4)



if __name__ == '__main__':

	# time = '13:00:41.79, 13:01:35.09, 13:02:20.27,13:03:11.06, 13:04:11.76, 13:04:53.14, 13:05:45.89, 13:06:30.60, 13:07:24.33, 13:08:11.05, 13:08:49.48, 13:10:11.31, 13:11:37.94, 13:12:50.64, 13:14:17.99, 13:15:43.02, 13:17:20.85'
	# f1 = 'log/log_lac/ahrs1_log_2022-10-12_12_36_08.log'
	# f2 = 'log/log_lac/ahrs2_log_2022-10-12_12_36_08.log'
	# f3 = 'log/log_lac/ahrs3_log_2022-10-12_12_36_08.log'
	# f4 = 'log/log_lac/ahrs4_log_2022-10-12_12_36_08.log'
	# f_point = 28578
	# l_point = 50367

	
	# time = '09:47:26.29, 09:48:03.00, 09:48:52.63, 09:49:20.63, 09:50:00.02, 09:50:37.90, 09:51:10.13, 09:51:50.16, 09:52:25.52, 09:53:07.41, 09:54:07.69, 09:54:43.70, 09:55:39.59, 09:56:24.71, 09:56:59.60, 09:58:00.15, 09:58:42.97'
	# f1 = 'log/log_lac/ahrs1_log_2022-10-13_09_34_42.log'
	# f2 = 'log/log_lac/ahrs2_log_2022-10-13_09_34_42.log'
	# f3 = 'log/log_lac/ahrs3_log_2022-10-13_09_34_42.log'
	# f4 = 'log/log_lac/ahrs4_log_2022-10-13_09_34_42.log'
	# gps1 = 'log/log_lac/gps1_log_2022-10-13_09_34_42.log'
	# f_point = 37085
	# l_point = 73700


	# main_post(time, f1, f2, f3, f4, f_point, l_point)

	# guerledan1()

	# rade1()

	# guerledan2()

	real_test1()

	plt.show()