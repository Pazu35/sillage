import serial
# from ahrs.filters import Mahony
# from ahrs import Quaternion
import numpy as np
import time
import filt
import datetime
import logging 
import threading  #/!\ Not used anymore because of unknown logging speed issues



# ===============================

f = open("config.txt", "r")
lines = f.readlines()

for line in lines:
	l = line.strip().split(': ')
	# print(l)
	if l[0] == 'buoy_id':
		buoy_id = l[1]
f.close()
# buoy_id = 'sillage2'
print("Id : ", buoy_id)


# ===============================


class AHRS:
	id_num = 1
	def __init__(self,port, baudrate=115200):
		
		self.__start_time = time.time()
		self.id = AHRS.id_num
		AHRS.id_num += 1
		self.__baudrate = baudrate
		self.__port = port
		self.__ser = serial.Serial(self.__port, self.__baudrate, timeout=1)
		self.__ser.reset_input_buffer()
		print("Serial connection for ahrs" + str(self.id) + " ok on port " + port)
		self.__last_data = []
		self.__frequency = 50.


		self.__acc_filt1 = filt.MovingAverageFilter(40*self.__frequency)
		self.__last_z_acc = 0
		self.__nb_mesure = 0

		now = datetime.datetime.now()
		current_time = now.strftime("_%H_%M_%S")
		self.__file_name = '/home/'+ buoy_id + '/sillage_python/Sillage/log/'+ buoy_id +'_ahrs'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time)
		# self.__file_name = 'test/ahrs'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time)
		# self.__file_name = 'log/ahrs'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time)
		print("File_name : ", self.__file_name, '\n')


		self.__logger = logging.getLogger('Ahrs' + str(self.id))
		self.f_handler = logging.FileHandler(self.__file_name + '.log')
		self.f_handler.setLevel(logging.INFO)
		self.f_format = logging.Formatter('%(asctime)s ; %(name)s ; %(message)s')
		self.f_handler.setFormatter(self.f_format)
		self.__logger.addHandler(self.f_handler)


		# self.__logger_detec = logging.getLogger('Detect AHRS' + str(self.id))
		# self.f_handler_detec = logging.FileHandler('/home/'+ buoy_id + '/sillage_python/Sillage/log/'+ buoy_id +'_detect_ahrs'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time) + '.log')
		# # self.f_handler_detec = logging.FileHandler('test/detect_ahrs'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time) + '.log')
		# self.f_handler_detec.setLevel(logging.INFO)
		# self.f_format_detec = logging.Formatter('%(asctime)s ; %(name)s ; %(message)s')
		# self.f_handler_detec.setFormatter(self.f_format_detec)
		# self.__logger_detec.addHandler(self.f_handler_detec)

		# self.__realtime_detect = filt.MovingAverageFilter(40*self.__frequency)


		self.__res1 = []
		self.__res2 = []

		self.__proximity = 5*self.__frequency
		self.__param_seuil = 3.6

		self.__last_detection_time = self.__start_time - time.time()

		# self.__thread = threading.Thread(target=self.run, name='AHRS' + str(self.id))

		# self.__thread.start()


	@property
	def last_data(self):
		return self.__last_data

		
	def run(self):
		while True:
			self.get_data()

	def get_data(self):
		if self.__ser.in_waiting > 0:
						
			# Whole raw datas :
			# rtcDate,rtcTime,aX,aY,aZ,gX,gY,gZ,mX,mY,mZ,imu_degC,output_Hz,
			line = self.__ser.readline()
			data_buff = []
			try:
				line = line.decode('utf-8').rstrip()
				line = line.split(',')
				line = line[2:13]
				string = ''
				for num in line:
					data_buff += [float(num)]
					string += str(float(num)) + ','
				self.__logger.error(string)
			except:
				print("Erreur ahrs ", self.id)
				# self.__ser.reset_input_buffer()

				pass

			if len(data_buff) == 11:
				self.__nb_mesure += 1
				self.__last_data = data_buff
				self.__last_z_acc = data_buff[2]
				# self.__ser.reset_input_buffer()


				# self.detect()
				# return self.__last_data
		# 	else:
		# 		return self.__last_data
		# else:
		# 	return self.__last_data


	# def detect(self):
	# 	value_i = self.__last_z_acc
	# 	mean = self.__realtime_detect.fir_radar(value_i)
	# 	seuil = np.std(np.array(self.__realtime_detect.data))*self.__param_seuil


	# 	if np.abs(value_i - mean) > seuil:
			

	# 		# Grouping close by values
	# 		if self.__res2 == []:
	# 			self.__res2.append(self.__nb_mesure)
	# 		elif self.__nb_mesure - self.__res2[-1] < self.__proximity:
	# 			self.__res2.append(self.__nb_mesure)
	# 			self.__last_detection_time = time.time() - self.__start_time #Reset the timer when detected above threshold

	# 		# When done grouping, get the average for detection time
	# 		else :
	# 			self.t_detect = int(np.round(np.mean(np.array(self.__res2))))
	# 			# print("AHRS " + str(self.id) + " detected wave at t = " + str(self.__last_detection_time))
	# 			self.__logger_detec.error(str(self.__last_detection_time))
	# 			self.__res1.append([self.t_detect, [self.__res2[0], self.__res2[-1]]])

	# 			self.__last_detection_time = time.time() - self.__start_time
	# 			self.__res2 = [self.__nb_mesure]

	# 	if time.time()- self.__start_time - self.__last_detection_time > 2.0 and self.__res2 != []:
	# 			self.__last_detection_time = time.time() - self.__start_time
	# 			self.t_detect = int(np.round(np.mean(np.array(self.__res2))))
	# 			# print("AHRS " + str(self.id) + " detected wave at t = " + str(self.__last_detection_time))
	# 			self.__logger_detec.error(str(self.__last_detection_time))
	# 			self.__res1.append([self.t_detect, [self.__res2[0], self.__res2[-1]]])
	# 			self.__res2 = []








