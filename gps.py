
import serial
import datetime
import logging
import threading




class gps:
	id_num = 1

	def __init__(self, port, baudrate = 4800):

		self.id = gps.id_num
		gps.id_num += 1

		self.__baudrate = baudrate
		self.__port = port
		self.__ser = serial.Serial(self.__port, self.__baudrate, timeout=0.1)
		self.__ser.reset_input_buffer()

		print("Serial connection for gps" + str(self.id) + (" ok"))


		self.__last_data = []

		now = datetime.datetime.now()
		current_time = now.strftime("_%H_%M_%S")
		self.__file_name = '/home/sillage2/sillage_python/Sillage/log/gps'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time)
		# self.__file_name = 'test/gps'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time)
		# self.__file_name = 'log/gps'+ str(self.id) + '_log_' + str(datetime.date.today())+ str(current_time)
		print("File_name : ", self.__file_name)


		self.__logger = logging.getLogger('Gps' + str(self.id))
		self.f_handler = logging.FileHandler(self.__file_name + '.log')
		self.f_handler.setLevel(logging.INFO)
		self.f_format = logging.Formatter('%(asctime)s ; %(name)s ; %(message)s')
		self.f_handler.setFormatter(self.f_format)
		self.__logger.addHandler(self.f_handler)


		self.__thread = threading.Thread(target=self.run, name='AHRS' + str(self.id))

		self.__thread.start()

	
	@property
	def last_data(self):
		return self.__last_data


	def run(self):
		while True:
			self.get_data()


	def get_data(self):
		if self.__ser.in_waiting > 0:
						
			# Whole raw datas :
			# NMEA, GPGGA here
			try:
				line = self.__ser.readline()
				line = line.decode('utf-8').rstrip()

				line = line.split(',')
				data_buff = []
				if line[0] == '$GPGGA':
					# print('GPS : ', line)
					self.__last_data = line
					self.__logger.error(line)
					# return line

			except:
				print("Erreur gps ", self.id)
				# return self.__last_data
				pass
