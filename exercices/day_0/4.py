def greet(hour):
	if hour < 0 or hour > 24:
		raise ValueError('Invalid hour: it should be between 0 and 24')
	elif hour < 12:
		print('Good morning !')
	elif hour >= 12 and hour < 20:
		print('Good afternoon !')
	else:
		print('Good evening !')

greet(50)
greet(-5)
greet(8)
greet(14)
greet(22)
