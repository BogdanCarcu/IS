txt = "The Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William, the Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson, beginning the Norman conquest of England. It took place approximately 7 miles (11 kilometres) northwest of Hastings, close to the present-day town of Battle, East Sussex, and was a decisive Norman victory."
key = "decisive Norman victory"
first_index = txt.find(key)
last_index = txt.find(key) + len(key)
print(first_index)
print(last_index)