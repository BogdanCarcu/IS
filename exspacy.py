import spacy
import csv
import random

nlp = spacy.load('en')
#doc = nlp('French soliers march against the German army!')
# data = ""
# with open('vienna.txt', 'r') as myfile:
#     data=myfile.read().replace('\n', '')

# doc = nlp(data)

# dictionary = {}
# with open('nationalities.csv', 'r', encoding='utf8') as csvfile:
#     csvreader = csv.reader(csvfile, delimiter=',')
#     for row in csvreader:
#     	k, v = row
#     	dictionary[k] = v

# #religions to filter out
# religions = []
# with open('religions.txt', 'r') as file:
#     reader = file.read()
#     reader = reader[0:len(reader)]
#     words = str(reader).split("\n")
#     for word in words:
#     	religions.append(word)


# nationalities = set()
# nations = set()
# later_nations = set()

# for token in doc.ents:
# 	if token.label_ is "NORP":
# 		if token.text not in religions:
# 			nationalities.add(token.text)
# 	if token.label_ is "GPE":
# 		nations.add(token.text)

# for nationality in nationalities:
# 	if nationality in dictionary:
# 		later_nations.add(dictionary[nationality])

TRAIN_DATA = [
     ("Holy Roman Empire was founded that day", {'entities': [(0, 17, 'LOC')]}),
     ("The Austrian Commonwealth took part in that war", {'entities': [(0, 25, "ALLIANCE")]})]

nlp = spacy.blank('en')
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk('/model')

nlp = spacy.load('/model')
ex = "Holy Roman Empire was founded that day"
doc = nlp(ex)

for ent in doc.ents:
	print(ent.text)



# print("Nationalities: ")
# print(nationalities)
# print("Nations:")
# print (nations)
# print("Later found nations:")
# print(later_nations)
# print(religions)
	
