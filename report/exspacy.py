import spacy
import csv

nlp = spacy.load('en')
#doc = nlp('French soliers march against the German army!')

with open('data.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '')

doc = nlp('The Battle of Vienna took place at Kahlenberg Mountain near Vienna on 12 September 1683[1] after the imperial city had been besieged by the Ottoman Empire for two months. The battle was fought by the Habsburg Monarchy, the Polish–Lithuanian Commonwealth and the Holy Roman Empire, under the command of King John III Sobieski against the Ottomans and their vassal and tributary states. The battle marked the first time the Commonwealth and the Holy Roman Empire had cooperated militarily against the Ottomans, and it is often seen as a turning point in history, after which "the Ottoman Turks ceased to be a menace to the Christian world".[18] In the ensuing war that lasted until 1699, the Ottomans lost almost all of Hungary to the Holy Roman Emperor Leopold I.[18]')

dictionary = {}
with open('nationalities.csv', 'r', encoding='utf8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
    	k, v = row
    	dictionary[k] = v

nationalities = set()
nations = set()
later_nations = set()

#religions to filter out
religions = ['Christian', 'Muslim']

for token in doc.ents:
	if token.label_ is "NORP":
		if token.text not in religions:
			nationalities.add(token.text)
	if token.label_ is "GPE":
		nations.add(token.text)

for nationality in nationalities:
	if nationality in dictionary:
		later_nations.add(dictionary[nationality])


# print("Nationalities: ")
# print(nationalities)
# print("Nations:")
# print (nations)
#print("Later found nations:")
#print(later_nations)

for chunk in doc.noun_chunks:
    print(chunk.text)
	
