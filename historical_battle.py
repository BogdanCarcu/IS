"""Training an additional entity type

This script shows how to add a new entity type to an existing pre-trained NER
model. In practice, you'll need many more sentences — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. 

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy

train_text = []
with open('Training.txt', 'r') as myfile:
    train_text = myfile.read().replace('\n', '').split('---')

# new entity labels
LABEL1 = 'BELLIGERENT1'
LABEL2 = 'BATTLE'
LABEL3 = 'LOCATION'
LABEL4 = 'DATE'
LABEL5 = 'LEADER'
LABEL6 = 'RESULT'
LABEL7 = 'BELLIGERENT2'

TRAIN_DATA = [
     (train_text[0], {'entities': [(0, 22, 'BATTLE'), (45, 57, 'DATE'), (59, 72, 'LOCATION'), (114, 147, 'LOCATION'), (151, 162, 'BELLIGERENT1'), (184, 202, 'LEADER'), (244, 261, 'BELLIGERENT2'), (314, 332, 'LEADER'), (340, 353, 'BELLIGERENT2'), (375, 404, 'LEADER'), (449, 475, 'RESULT')]}),
     (train_text[1], {'entities': [(0, 22, 'BATTLE'), (37, 61, 'BATTLE'), (76, 105, 'LOCATION'), (157, 175, 'DATE'), (185, 218, 'LEADER'), (221, 246, 'BELLIGERENT1'), (251, 284, 'BELLIGERENT2'), (287, 306, 'LEADER'), (308, 323, 'LOCATION'), (325, 333, 'LOCATION'), (338, 352, 'LOCATION'), (392, 506, 'RESULT'), (584, 616, 'RESULT')]}),
     (train_text[2], {'entities': [(0, 21, 'BATTLE'), (25, 46, 'BATTLE'), (63, 84, 'DATE'), (86, 104, 'LOCATION'), (106, 166, 'BELLIGERENT1'), (175, 201, 'LEADER'), (206, 243, 'LEADER'), (269, 280, 'BELLIGERENT2'), (284, 317, 'LEADER'), (448, 491, 'RESULT'), (564, 609, 'RESULT'), (792, 823, 'RESULT')]}),
     (train_text[3], {'entities': [(0, 24, 'BATTLE'), (26, 58, 'DATE'), (115, 137, 'BELLIGERENT1'), (149, 161, 'BELLIGERENT2'), (177, 199, 'LOCATION'), (216, 234, 'LOCATION'), (410, 451, 'RESULT'), (493, 537, 'RESULT'), (1074, 1102, 'LOCATION'), (1387, 1399, 'LEADER')]}),
     (train_text[4], {'entities': [(0, 21, 'BATTLE'), (41, 59, 'BATTLE'), (105, 129, 'BATTLE'), (141, 157, 'DATE'), (162, 180, 'LOCATION'), (228, 253, 'BELLIGERENT1'), (268, 293, 'LEADER'), (298, 309, 'BELLIGERENT1'), (328, 347, 'LEADER'), (355, 367, 'BELLIGERENT2'), (398, 435, 'LEADER'), (502, 592, 'RESULT'), (737, 823, 'RESULT'), (836, 897, 'RESULT')]}),
     (train_text[5], {'entities': [(0, 25, 'BATTLE'), (48, 77, 'BELLIGERENT1'), (86, 109, 'LEADER'), (119, 133, 'BELLIGERENT2'), (137, 145, 'LEADER'), (292, 318, 'DATE'), (323, 361, 'LOCATION'), (687, 699, 'LEADER')]}),
     (train_text[6], {'entities': [(0, 24, 'BATTLE'), (120, 168, 'DATE'), (181, 196, 'BELLIGERENT1'), (204, 221, 'LEADER'), (226, 242, 'BELLIGERENT2'), (250, 266, 'LEADER'), (794, 834, 'LOCATION'), (889, 923, 'LOCATION'), (1097, 1142, 'RESULT')]}),
     (train_text[7], {'entities': [(0, 20, 'BATTLE'), (35, 46, 'DATE'), (48, 67, 'BELLIGERENT1'), (75, 118, 'LEADER'), (166, 196, 'LEADER'), (198, 215, 'BELLIGERENT2')]}),
     (train_text[8], {'entities': [(0, 21, 'BATTLE'), (25, 37, 'DATE'), (66, 73, 'LEADER'), (77, 83, 'BELLIGERENT1'), (93, 107, 'BELLIGERENT2'), (114, 150, 'LEADER'), (250, 317, 'RESULT'), (325, 407, 'RESULT')]}),
     (train_text[9], {'entities': [(0, 22, 'BATTLE'), (37, 52, 'DATE'), (66, 73, 'LEADER'), (65, 83, 'BELLIGERENT1'), (87, 116, 'LEADER'), (125, 137, 'BELLIGERENT2'), (148, 181, 'LEADER'), (183, 223, 'RESULT'), (277, 298, 'LOCATION'), (364, 387, 'RESULT')]})
    ]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, new_model_name='model', output_dir=None, n_iter=20):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe('ner')

    ner.add_label(LABEL1)   # add new entity labels to entity recognizer
    ner.add_label(LABEL2)
    ner.add_label(LABEL3)
    ner.add_label(LABEL4)
    ner.add_label(LABEL5)
    ner.add_label(LABEL6)
    ner.add_label(LABEL7)

    if model is None:
        optimizer = nlp.begin_training()
    else:
        # Note that 'begin_training' initializes the models, so it'll zero out
        # existing entity types.
        optimizer = nlp.entity.create_optimizer()


    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update([text], [annotations], sgd=optimizer, drop=0.35,
                           losses=losses)
            print(losses)

    # test the trained model

    evaluation_data = [{'location' : 'near Kahlenberg Mountain', 
                        'name' : 'The Battle of Vienna',
                        'date' : 'September 12, 1683',
                        'belligerent1' : 'Polish-Austrian-German forces',
                        'belligerent2' : 'Ottoman Empire',
                        'leaders' : ['King of Poland John III Sobieski', 'Grand Vizier Merzifonlu Kara Mustafa Pasha'],
                        'result' : ['turning point in the 300-year struggle between the forces']},
                       {'location' : 'Kagoshima, Japan',
                        'name' : 'The Battle of Shiroyama',
                        'date' : '24 September 1877',
                        'belligerent1' : 'Samurai',
                        'belligerent2' : 'Imperial Japanese Army',
                        'leaders' : ['Saigo Takamori', 'Yamagata Aritomo', 'Kawamura Sumiyoshi'],
                        'result' : ['the end of the Satsuma Rebellion', "the annihilation of Saigo's army"]}]

    accuracy = 0.0
    print(" ")

    for i in range(0, 2):
        data = []
        with open('Test.txt', 'r') as myfile:
            data = myfile.read().replace('\n', '').split('---')
        doc = nlp(data[i])
        
        locations = set()
        dates = set()
        first_army = set()
        second_army = set()
        results = set()
        battle_names = set()
        leaders = set()

        for ent in doc.ents:
            if ent.label_ == 'BATTLE':
                battle_names.add(ent.text)
            elif ent.label_ == 'DATE':
                dates.add(ent.text)
            elif ent.label_ == 'BELLIGERENT1':
                first_army.add(ent.text)
            elif ent.label_ == 'BELLIGERENT2':
                second_army.add(ent.text)
            elif ent.label_ == 'LEADER':
                leaders.add(ent.text)
            elif ent.label_ == 'LOCATION':
                locations.add(ent.text)
            elif ent.label_ == 'RESULT':
                results.add(ent.text)

        #print the labeled items

        print("Battle name(s): " + str(battle_names))

        print("Date(s): " + str(dates))
       
        print("Location(s): " + str(locations))

        print("Belligerents: " + str(first_army) + " VS. " + str(second_army))

        print("Leader(s): " + str(leaders))

        print("Result(s): " + str(results))

        print("---------------------------\n")

        for name in battle_names:
            if name in evaluation_data[i]['name']:
                accuracy += 1
        for date in dates:
            if date in evaluation_data[i]['date']:
                accuracy += 1
        for location in locations:
            if location in evaluation_data[i]['location']:
                accuracy += 1
        for belligerent in  first_army:
            if belligerent in evaluation_data[i]['belligerent1']:
                accuracy += 1
        for belligerent in second_army:
            if belligerent in evaluation_data[i]['belligerent2']:
                accuracy += 1
        for leader in leaders:
            for ev_leader in evaluation_data[i]['leaders']:
                if leader in ev_leader:
                    accuracy += 1
        for result in results:
            for ev_result in evaluation_data[i]['result']:
                if result in ev_result:
                    accuracy += 1

    accuracy = (accuracy * 100.0) / 18.0
    print("Accuracy: %.2f" %(accuracy))
   
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)