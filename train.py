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
LABEL1 = 'BELLIGERENT'
LABEL2 = 'BATTLE'
LABEL3 = 'LOCATION'
LABEL4 = 'DATE'
LABEL5 = 'LEADER'
LABEL6 = 'RESULT'

TRAIN_DATA = [
     (train_text[0], {'entities': [(0, 22, 'BATTLE'), (45, 57, 'DATE'), (59, 72, 'LOCATION'), (114, 147, 'LOCATION'), (151, 162, 'BELLIGERENT'), (184, 202, 'LEADER'), (244, 261, 'BELLIGERENT'), (314, 332, 'LEADER'), (340, 353, 'BELLIGERENT'), (375, 404, 'LEADER'), (449, 475, 'RESULT')]}),
     (train_text[1], {'entities': [(0, 22, 'BATTLE'), (37, 61, 'BATTLE'), (76, 105, 'LOCATION'), (157, 175, 'DATE'), (185, 218, 'LEADER'), (221, 246, 'BELLIGERENT'), (251, 284, 'BELLIGERENT'), (287, 306, 'LEADER'), (308, 323, 'LOCATION'), (325, 333, 'LOCATION'), (338, 352, 'LOCATION'), (392, 506, 'RESULT'), (584, 616, 'RESULT')]}),
     (train_text[2], {'entities': [(0, 21, 'BATTLE'), (25, 46, 'BATTLE'), (63, 84, 'DATE'), (89, 104, 'LOCATION'), (106, 166, 'BELLIGERENT'), (175, 201, 'LEADER'), (206, 243, 'LEADER'), (269, 280, 'BELLIGERENT'), (284, 317, 'LEADER'), (448, 491, 'RESULT'), (564, 609, 'RESULT'), (792, 823, 'RESULT')]}),
     (train_text[3], {'entities': [(0, 24, 'BATTLE'), (26, 58, 'DATE'), (115, 137, 'BELLIGERENT'), (149, 161, 'BELLIGERENT'), (177, 199, 'LOCATION'), (216, 234, 'LOCATION'), (410, 451, 'RESULT'), (493, 537, 'RESULT'), (1074, 1102, 'LOCATION'), (1387, 1399, 'LEADER')]}),
     (train_text[4], {'entities': [(0, 21, 'BATTLE'), (41, 59, 'BATTLE'), (105, 129, 'BATTLE'), (141, 157, 'DATE'), (162, 180, 'LOCATION'), (228, 253, 'BELLIGERENT'), (268, 293, 'LEADER'), (298, 309, 'BELLIGERENT'), (328, 347, 'LEADER'), (355, 367, 'BELLIGERENT'), (398, 435, 'LEADER'), (502, 592, 'RESULT'), (737, 823, 'RESULT'), (836, 897, 'RESULT')]}),
     (train_text[5], {'entities': [(0, 25, 'BATTLE'), (48, 77, 'BELLIGERENT'), (86, 109, 'LEADER'), (119, 133, 'BELLIGERENT'), (137, 145, 'LEADER'), (292, 318, 'DATE'), (323, 361, 'LOCATION'), (687, 699, 'LEADER')]}),
     (train_text[6], {'entities': [(0, 24, 'BATTLE'), (120, 168, 'DATE'), (181, 196, 'BELLIGERENT'), (204, 221, 'LEADER'), (226, 242, 'BELLIGERENT'), (250, 266, 'LEADER'), (794, 834, 'LOCATION'), (889, 923, 'LOCATION'), (1097, 1142, 'RESULT')]}),
     (train_text[7], {'entities': [(0, 20, 'BATTLE'), (35, 46, 'DATE'), (48, 67, 'BELLIGERENT'), (75, 118, 'LEADER'), (166, 196, 'LEADER'), (198, 215, 'BELLIGERENT')]})
    ]

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, new_model_name='battle', output_dir=None, n_iter=20):
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
                nlp.update([text], [annotations], sgd=optimizer, drop=0.0,
                           losses=losses)
            print(losses)

    # test the trained model
    data = ""
    with open('vienna.txt', 'r') as myfile:
        data = myfile.read().replace('\n', '')
    doc = nlp(data)
    print("Entities: ")
    for ent in doc.ents:
        print(ent.label_, ent.text)

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