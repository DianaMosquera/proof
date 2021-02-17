from polyglot.text import Text
from polyglot.detect import Detector
import icu
import string
import pandas as pd
from xs
downloader import downloader
import numpy as np
import nltk
import re
import csv



def strip_links(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def strip_all_entities(text):
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)

def detect_language(x):
    try:
        poly_obj=Detector(x, quiet=True)
        text_lang = icu.Locale.getDisplayName(poly_obj.language.locale)
    except Exception as e:
        print(e)
        text_lang = 'Unknown'
        
    return text_lang

def read_age_lexica(directory):
    """reads in the raw text

        Args:
            directory: location of the age lexicon file
        Returns:
            df: dataframe containing words and weights from age lexicon
    """

    age_lexica = {}
    with open(directory, mode='r') as infile:
        reader = csv.DictReader(infile)
        for data in reader:
            weight = float(data['weight'])
            term = data['term']
            age_lexica[term] = weight

    del age_lexica['_intercept']
    return age_lexica

def age_predictor(text, age_lexica, age_intercept):
    """cleans the raw text
        Args:
            text: social media post based on which age needs to be inferred
            age_lexica: words and weights pre-calculated
            age_intercept: mean age
        Returns:
            age: predicted age

    """
    ###Test if text contains nulls###
#    if type(text) != str: assert np.isnan(text) == False, 'Text contains nulls'

    words = text.split()
    text_scores = {}
    for word in words:
        text_scores[word] = text_scores.get(word, 0) + 1
    age = 0
    words_count = 0
    for word, count in text_scores.items():
        if word in age_lexica:
            words_count = words_count + count
            age = age + (count * age_lexica[word])

    try:
        age = (age / words_count) + age_intercept
    except:
        age = 0
        
    ###TESTING###
    assert age_intercept == 23.2188604687, 'Age Intercept should be equal to 23.2188604687'
    
    return age



if __name__ == '__main__':
    project_dir ="c://"  #dataset directory
    list_text = "Since lock down I've been smoking (usually only when I have a drink and not many) but I smoked 20 one of the sunny days last week and thought that's it! Been using vype_worldwide for a week now. It stops my cravings for cigarettes and I use the mint filters. Half the price of a packet of cigarettes in Ireland"
    list_text = strip_all_entities(strip_links(list_text))
    language = detect_language(list_text)
    print(language)
    
    #solo ingles
    age_lexica = read_age_lexica(project_dir + 'emnlp14age.csv')
    text = list_text.lower()
    valAge = age_predictor(text, age_lexica, age_intercept=23.2188604687)
    print (valAge)
    