#!/usr/bin/env python3.5
#!/usr/bin/python3.5

'''
@summary: Fact Finding and Checking
@author: Anil Kumar Nayak
@copyright: IDIR lab
@organization: University of Texas at Arlington
@group HeroX: 
@since: 5th Nov 2016
@version: 8.0
@attention: This program will read two txt file,
@attention: First txt file, whose contents output of googler api  [Name of the first file must be : googlesearch.txt]
@attention: googler api at github https://github.com/jarun/googler 
@attention: Second txt file, which has one line for the claim [Name of the second file must be : claim.txt]
@note: This program will take input from the output of the googler api from git as mentioned above
@note: and will crawl through the linked from the google search and find the similarity of the query 
@note: that has entered against the google search
@todo: Implement google search, so that it could be independent of the above library

@requires: Following Modules to be installed in python
    beautifulsoup4
    nltk
    puket
    urllib.request
    http.cookiejar

@requires: Python 3.5
@return: Python dictionary as {'url': url,'rating': rating,'sentence': sentence,'justification': justification}


@change: v6.0 : 3th Nov 2016 : Modified the cosine similarity calculation
@change: v7.0 : 4th Nov 2016 : Implemented the whole google search link scan and finding cosine similarity of whole page instead of selected paragraph
@change: v8.0 : 5th Nov 2016 : Added Semilar integration to fact finding
@change: v9.0 : 23rd Nov 2016 
'''
import collections
import concurrent.futures as futures
import json
import logging
import math
import os
import re
import urllib
import urllib.request as _request
import bs4
import nltk
import nltk.tokenize as _tokenizer
import numpy as np
import requests
import pdb

_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
            'Accept-Encoding': 'none',
            'Accept-Language': 'en-US,en;q=0.8',
            'Connection': 'keep-alive'}

_MAX_THREADS = 10
_THREAD_TIME_OUT = 20

#========================================================
#========================================================
#========================================================
def g_comp_temp(pos,similarity_measure):
    lines = open('./Claims/claim'+str(pos)+'.txt','r').readlines()
    claim = lines[0]
    search_results = read_file(pos)
    print(search_results)
    rating_func = sentence_similarity if similarity_measure=='cosine' else sentence_semilar_api
    results = []
    try:
        with futures.ThreadPoolExecutor(max_workers=_MAX_THREADS) as executor:
            workers = [executor.submit(context_crawler, rating_func, claim, link, description)
                        for link, description in search_results]
            for worker in futures.as_completed(workers, _THREAD_TIME_OUT):
                try:
                    result = worker.result()
                except ( Exception, urllib.error.URLError ) as e:
                    link, description = search_results[workers.index(worker)]
                    result = rate_paragraphs(rating_func, claim, link, description, [description])
                    logging.exception('Exception when processing search results.')
                results.append(result)
    except futures._base.TimeoutError:
        logging.exception('TimeoutError.')
    print(results)

def read_file(pos):
    lines = open('./Claims/googlesearch'+str(pos)+'.txt','r').readlines()
    line_dtl = [line[:-1] for line in lines]
    results = []
 
    for i in range(1,len(line_dtl),4):
        result = {'url': line_dtl[i], 'sentence': line_dtl[i+1]}
        results.append(result)
    
    return [(result.get('url'), result.get('sentence')) for result in (results)]

#========================================================
#========================================================
#========================================================


#This method will read the Google Search File created by googler api
_clean_desc_regex = re.compile(r'^[A-Z]{1}[a-z]{2} [0-9]{2}[,] [0-9]{4} [-] ', re.I)

_URL_TIME_OUT = 15
def context_crawler(rating_func, claim, link, description):
    req = _request.Request(link, headers=_HEADERS)  
    with _request.urlopen(req, timeout=_URL_TIME_OUT) as response:
        content_type = response.getheader('Content-Type', 'skip').lower()
        if content_type.startswith('text'):
            paragraphs = extract_paragraphs(response.read()) 
        else:
            logging.warn('Expected text format, but got %s @%s', content_type, link)
            paragraphs = [description]
        return rate_paragraphs(rating_func, claim, link, description, paragraphs)


def extract_paragraphs(html):
    p_tags = bs4.BeautifulSoup(html, 'lxml').find_all('p')
    p_text = [' '.join(p.stripped_strings) for p in p_tags]
    return [text for text in p_text if text]



def rate_paragraphs(rating_func, claim, link, description, paragraphs):
    result = {'url': link, 'rating': None, 'sentence': description, 'justification': description}
    contexts = find_context_paragraph(paragraphs, description)
    max_rating = rating_func(claim, description)
    result['rating'] = max_rating
    for context, sentences in contexts:
        for sentence in sentences:
            rating = rating_func(sentence, claim)
            if rating > max_rating:
                max_rating = rating
                result.update(sentence=sentence, justification=context, rating=rating)
    return result

#This method will filter the paragraphs taken from crawler by the help of the snippet received from googler
def find_context_paragraph(paragraphs, description):
    segments = [desctiption.strip() for desctiption in description.split('...')]
    contexts = []
    for paragraph in paragraphs:
        sentences = _tokenizer.sent_tokenize(paragraph)
        included = set()
        matched = set()
        for i, sentence in enumerate(sentences):
            for segment in segments:
                if segment in sentence:
                    matched.add(i)
                    for j in range(i-2, i+3):
                        if 0 <= j < len(sentences): included.add(j)
        included = sorted(included)
        contexts.append((' '.join(sentences[i] for i in included),
                         [sentences[i] for i in matched]))
    return contexts


def clean_string( dirty_string ):
    spec_char_gone = re.sub( r'(.)[^A-Za-z0-9\s](.)',r'\1\2',dirty_string)
    mr_clean = re.sub( r' ', '+', spec_char_gone )
    return mr_clean


def sentence_semilar_api(first_sentence,second_sentence):
    semilar_rating = 0
    api_url = 'http://localhost:9000/api/match/getscore'
    api_params = { 'text1':clean_string(first_sentence), 'text2':clean_string(second_sentence) }
    
    resp = requests.get( url=api_url, params=api_params, auth=('sh@sh.com','idir1fact2check1') )
    data = json.loads(resp.text)
    try:
        semilar_rating = data['score']
    except KeyError as e:
        print('Ran into a KeyError in getting the similar response')
        return None
    return semilar_rating


#This will used to find the similarity of the sentence
def sentence_similarity(first_sentence,second_sentence):
    first_sentence = first_sentence.lower()
    second_sentence = second_sentence.lower()

    ratio = get_cosine(first_sentence, second_sentence)
    return ratio

def get_cosine(text1, text2):
    vec1 = text_to_vector(text1)
    vec2 = text_to_vector(text2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
       return float(numerator) / denominator

_word_regex = re.compile(r'\w+')
def text_to_vector(text):
    words = _word_regex.findall(text)
    return collections.Counter(words)

for i in range(2,15):
    g_comp_temp(i,'cosine')