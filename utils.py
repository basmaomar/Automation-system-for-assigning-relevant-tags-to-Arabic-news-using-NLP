import re
import pandas as pd
import numpy as np
import string
#import emoji


from nltk.corpus import stopwords 

ar_stp = pd.read_fwf('stop_words.txt', header=None)
stop_words = set(stopwords.words('arabic') + list(ar_stp[0])) 

def count_word(arr):
    """ takes an array of tweets and return the number of words per tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    Series contains the count of words in each tweet
    """
    count = lambda x: len(str(x).split(" "))

    return arr.apply(count)

def len_tweet(arr):
    """ takes an array of tweets and return the length of tweet per tweet including characters
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    Series contains the length of each tweet
    """
    return arr.str.len() 

def avg_word_len(arr):
    """
    takes an array of tweets and return the the average length of words per tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a Series of the average of words length per tweet
    """
    
    split = lambda x: x.split() # return a list of words. sep=" "
    new_arr = []
    for text in arr:
        words = split(text)
        
        total_words_sum = (sum(len(word) for word in words)) # sum of the lenght of words in a tweet
        
        new_arr.append(total_words_sum/len(words)) # calculate the average
    return pd.Series(new_arr)

def count_stopwords(arr):
    """
    takes an array of tweets and return the the number of stopwords per tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a Series of the count of stopwords in each tweet
    """ 
    count = lambda x: len([x for x in x.split() if x in stop_words])
    
    return pd.Series(arr.apply(count))

def count_tagging(arr):
    """
    takes an array of tweets and return the the number of hashtags / mentions per tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a Series of the count of mentions and hashtags in each tweet
    """
    
    new_arr = []
    for text in arr:
        mentions = re.findall('@[^\s]+', text) # find mentions
        hashtags = re.findall(r'#([^\s]+)', text) # find hashtags 
        
        # print(f'hashtags found {hashtags}, mentions found {mentions}') 
        new_arr.append(len(mentions) + len(hashtags))
    return pd.Series(new_arr)

        
def count_numbers(arr):
    """
    takes an array of tweets and return the count of numbers present per tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a Series of the count of numbers per tweet
    """
    count = lambda x: len([x for x in x.split() if x.isdigit()]) # count of digit present per tweet
    return arr.apply(count)


def frequent_words(arr, topk=10, ascending=False):
    """
    takes an array of tweets and return the top [k] frequent words to all tweets 
    Arguments:
    arr: Series. array of tweets.
    topk: int. top [k] words to return. default = 10.
    ascending: boolean. True: ascending, False: descending. default = False.
    
    Returns:
    a Series of the top [k] frequent words in tweets.
    """
    
    arr = _get_arabic_words(arr, handle_emojies='remove')
    top_words = pd.Series(' '.join(arr).split()).value_counts(ascending=ascending)[:topk]
    return top_words

def view_emojie(arr):
    """
    takes an array of tweets and return the the emojies present in a tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a Series of the emojies of present per tweets.
    """
    
    new_arr = []
    for text in arr:

        # emojies = re.findall('@[^\s]+', tweet) # find emojies
        
        # print(f'emojies found {emojies}) 
        new_arr.append(_extract_emojis(text))
    return pd.Series(new_arr)

def _extract_emojis(str):
  return ''.join(c for c in str if c in emoji.UNICODE_EMOJI)

def view_emoticon(arr):
    """
    takes an array of tweets and return the the emoticon present in a tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a Series of the emoticon of present per tweets.
    """
    arr_emojies = [re.findall(emoji.get_emoji_regexp(), text) for text in arr]
    arr_emot = [_get_emoticon(item) for item in arr_emojies]
    
    new_arr = []
    for emoticon in arr_emot:
        new_arr.append(' '.join(emoticon))
        
    return pd.Series(new_arr)

def term_freq(arr):
    """
    takes an array of tweets and return the freqency of a word in a tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a dataframe of the frequency of words
    """
    arr = _get_arabic_words(arr, handle_emojies='remove')
    
    df = arr.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
    df.columns = ['words','tf']
    return df

def inverse_term_freq(arr):
    """
    takes an array of tweets and return the inverse freqency of a word in a tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a dataframe of the inverse frequency of words
    """
    n = arr.shape[0] # number of rows [tweets]
    
    tf = term_freq(arr)
    for i, word in enumerate(tf['words']):
        sum_words_present = sum(arr.str.contains(word, regex=False))
        log = np.log(n/(sum_words_present + 1)) # +1 to avoid divison by zero.
    
        tf.loc[i, 'idf'] = log
        
    return tf

def tf_idf(arr):
    
    """
    takes an array of tweets and return the term frequency inverse document frequency (tf-idf) of a word in a tweet
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a dataframe of the term frequency inverse document frequency (tf-idf) of words
    """
    
    tf = inverse_term_freq(arr)
    tf['tf-idf'] = tf['tf'] * tf['idf']
    return tf

def _get_arabic_words(arr, handle_emojies='emoticon'):
    """
    the purpose of this function is to get arabic words only out of texts. 
    to be used in term frequncy - inverse - term frequency.
    takes an array of texts and return only arabic words
    Arguments:
    arr: Series. array of tweets.
    emjoies: String. emotiocon or emojie or remove. How to handle emojies either to keep it as emoji or remove 
    or keep the emoticon of an emoji
    
    Returns:
    a Series of arabic words per tweet
    """
    # remove (?, !, , ...)
    punctuation = string.punctuation  + '،'
    arr = [sentence.translate(str.maketrans('', '', punctuation)) for sentence  in arr]
    
    # keep only arabic words  
    arr_text = [re.findall(r'[\u0600-\u06FF]+', sentence) for sentence in arr]
   
    
    arr_text = [_handle_char(item) for item in arr_text]
    
    # remove stop words
    arr_text = [_remove_stopwords(text) for text in arr_text]
    
    new_arr = []

    for text in arr_text:
        new_arr.append(' '.join(text))


    return pd.Series(new_arr)

def _remove_stopwords(arr):
    """"
    the purpose of this function is to remove stop words from a text
   
    Arguments:
    arr: List. of texts
    
    Returns:
    a List of text removed of stop words
    """
    
    return [word for word in arr if word not in stop_words]

def _get_emoticon(arr):
    """"
    the purpose of this function is to get the matching emoticon of an emoji  
    
    takes an array of emojies and return only emoticon per text
    Arguments:
    arr: List. List of List of emojies.
    
    Returns:
    a List of emotiocn per tweet
    """
    new_arr = []
    
    for item in arr:
        new_arr.append(emoji.demojize(item))
        
    return new_arr
    
    

def _handle_char(list):
    
    """
    takes a list of words represent a tweet and return words removed of repeated characters and irregular short words
    Arguments:
    arr: Series. array of tweets.
    
    Returns:
    a list of words removed of repeated characters ( e.g احببب: احب) and irregular short words
    """
    arr = [re.sub(r'(.)\1+', r'\1', word) for word in list]
    arr = [word for word in list if len(word) >= 3]
    return arr

# def remove_spam(arr):
    
#     """
#     takes a array of tweets and remove spam
#     Arguments:
#     arr: Series. array of tweets.
    
#     Returns:
#     a Series of tweets removed of spam tweets
#     """
    
    
#     return arr

def df_to_pdf(df, filename):
    """
    (Required dependencies: https://pypi.org/project/pdfkit/)
    
    takes a dataframe and create a pdf page 
    Arguments:
    df: Pandas Dataframe. containes data to map to pdf page.
    filename: String. filename of a pdf page 
    
    Returns:
    None
    """
    
    try:
        import pdfkit as pdf
        import os
        
        html = df.to_html()
        
        with open(filename+'.html', "w", encoding="utf-8") as file:
            
            file.writelines('<meta charset="UTF-8">\n')
            file.write(html)
            
        pdf.from_file(filename+'.html', filename+'.pdf')
        
        os.remove(filename+'.html')
        
    except:
        pass