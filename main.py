from dataclasses import dataclass

import csv

import math

import re


@dataclass
class Song:
    id: int

    title: str

    year: int

    artist: str

    genre: str

    lyrics: list


"""



1.Term frequency is how often a word appears in an individual song. Inverse 

document frequency takes into account how often a word appears in all of the

songs in a corpus. The more a word appears in different songs, the lower

its idf value is, therefore the lower the importance of the word. If a word

appears in only one or a few songs, the higher its idf value is, meaning the

more important a word is. TFIDF is when each word in a song is given a value

by multiplying the number of times the words appears in the individual song (tf)

by the weighted importance of the word (idf). The step by step process for 

compute_tf and compute_idf are outlined in their respective functions below.





2. We can run the given training data and use it to test the songs

whose genre we already know. These should be ones that are obviously part

of that genre. An example being "Big Green Tractor" by Jason Aldean should be

labeled as a country song. We could create a songs made up of only key words.

We want to ensure that these test songs are labeled with their appropriate 

genre.





"""

bad_characters = re.compile(r"[^\w]")


def clean_word(word: str) -> str:
    """input: string

    output: string

    description: using the bad characters regular expression, this function

    strips out invalid characters

    """

    word = word.strip().lower()

    return bad_characters.sub("", word)


def clean_lyrics(lyrics: str) -> list:
    """input: string representing the lyrics for a song

    output: a list with each of the words for a song

    description: this function parses through all of the lyrics for a song and

    makes sure they contain valid characters

    """

    lyrics = lyrics.replace("\n", " ")

    return [clean_word(word) for word in lyrics.split(" ")]


def create_corpus(filename: str) -> list:
    """input: a filename

    output: a list of Songs

    description: this function is responsible for creating the collection of

    songs, including some data cleaning

    """

    with open(filename) as f:

        corpus = []

        iden = 0

        for s in csv.reader(f):

            if s[4] != "Not Available":
                new_song = Song(iden, s[1], s[2], s[3], s[4],

                                clean_lyrics(s[5]))

                corpus.append(new_song)

                iden += 1

        return corpus


def compute_idf(corpus: list) -> dict:
    """input: a list of Songs

    output: a dictionary from words to inverse document frequencies (as floats)

    description: this function is responsible for calculating inverse document

      frequencies of every word in the corpus



      Step-by-step:

      1. Create an empty hashtable.

      2. In this hashtable, the keys are words that appear in every song and the

      value is the amount of songs that word appears in.

      3. Iterate over the corpus and run a for-loop on each song lyrics.

      4. If a word is not in the hashtable, add it to the hashtable and set its

      value equal to 1. If the word is in the hashtable, add one to the value

      of that word. (We want to make sure that each word in each individual song

      is counted once even if it appears in the songs multiple times.)

      5. create a new hashtable. (The key in hashtable is a word and the value

      is the idf value of those words.)

      6. Take the natural log of the length of the corpus over individual words

      ni value. (from the first hashtable.) This can be accomplished through

      the use of a for-loop.



    """

    df = {}

    for song in corpus:

        for word in song.lyrics:

            if word in df:

                df[word].add(song.id)

            else:

                df[word] = set()

                df[word].add(song.id)

    N = len(corpus)

    idf = {}

    for word in df:
        idf[word] = math.log(N / len(df[word]))

    return idf


def compute_tf(song_lyrics: list) -> dict:
    """input: list representing the song lyrics

    output: dictionary containing the term frequency for that set of lyrics

    description: this function calculates the term frequency for a set of lyrics

    """

    tf = {}

    for word in song_lyrics:

        if word in tf:

            tf[word] += 1

        else:

            tf[word] = 1

    return tf


def compute_tf_idf(song_lyrics: list, corpus_idf: dict) -> dict:
    """input: a list representing the song lyrics and an inverse document

    frequency dictionary output: a dictionary with tf-idf weights for the song

    (words to weights) description: this function calculates the tf-idf weights

    for a song

    """

    tf_idf = {}

    tf = compute_tf(song_lyrics)

    for word in tf:
        tf_idf[word] = tf[word] * corpus_idf[word]

    return tf_idf


def compute_corpus_tf_idf(corpus: list, corpus_idf: dict) -> dict:
    """input: a list of songs and an idf dictionary

    output: a dictionary from song ids to tf-idf dictionaries

    description: calculates tf-idf weights for an entire corpus

    """

    corpus_tf_idf = {}

    for song in corpus:
        corpus_tf_idf[song.id] = compute_tf_idf(song.lyrics, corpus_idf)

    return corpus_tf_idf


def cosine_similarity(l1: dict, l2: dict) -> float:
    """input: dictionary containing the term frequency - inverse document

     frequency for a song, dictionary containing the term frequency - inverse

    document frequency for a song output: float representing the similarity

    between the values of the two dictionaries description: this function finds

    the similarity score between two dictionaries

    """

    magnitude1 = math.sqrt(sum(w * w for w in l1.values()))

    magnitude2 = math.sqrt(sum(w * w for w in l2.values()))

    dot = sum(l1[w] * l2.get(w, 0) for w in l1)

    return dot / (magnitude1 * magnitude2)


def nearest_neighbor(

        song_lyrics: str, corpus: list, corpus_tf_idf: dict, corpus_idf: dict

) -> Song:
    """input: a string representing the lyrics for a song, a list of songs,

      tf-idf weights for every song in the corpus, and idf weights for every

    word in the corpus

    output: a song object

    description: this function produces the song in the corpus that is most

    similar to the lyrics it is given

    """

    input_tf_idf = compute_tf_idf(song_lyrics, corpus_idf)

    # corpus_tf_idf2 = compute_corpus_tf_idf()


def main(filename: str, lyrics: str):
    corpus = create_corpus(filename)

    corpus_idf = compute_idf(corpus)

    corpus_tf_idf = compute_corpus_tf_idf(corpus, corpus_idf)

    print(nearest_neighbor(lyrics, corpus, corpus_tf_idf, corpus_idf).genre)



