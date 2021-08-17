from song_analysis import *

song_test = Song(1, "love", 2000, "me", "pop", ['i', 'love', 'you',
                                                'you', 'love', 'me',
                                                'we', 'are', 'one', 'big',
                                                'family'])
song_test2 = Song(2, "test", 2012, 'you', 'rock', ['can', 'you', 'can',
                                                   'you', 'hear', 'me',
                                                   'say', 'when', 'i',
                                                   'am', 'moving', 'closer',
                                                   'you', 'keep', 'going',
                                                   'away'])
song_test3 = Song(3, "just the way you are", 2010, "bruno mars", 'pop',
                  ['oh', 'her', 'eyes', 'her', 'eyes',
                   'make', 'the', 'stars', 'look', 'like',
                   "theyre", 'not', 'shining', 'boots'])
song_test4 = Song(4, "same", 2020, "myself", 'country',
                  ['boots', 'boots', 'boots', 'boots', 'boots', 'boots'])
song_test5 = Song(5, 'word', 2020, 'again', 'empty', [])
song_test6 = Song(6, "short", 2000, "You", "Rock", ['you', 'and', 'me', 'boots',
                                                    'like'])
song_test_ = Song(7, "short", 2000, "You", "Rock", ['you', 'and', 'me'])


def test_compute_tf():
    assert compute_tf(song_test3.lyrics) == {'oh': 1, 'her': 2, "eyes": 2,
                                             'make': 1, 'the': 1, 'stars': 1,
                                             'look': 1,
                                             'like': 1, "theyre": 1, 'not': 1,
                                             'shining': 1, 'boots': 1}
    assert compute_tf(song_test4.lyrics) == {'boots': 6}
    assert compute_tf(song_test.lyrics) == {'i': 1, 'love': 2, 'you': 2,
                                            'me': 1,
                                            'we': 1, 'are': 1, 'one': 1,
                                            'big': 1,
                                            'family': 1}
    assert compute_tf(song_test2.lyrics) == {'can': 2, 'you': 3, 'hear': 1,
                                             'me': 1, 'say': 1,
                                             'when': 1, 'i': 1, 'am': 1,
                                             'moving': 1, 'closer': 1,
                                             'keep': 1, 'going': 1, 'away': 1}
    assert compute_tf(song_test5.lyrics) == {}


def test_compute_idf():
    song_test_list1 = [song_test, song_test_]
    song_test_list2 = [song_test4, song_test3, song_test6]

    assert compute_idf(song_test_list1) == {'i': math.log(2 / 1),
                                            'love': math.log(2),
                                            'you': 0,
                                            'me': 0,
                                            'we': math.log(2),
                                            'are': math.log(2),
                                            'one': math.log(2),
                                            'big': math.log(2),
                                            'family': math.log(2),
                                            'and': math.log(2)}
    assert compute_idf(song_test_list2) == {'oh': math.log(3 / 1),
                                            'her': math.log(3 / 1),
                                            "eyes": math.log(3 / 1),
                                            'make': math.log(3 / 1),
                                            'the': math.log(3 / 1),
                                            'stars': math.log(3 / 1),
                                            'look': math.log(3 / 1),
                                            'like': math.log(3 / 2),
                                            "theyre": math.log(3 / 1),
                                            'not': math.log(3 / 1),
                                            'shining': math.log(3 / 1),
                                            'boots': math.log(3 / 3),
                                            'you': math.log(3 / 1),
                                            'me': math.log(3 / 1),
                                            'and': math.log(3 / 1)}
    assert compute_idf([]) == {}


def test_compute_tf_idf():
    song_test_list = [song_test, song_test_]
    song_test_list2 = [song_test4, song_test3, song_test6]
    idf = compute_idf(song_test_list)
    idf2 = compute_idf(song_test_list2)

    assert compute_tf_idf(song_test.lyrics, idf) == {'i': 1 * math.log(2),
                                                     'love': 2 * math.log(2),
                                                     'you': 2 * 0,
                                                     'me': 0,
                                                     'we': math.log(2),
                                                     'are': math.log(2),
                                                     'one': math.log(2),
                                                     'big': math.log(2),
                                                     'family': math.log(2)}
    assert compute_tf_idf(song_test3.lyrics, idf2) == {
        'oh': 1 * math.log(3 / 1), 'her': 2 * math.log(3 / 1),
        "eyes": 2 * math.log(3 / 1), 'make': 1 * math.log(3 / 1),
        'the': 1 * math.log(3 / 1), 'stars': 1 * math.log(3 / 1),
        'look': 1 * math.log(3 / 1), 'like': 1 * math.log(3 / 2),
        "theyre": 1 * math.log(3 / 1), 'not': 1 * math.log(3 / 1),
        'shining': 1 * math.log(3 / 1), 'boots': 1 * math.log(3 / 3)}
    assert compute_tf_idf('', idf) == {}


def test_corpus_tf_idf():
    song_test_list = [song_test, song_test_]
    song_test_list2 = [song_test4, song_test3, song_test6]
    idf = compute_idf(song_test_list)
    idf2 = compute_idf(song_test_list2)
    dict = compute_corpus_tf_idf(song_test_list, idf)

    assert dict == {1: {'i': math.log(2), 'love': 2 * math.log(2), 'you': 2 * 0,
                        'me': 0, 'we': math.log(2), 'are': math.log(2),
                        'one': math.log(2), 'big': math.log(2),
                        'family': math.log(2)}, 7: {'you': 1 * 0,
                                                    'and': 1 * math.log(2),
                                                    'me': 1 * 0}}
    assert compute_corpus_tf_idf(song_test_list2, idf2) == {
        3: {'oh': 1 * math.log(3 / 1), 'her': 2 * math.log(3 / 1),
            "eyes": 2 * math.log(3 / 1), 'make': 1 * math.log(3 / 1),
            'the': 1 * math.log(3 / 1), 'stars': 1 * math.log(3 / 1),
            'look': 1 * math.log(3 / 1), 'like': 1 * math.log(3 / 2),
            "theyre": 1 * math.log(3 / 1), 'not': 1 * math.log(3 / 1),
            'shining': 1 * math.log(3 / 1), 'boots': 1 * math.log(3 / 3)},
        4: {'boots': 6 * math.log(3 / 3)},
        6: {'you': 1 * math.log(3 / 1), 'and': 1 * math.log(3 / 1),
            'me': 1 * math.log(3 / 1), 'boots': 1 * math.log(3 / 3),
            'like': 1 * math.log(3 / 2)}}
    assert compute_corpus_tf_idf([], idf) == {}


def test_nearest_neighbour():
    # make a song where all the lyrics are the same and seeing which song
    corpus_list = [song_test, song_test2, song_test3, song_test4, song_test6,
                   song_test_]
    corpus_idf = compute_idf(corpus_list)
    corpus_tf_idf = compute_corpus_tf_idf(corpus_list, corpus_idf)
    same_song3 = 'oh her eyes her eyes make the stars look like theyre not shining boots'
    assert nearest_neighbor(same_song3, corpus_list, corpus_tf_idf,
                            corpus_idf) == song_test3

    different_song = 'i like you me we are one big'
    assert nearest_neighbor(different_song, corpus_list, corpus_tf_idf,
                            corpus_idf) == song_test

    # testing using the small csv
    csv_corpus = create_corpus('small_songdata.csv')
    csv_corpus_idf = compute_idf(csv_corpus)
    csv_corpus_tf_idf = compute_corpus_tf_idf(csv_corpus, csv_corpus_idf)
    country_song = 'boots tractor bar love hat horse'

    assert nearest_neighbor(country_song, csv_corpus, csv_corpus_tf_idf,
                            csv_corpus_idf) == csv_corpus[39725]
