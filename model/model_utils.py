import numpy as np
import numpy.typing as npt

from typing import Tuple, List, Set


def bag_of_words_matrix(sentences: List[str]) -> npt.ArrayLike:
    """
    Convert the dataset into V x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    # CREATING VOCABULARY
    mega_document = []
    for sentence in sentences:
        for word in sentence.split():
            mega_document.append(word)

    # SORTING MEGA DOCUMENT FOR COUNT
    mega_document = np.sort(np.array(mega_document))

    # CREATING A UNIQUE LIST OF WORDS TO FORM VOCABULARY (V) AND THEIR COUNTS (count_words)
    V = [mega_document[0]]
    count_words = {}
    count = 1
    for i in range(1, len(mega_document)):
        if mega_document[i] == mega_document[i-1]:
            count += 1
        else:
            count_words[mega_document[i-1]] = count
            V.append(mega_document[i])
            count = 1
    count_words[mega_document[-1]] = count

    # STORING THE ORIGINAL VOCABULARY (WITHOUT UNK TOKEN)
    backup_V = V

    # REPLACING RARE OCCURRING WORDS WITH <UNK> TOKEN AND ADDING THEIR COUNT
    unk_count = 0
    remove_keys = []
    for key in count_words.keys():
        if count_words[key] < 2:
            unk_count += count_words[key]
            remove_keys.append(key)

    count_words["<UNK>"] = unk_count
    V.append("<UNK>")

    # REMOVING RARE OCCURRING WORDS FROM DICTIONARY AND VOCABULARY
    for k in remove_keys:
        count_words.pop(k, None)
        index_temp = V.index(k)
        V.pop(index_temp)

    # INITIALIZING BAG OF WORDS MATRIX WITH ZEROS
    X = np.zeros((len(V), len(sentences)))

    # CREATING X MATRIX OF SIZE VxM
    for i in range(len(sentences)):
        for word in sentences[i].split():
            # CHECK FOR RARE WORDS OCCURRING IN TRAINING DATA
            break_flag = 0
            for rare in remove_keys:
                if word == rare:
                    X[-1][i] = 1
                    break_flag = 1
                    break

            if break_flag == 1:
                break

            ind = V.index(word)
            X[ind][i] = 1

    return X
    #########################################################################


def labels_matrix(data: Tuple[List[str], Set[str]]) -> npt.ArrayLike:
    """
    Convert the dataset into K x M matrix.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    intent_list = data[0]
    unique = data[1]
    unique = list(unique)

    Y = np.zeros((len(unique), len(intent_list)))

    # CREATING Y MATRIX OF SIZE KxM
    for i in range(len(intent_list)):
        for check in unique:
            if intent_list[i] == check:
                index_uni = unique.index(check)
                Y[index_uni][i] = 1

    return Y
    #########################################################################


def softmax(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Softmax function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    Z1 = np.exp(z)
    summation = np.sum(Z1, axis=0)

    for j in range(len(Z1[0])):
        for i in range(len(Z1)):
            Z1[i][j] = Z1[i][j] / summation[j]

    return Z1
    #########################################################################


def relu(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    Rectified Linear Unit function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    Z2 = z
    for i in range(len(Z2)):
        for j in range(len(Z2[0])):
            if Z2[i][j] < 0:
                Z2[i][j] = 0

    return Z2
    #########################################################################


def relu_prime(z: npt.ArrayLike) -> npt.ArrayLike:
    """
    First derivative of ReLU function.
    """
    ############################# STUDENT SOLUTION ##########################
    # YOUR CODE HERE
    Z3 = z
    for i in range(len(Z3)):
        for j in range(len(Z3[0])):
            if Z3[i][j] < 0:
                Z3[i][j] = 0
            else:
                Z3[i][j] = 1

    return Z3
    #########################################################################