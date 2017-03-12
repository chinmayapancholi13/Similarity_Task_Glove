
# coding: utf-8

import gzip
import os

import numpy as np
import scipy.spatial.distance as sp_dist
import random
import math
import tensorflow as tf

from sklearn.cross_validation import KFold

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

import scipy.special as ss

simInputFile = "./word-similarity-dataset"
vectorTxtFile = "./glove.6B.300d.txt"
simOutputFile = "./simOutput.csv"
simSummaryFile = "./simSummary.csv"
simDataset = [item.split(" | ") for item in open(simInputFile).read().splitlines()]

def vectorExtract(simD = simDataset, vect = vectorTxtFile):
    simList = [stuff for item in simD for stuff in item]
    wordList = set(simList)
    print(len(wordList))
    wordDict = dict()

    vectorFile = open(vect, 'r')
    for line in vectorFile:
        if line.split()[0].strip() in wordList:
            wordDict[line.split()[0].strip()] = line.split()[1:]

    vectorFile.close()

    print('retrieved', len(wordDict.keys()))
    return wordDict

validateVectors = vectorExtract()

def similarityTask(inputDS = simDataset, outputFile = simOutputFile, summaryFile=simSummaryFile, vectors=validateVectors):
    mrr_c = 0
    mrr_e = 0
    mrr_m = 0

    num_correct_c = 0
    num_correct_e = 0
    num_correct_m = 0

    num_eval_c = 0
    num_eval_e = 0
    num_eval_m = 0

    output_file_fp1 = open(outputFile, 'w')

    for i in range(len(inputDS)):
        input_word = inputDS[i][0]
        solution_word = inputDS[i][1]
        output_word_1 = inputDS[i][2]
        output_word_2 = inputDS[i][3]
        output_word_3 = inputDS[i][4]
        # print(input_word,output_word_1,output_word_2,output_word_3,output_word_4)

        if (input_word not in vectors) or (solution_word not in vectors) or (output_word_1 not in vectors) or (output_word_2 not in vectors) or (output_word_3 not in vectors) :
            continue

        num_eval_c = num_eval_c + 1
        num_eval_e = num_eval_e + 1
        num_eval_m = num_eval_m + 1

        #get vectors for the words
        input_word_vec = vectors[input_word]
        solution_word_vec = vectors[solution_word]
        output_word_1_vec = vectors[output_word_1]
        output_word_2_vec = vectors[output_word_2]
        output_word_3_vec = vectors[output_word_3]

        input_word_vec = np.array(input_word_vec, dtype=float)
        solution_word_vec = np.array(solution_word_vec, dtype=float)
        output_word_1_vec = np.array(output_word_1_vec, dtype=float)
        output_word_2_vec = np.array(output_word_2_vec, dtype=float)
        output_word_3_vec = np.array(output_word_3_vec, dtype=float)

        #cosine distance
        solution_word_score = cosine_similarity(input_word_vec, solution_word_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(solution_word) + str(",") + "C"+str(",") + str(solution_word_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_1_score = cosine_similarity(input_word_vec, output_word_1_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_1) + str(",") + "C"+str(",") + str(output_word_1_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_2_score = cosine_similarity(input_word_vec, output_word_2_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_2) + str(",") + "C"+str(",") + str(output_word_2_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_3_score = cosine_similarity(input_word_vec, output_word_3_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_3) + str(",") + "C"+str(",") + str(output_word_3_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        # print (solution_word_score, output_word_3_score,output_word_3_score,output_word_3_score)

        rank_sol = 1
        if output_word_1_score > solution_word_score:
            rank_sol = rank_sol + 1
        if output_word_2_score > solution_word_score:
            rank_sol = rank_sol + 1
        if output_word_3_score > solution_word_score:
            rank_sol = rank_sol + 1

        if rank_sol == 1 :
            num_correct_c  = num_correct_c + 1

        mrr_c = mrr_c + (1.0/rank_sol)

        #euclidean distance
        solution_word_score = euclidean_distances(input_word_vec, solution_word_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(solution_word) + str(",") + "E"+str(",") + str(solution_word_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_1_score = euclidean_distances(input_word_vec, output_word_1_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_1) + str(",") + "E"+str(",") + str(output_word_1_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_2_score = euclidean_distances(input_word_vec, output_word_2_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_2) + str(",") + "E"+str(",") + str(output_word_2_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_3_score = euclidean_distances(input_word_vec, output_word_3_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_3) + str(",") + "E"+str(",") + str(output_word_3_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        # print (solution_word_score, output_word_3_score,output_word_3_score,output_word_3_score)

        rank_sol = 1
        if output_word_1_score < solution_word_score:
            rank_sol = rank_sol + 1
        if output_word_2_score < solution_word_score:
            rank_sol = rank_sol + 1
        if output_word_3_score < solution_word_score:
            rank_sol = rank_sol + 1

        if rank_sol == 1:
            num_correct_e  = num_correct_e + 1

        mrr_e = mrr_e + (1.0/rank_sol)

        #manhattan distance
        solution_word_score = manhattan_distances(input_word_vec, solution_word_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(solution_word) + str(",") + "M"+str(",") + str(solution_word_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_1_score = manhattan_distances(input_word_vec, output_word_1_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_1) + str(",") + "M"+str(",") + str(output_word_1_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_2_score = manhattan_distances(input_word_vec, output_word_2_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_2) + str(",") + "M"+str(",") + str(output_word_2_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        output_word_3_score = manhattan_distances(input_word_vec, output_word_3_vec)
        str_c = str(i+1) + str(",") + str(input_word) + str(",") + str(output_word_3) + str(",") + "M"+str(",") + str(output_word_3_score[0][0]) + "\n"
        output_file_fp1.write(str_c)

        # print (solution_word_score, output_word_3_score,output_word_3_score,output_word_3_score)

        rank_sol = 1
        if output_word_1_score < solution_word_score:
            rank_sol = rank_sol + 1
        if output_word_2_score < solution_word_score:
            rank_sol = rank_sol + 1
        if output_word_3_score < solution_word_score:
            rank_sol = rank_sol + 1

        if rank_sol == 1:
            num_correct_m  = num_correct_m + 1

        mrr_m = mrr_m + (1.0/rank_sol)

    output_file_fp1.close()

    if num_eval_c != 0:
        mrr_c = mrr_c / num_eval_c
    print (num_eval_c, num_correct_c, mrr_c)

    if num_eval_e != 0:
        mrr_e = mrr_e / num_eval_e
    print (num_eval_e, num_correct_e, mrr_e)

    if num_eval_m != 0:
        mrr_m = mrr_m / num_eval_m
    print (num_eval_m, num_correct_m, mrr_m)

    output_file_fp = open(summaryFile, 'w')
    str_c = 'C' + "," + str(num_correct_c)+","+str(num_eval_c)+","+str(mrr_c)+"\n"
    output_file_fp.write(str_c)
    str_e = 'E' + "," + str(num_correct_e)+","+str(num_eval_e)+","+str(mrr_e)+"\n"
    output_file_fp.write(str_e)
    str_m = 'M' + "," + str(num_correct_m)+","+str(num_eval_m)+","+str(mrr_m)+"\n"
    output_file_fp.write(str_m)
    output_file_fp.close()

def main():
    similarityTask()

if __name__ == '__main__':
    main()
