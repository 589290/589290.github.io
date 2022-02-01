#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# PROGRAMMER: Greg
# PURPOSE: Check images & report results: read them in, predict their content (classifier), compare prediction to actual value labels and output results
#
# Use argparse Expected Call with <> indicating expected user input: python check_images.py --dir <directory with images> --arch <model> --dogfile <file that contains dognames>
#   Example call: python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt


# Imports python modules
import argparse
import time
from os import listdir
from workspace_utils import keep_awake # keep awake
from classifier import classifier # Imports classifier function for using CNN to classify images 
from print_functions_for_lab_checks import * # Imports print functions that check the lab

# if keep awake needed
# for i in keep_awake(range(5)):
# do iteration with lots of work here

# Main program function defined below
def main():
    start_time = time.time()
    in_arg = get_input_args()
    #print("{} {} {}".format(in_arg.dir, in_arg.arch, in_arg.dogfile))
    answers_dic = get_pet_labels(in_arg.dir)
    #print(answers_dic)
    results_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    #check_classifying_images(results_dic)
    #print(results_dic)
    adjust_results4_isadog(results_dic, in_arg.dogfile)
    #check_classifying_labels_as_dogs(results_dic)
    #print(results_dic)
    results_stats = calculates_results_stats(results_dic)
    #check_calculating_results(results_dic, results_stats)
    #print(results_dic)
    print_results(results_dic, results_stats, in_arg.arch, True, True)
    # calc display times
    end_time = time.time()
    tot_time = end_time - start_time
    print("\nTotal Runtime: {} seconds".format(tot_time))


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using the argparse module. This function returns these arguments as an ArgumentParser object.  3 command line arguements are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default 'dognames.txt' 
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='Path to the pet image files; default: pet_images/', required=False, default='pet_images/')
    parser.add_argument('--arch', type=str, help='CNN model architecture to use for image classification (vgg, alexnet, resnet); default: vgg', required=False, default='vgg')
    parser.add_argument('--dogfile', type=str, help='Text file that contains all labels associated to dogs; default: dognames.txt', required=False, default='dognames.txt')
    return(parser.parse_args())


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image files. Reads in pet filenames and extracts the pet image labels from the  filenames and returns these label as petlabel_dic.
    This is used to check the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image Labels (as value)  
    """
    petlabel_dic = {} # dictionary
    # Retrieve the filenames from folder pet_images/
    filename_list = listdir(image_dir)
    for idx in range(0, len(filename_list), 1):
        # Sets string to lower case letters
        low_pet_image = filename_list[idx].lower()
        # Splits lower case string by _ to break into words 
        word_list_pet_image = low_pet_image.split("_")
        # Create pet_name starting as empty string
        pet_name = ""
        # Loops to check if word in pet name is only alphabetic characters - if true append word to pet_name separated by trailing space 
        for word in word_list_pet_image:
            if word.isalpha():
                pet_name += word + " "
        # Strip off starting/trailing whitespace characters 
        pet_name = pet_name.strip()
        # put in dict
        if filename_list[idx] not in petlabel_dic:
            petlabel_dic[filename_list[idx]] = pet_name
    return(petlabel_dic)


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and  creates a dictionary containing both labels and comparison of them to be returned.
     PLEASE NOTE: This function uses the classifier() function defined in classifier.py within this function. The proper use of this function is in test_classifier.py
     Please refer to this program prior to using the  classifier() function to classify images in this function. 
     Parameters: 
      images_dir - The (full) path to the folder of images that are to be classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels that classify what's in the image, where its' key is the pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label 
      model - pretrained CNN whose architecture is indicated by this parameter, values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and classifer labels and 0 = no match between labels
    """
    results_dic = {}
    # Process all files in the petlabels_dic - use images_dir to give fullpath
    for key in petlabel_dic:
       # Runs classifier function to classify the images classifier function 
       model_label = classifier(images_dir+key, model)
       # Processes the results so they can be compared with pet image labels
       model_label = model_label.lower()
       model_label = model_label.strip()
       # Put separate terms that 'may' compose the classifier label into a list so that each term is an item in the list.
       model_label_list = model_label.split(", ")
       # defines truth as pet image label 
       truth = petlabel_dic[key]
       # If the pet image label is found within the classifier label list of terms as an exact match to one of the terms in the list then they are added to results_dic as an exact match
       if truth in model_label_list:
           results_dic[key] = [truth, model_label, 1]
       # For those that aren't an exact term match to a term - checks if the pet_label is part of the term like: "poodle" matching to "standard poodle" OR "cat" matching to "tabby cat" 
       else:
           # Sets found to False - IF pet image label is FOUND as part of a term within the list of classifier label terms then will be set to True
           found = False
           # For loop to iterate through each term from model_label_list
           for term in model_label_list:
               # splits the term into a word list using split()
               word_list = term.split(' ')
               # if the pet image label hasn't been found AND it exists in the word list like 'poodle' in ['standard', 'poodle'] or 'cat' in ['tabby', 'cat'] then found!
               if (not found) and truth in word_list:
                   found = True
                   results_dic[key] = [truth, model_label, 1]
                   break 
           # If pet image label isn't found within the terms that exist in the list of labels the classifier function produces then set match = 0 (not a match)
           if not found:
               results_dic[key] = [truth, model_label, 0]
    return(results_dic)


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly classified images 'as a dog' or 'not a dog' especially when not a match. 
    Demonstrates if model architecture correctly classifies dog images even if it gets dog breed wrong (not a match). 
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 'as-a' dog and 0 = Classifier classifies image 'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet 1000 labels (used by classifier model) and dog names from the pet image files. This file has one dog name per line dog names are all
                in lowercase with spaces separating the distinct words of the dogname. This file should have been passed in as a command line argument. (string - indicates text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """           
    # load dog names
    dognames_dic = {} # dict()
    with open(dogsfile) as f:
        for line in f:
            rline = line.rstrip()
            if rline in dognames_dic:
                print('warning: {} is already in the dognames_dic'.format(rline))
            else:
                dognames_dic[rline] = 1
    # loop for comparisons
    for key in results_dic:
        #print('{} {}'.format(results_dic[key][0], results_dic[key][1]))
        if results_dic[key][0] in dognames_dic:
            idx3 = 1
        else:
            idx3 = 0
        if results_dic[key][1] in dognames_dic:
            idx4 = 1
        else:
            idx4 = 0
        # update results_dic
        results_dic[key].extend((idx3, idx4))
    return


def calculates_results_stats(results_dic):
    """
    Calculates statistics of results of the run using classifier's model architecture on classifying images. Then puts the results statistics in a dictionary so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that the statistics calculated as the results are either percentages or counts. 
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 'as-a' dog and 0 = Classifier classifies image 'as-NOT-a' dog.
    Returns:
     results_stats - Dict that contains the results statistics (percentage or count) where the key is the statistic's name (starting with 'pct' for percentage or 'n' for count) and the value is the statistic's value 
    """
    results_stats = {}
    results_stats['n_dogs_img'] = 0
    results_stats['n_match'] = 0
    results_stats['n_correct_dogs'] = 0
    results_stats['n_correct_notdogs'] = 0
    results_stats['n_correct_breed'] = 0       
    # process through the results dictionary
    for key in results_dic:
        # Labels Match Exactly
        if results_dic[key][2] == 1:
            results_stats['n_match'] += 1
        # Pet Image Label is a Dog AND Labels match- counts Correct Breed
        if sum(results_dic[key][2:]) == 3:
                results_stats['n_correct_breed'] += 1
        # Pet Image Label is a Dog - counts number of dog images
        if results_dic[key][3] == 1:
            results_stats['n_dogs_img'] += 1
            # Classifier classifies image as Dog (& pet image is a dog) // counts number of correct dog classifications
            if results_dic[key][4] == 1:
                results_stats['n_correct_dogs'] += 1
        # Pet Image Label is NOT a Dog
        else:
            # Classifier classifies image as NOT a Dog(& pet image isn't a dog) counts number of correct NOT dog clasifications.
            if results_dic[key][4] == 0:
                results_stats['n_correct_notdogs'] += 1
    # Calculates run statistics (counts & percentages) below that are calculated using counters from above.
    # calculates number of total images
    results_stats['n_images'] = len(results_dic)
    # calculates number of not-a-dog images using - images & dog images counts
    results_stats['n_notdogs_img'] = (results_stats['n_images'] - results_stats['n_dogs_img']) 
    # Calculates % correct for matches
    results_stats['pct_match'] = (results_stats['n_match'] / results_stats['n_images'])*100.0
    # Calculates % correct dogs
    results_stats['pct_correct_dogs'] = (results_stats['n_correct_dogs'] / results_stats['n_dogs_img'])*100.0    
    # Calculates % correct breed of dog
    results_stats['pct_correct_breed'] = (results_stats['n_correct_breed'] / results_stats['n_dogs_img'])*100.0
    # Calculates % correct not-a-dog images
    # Uses conditional statement for when no 'not a dog' images were submitted 
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pct_correct_notdogs'] = (results_stats['n_correct_notdogs'] / results_stats['n_notdogs_img'])*100.0
    else:
        results_stats['pct_correct_notdogs'] = 0.0
    # returns results_stast dictionary 
    return results_stats


def print_results(results_dic, results_stats, model, print_incorrect_dogs = False, print_incorrect_breed = False):
    """
    Prints summary results on the classification and then prints incorrectly classified dogs and incorrectly classified dog breeds if user indicates they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List 
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and 0 = pet Image 'is-NOT-a' dog. 
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image 'as-a' dog and 0 = Classifier classifies image 'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a percentage or a count) where the key is the statistic's name (starting with 'pct' for percentage or 'n' for count) and the value is the statistic's value 
      model - pretrained CNN whose architecture is indicated by this parameter, values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and False doesn't print anything(default) (bool)  
      print_incorrect_breed - True prints incorrectly classified dog breeds and False doesn't print anything(default) (bool) 
    Returns:
           None - simply printing results.
    """    
    # Prints summary statistics over the run
    print("\n\n*** Results Summary for CNN Model Architecture",model.upper(), "***")
    print("%20s: %3d" % ('N Images', results_stats['n_images']))
    print("%20s: %3d" % ('N Dog Images', results_stats['n_dogs_img']))
    print("%20s: %3d" % ('N Not-Dog Images', results_stats['n_notdogs_img']))
    # Prints summary statistics (percentages) on Model Run
    print(" ")
    for key in results_stats:
        if key[0] == "p":
            print("%20s: %5.1f" % (key, results_stats[key]))
    # IF print_incorrect_dogs == True AND there were images incorrectly classified as dogs or vice versa - print out these cases
    if (print_incorrect_dogs and ((results_stats['n_correct_dogs'] + results_stats['n_correct_notdogs']) != results_stats['n_images'])):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        # process through results dict, printing incorrectly classified dogs
        for key in results_dic:
            # Pet Image Label is a Dog - Classified as NOT-A-DOG -OR- Pet Image Label is NOT-a-Dog - Classified as a-DOG
            if sum(results_dic[key][3:]) == 1:
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0], results_dic[key][1]))
    # IF print_incorrect_breed == True AND there were dogs whose breeds were incorrectly classified - print out these cases                    
    if (print_incorrect_breed and (results_stats['n_correct_dogs'] != results_stats['n_correct_breed'])):
        print("\nINCORRECT Dog Breed Assignment:")
        # process through results dict, printing incorrectly classified breeds
        for key in results_dic:
            # Pet Image Label is-a-Dog, classified as-a-dog but is WRONG breed
            if (sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0):
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0], results_dic[key][1]))
                
                
# Call to main function to run the program
if __name__ == "__main__":
    main()
