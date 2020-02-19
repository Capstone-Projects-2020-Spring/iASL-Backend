#!/usr/bin/env python
#
# file: $iASL_SCRIPTS/params_tool.py
#
# revision history:
#  20200219 (TE): first version
#
# usage:
#  import as a class
#
# This script parses through a parameter file and uses
# a dictionary to define key/pair values
#------------------------------------------------------------------------------

# import system modules
#
import sys

#-----------------------------------------------------------------------------
#
# global variables are listed here
#
#-----------------------------------------------------------------------------

# define global variables
#
NEW_LINE = "\n"
SPACE = " "
DELIM_NULL = ""
DELIM_COMMENT = "#"
DELIM_KEY = "{"
DELIM_END_KEY = "}"
DELIM_EQUAL = "="

#------------------------------------------------------------------------------
#
# the ParamParser class is defined here
#
#------------------------------------------------------------------------------

# define the ParamParser class
#
class ParamParser:

    # function: init
    #
    # arguments: pfile - path to the parameter file
    #
    # return: none
    #
    # This is the constructor for the class
    #
    def __init__(self, pfile):
        self.pfile = pfile
        self.mapping = {}
        self.init_params()
    #
    # end of function
        

    # function: init_params
    #
    # arguments: none
    #
    # return: none
    #
    # This method parses through the file and tokenizes
    # appropriately to create a dictionary mapping
    #
    def init_params(self):

        # try to open the file
        #
        try:
            fp = open(self.pfile, "r")
        except IOError as e:
            print("[%s]: %s" % (sys.argv[0], e))
            exit(-1)

        # tokenize the file by new line
        #
        content = fp.read().split(NEW_LINE)

        # close the file
        #
        fp.close()

        # initially no section is found
        #
        key = None

        # for each line in the file
        #
        for line in content:

            # replace all spaces with null
            #
            line = line.replace(SPACE, DELIM_NULL)

            # if this line starts with a comment
            #
            if(line.startswith(DELIM_COMMENT)):
                continue

            # if the line is just space
            #
            elif(line == DELIM_NULL):
                continue

            # if we have not found a section
            #
            if key is None:

                # if the line is empty or the last char is not
                # the delimeter
                #
                if len(line) == 0 or not line[-1] == DELIM_KEY:
                    continue

                # grab the key
                #
                key = line[:-1]

                # append the key to the mapping
                #
                self.mapping[key] = {}

            # if we did find a section
            #
            else:

                # if this is the end of the section
                #
                if(line[-1] == DELIM_END_KEY):
                    key = None
                    continue

                # tokenize by the equal sign
                #
                tokenized = line.split(DELIM_EQUAL)

                # if we don't have two values...
                #
                if(len(tokenized) != 2):
                    continue

                # append the key/pair value to the section
                #
                self.mapping[key][tokenized[0]] = tokenized[1]
    #
    # end of function
    

    # function: __len__
    #
    # arguments: none
    #
    # return: int - length of the mapping
    #
    # This method returns the number of sections in the param file
    #
    def __len__(self):
        return len(self.mapping)
    #
    # end of function

    
    # function: __getitem__
    #
    # arguments: key - string representing name of section
    #
    # return: dict - dictionary of section of param file
    #
    # This method returns the section of the given param file
    #
    def __getitem__(self, key):
        if key not in self.mapping:
            raise KeyError('Invalid key for parameter file: %s' % (key))
        return self.mapping[key]
    #
    # end of function
#
# end of class

#
# end of file
