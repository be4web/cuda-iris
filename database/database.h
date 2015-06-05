#include <stdio.h>
#include <bson.h>
#include <mongoc.h>
/***
 * Execute a arbitary database command
 * Arguments: Client, Collection, Command String and Argument for the Command
 * 
 ***/
void exec_database_command_database(mongoc_collection_t *collection, char *command_string, char *argument);

/***
 * Execute a arbitary client command
 * Arguments: client, Database Name, Command String and Argument for the Command
 * 
 ***/
void exec_client_command_database(mongoc_client_t *client,char *database_name, char *command_string, char *argument);

/***
 * Insert database entries
 * Arguments: Client, Collection, Command String and Argument for the Command
 * 
 ***/
void insert_data_database(mongoc_collection_t *collection, char *datastring, char *subkey);

/***
 * Search for substrings in 256Byte Arrays
 * Arguments: Collection, searchstring
 * 
 ***/
char *search_substring_database(mongoc_collection_t *collection, char *search_string);

/***
 * Compare Hamming distances of 256Byte Arrays
 * Arguments: Array1, Array2, Threshold
 * Returns: True, False
 ***/
bool hamming_dist_match(char *array1, char *array2, int thres);
