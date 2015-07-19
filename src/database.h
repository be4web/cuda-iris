#include <stdio.h>
#include <bson.h>
#include <mongoc.h>
#include <math.h>
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
void insert_data_database(mongoc_collection_t *collection, float *data_vector, char *subkey);

/***
 * Search for feature vector in the whole database
 * Arguments: Collection, searchstring
 * 
 ***/
char *search_vector_database(mongoc_collection_t *collection, float *data_vector);

/***
 * calculates the distances between the source and destination iris feature vector
 * Arguments: float array1, float array2
 * 
 ***/
float distance_calculation(float *array1,float *array2);
