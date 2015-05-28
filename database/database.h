#include <stdio.h>
#include <bson.h>
#include <mongoc.h>
/***
 * Execute a arbitary database command
 * Arguments: Client, Collection, Command String and Argument for the Command
 * 
 ***/
void exec_command_database(mongoc_collection_t *collection, char *command_string, char *argument);

/***
 * Insert database entries
 * Arguments: Client, Collection, Command String and Argument for the Command
 * 
 ***/
void insert_data_database(mongoc_collection_t *collection, char *datastring, char *subkey);
