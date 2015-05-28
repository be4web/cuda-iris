#include "database.h"
/***
 * Execute a arbitary database command
 * Arguments: Collection, Command String and Argument for the Command
 * 
 ***/
void exec_command_database(mongoc_collection_t *collection, char *command_string, char *argument)
{
    bson_error_t error;
    bson_t *command;
    bson_t reply;
    char *str;                /* reply string */
    
    command = BCON_NEW (command_string, BCON_UTF8 (argument));
    if (mongoc_collection_command_simple (collection, command, NULL, &reply, &error)) {
        str = bson_as_json (&reply, NULL);
        printf ("%s\n", str);
        bson_free (str);
    } else {
        fprintf (stderr, "Failed to run command: %s\n", error.message);
    }
}

/***
 * Insert database entries
 * Arguments: Collection, Datastring and subkey
 * 
 ***/
void insert_data_database(mongoc_collection_t *collection, char *datastring, char *subkey)
{
    bson_oid_t oid;
	bson_t *doc;
    bson_error_t error;
    bson_t *command;
    bson_t reply;
    char *str;                /* reply string */
    
    doc = bson_new();
    bson_oid_init (&oid, NULL);
    BSON_APPEND_OID (doc, "ID", &oid);    
    BSON_APPEND_UTF8 (doc, subkey, datastring);

    if (!mongoc_collection_insert (collection, MONGOC_INSERT_NONE, doc, NULL, &error)) {
        printf ("Insert error %s\n", error.message);
    }else{
		printf("Insert success\n");
	}

    bson_destroy(doc);
}
