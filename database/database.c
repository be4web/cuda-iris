#include "database.h"
/***
 * Execute a arbitary database command
 * Arguments: Collection, Command String and Argument for the Command
 * 
 ***/
void exec_database_command_database(mongoc_collection_t *collection, char *command_string, char *argument)
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
 * Execute a arbitary client command
 * Arguments: client, Database Name, Command String and Argument for the Command
 * 
 ***/
void exec_client_command_database(mongoc_client_t *client,char *database_name, char *command_string, char *argument)
{
    bson_error_t error;
    bson_t *command;
    bson_t reply;
    char *str;                /* reply string */
    
    command = BCON_NEW (command_string, BCON_UTF8 (argument));
    if (mongoc_client_command_simple(client, database_name, command, NULL, &reply, &error)) {
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
    BSON_APPEND_OID (doc, "_id", &oid);    
    BSON_APPEND_UTF8 (doc, subkey, datastring);

    if (!mongoc_collection_insert (collection, MONGOC_INSERT_NONE, doc, NULL, &error)) {
        printf ("Insert error %s\n", error.message);
    }else{
		printf("Insert success\n");
	}

    bson_destroy(doc);
}

/***
 * Search for substrings in 256Byte Arrays
 * Arguments: Collection, searchstring
 * 
 ***/
char *search_substring_database(mongoc_collection_t *collection, char *search_string)
{
    const bson_t *fixdoc;
    char iris_data[257];  
    char *str;							/*actual string*/  
    bson_t *query = bson_new();
    mongoc_cursor_t *cursor = mongoc_collection_find(collection, MONGOC_QUERY_NONE,0,0,0,query,NULL,NULL);
    
    while(mongoc_cursor_next(cursor, &fixdoc)){
		str = bson_as_json(fixdoc,NULL);
		memcpy(iris_data,&str[61],256);
		iris_data[256] = '\0';
//		printf("%s\n",iris_data);
		if(strstr(iris_data,search_string))
		{
			printf("%s\n",iris_data);
			printf("Match found in object %s\n",str);
			bson_destroy(query);
			mongoc_cursor_destroy(cursor);
			return "No match!";	
		}
		//printf("%s",str);
		bson_free(str);
	}
	bson_destroy(query);
	mongoc_cursor_destroy(cursor);
	return "No match!";
}

/***
 * Compare Hamming distances of 256Byte Arrays
 * Arguments: Array1, Array2, Threshold
 * Returns: True, False
 ***/
bool hamming_dist_match(char *array1, char *array2,int thres)
{
    char *pos1 = array1, *pos2 = array2;
    uint8_t pattern1[256], pattern2[256];
    uint32_t *pat1 = (uint32_t *) pattern1, *pat2 = (uint32_t *) pattern2;
                         
    int hamming_dist = 0;
       
    while (*pos1)
    {
      sscanf(pos1, "%2hhx", &pattern1[(pos1-array1)>>1]);
      sscanf(pos2, "%2hhx", &pattern2[(pos2-array2)>>1]);
      pos1 += 2, pos2 += 2;
    }
 
    int i;
    for (i = 0; i < 256/sizeof(int); i++)  {
        int sum = 0,
        op = pat1[i] ^ pat2[i];
               
        __asm__ ("popcnt %1, %0"
                         : "=r" (sum)
                         : "r" (op)
                );
 
        hamming_dist += sum;
     }
       
     printf("Hamming distance: %d\n", hamming_dist);
     if(thres > hamming_dist)
     {
		return true;
	 }
	return false;
}
