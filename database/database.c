#include "database.h"
/***
 * Execute a arbitary database command
 * Arguments: Client, Collection, Command String and Argument for the Command
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
 ***/
void exec_client_command_database(mongoc_client_t *client,char *database_name, char *command_string, char *argument)
{
    bson_error_t error;
    bson_t *command;
    bson_t reply;
    char *str;                                     /* reply string */
    
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
 * Arguments: Collection, feature data, subkey(Iris in this case)
 ***/
void insert_data_database(mongoc_collection_t *collection, float *data_vector, char *subkey)
{
    bson_oid_t oid;
	bson_t *doc;
    bson_error_t error;
    bson_t *command;
    bson_t reply;
    char *str;                                      /* reply string */
    char *datastring = malloc(256);
    memset(datastring,'\0',256);
	
    for(int k=0;k<32;k++)                           /* process all 32 floating point variables */
    {
		char *number=malloc(8);                     /* allocate memory */
		sprintf(number,"%-.6f",data_vector[k]);     /* print the float as string */
		(void)strncat(datastring,number,8);         /* concatenate to one big datastring */
	}
	
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
 * Search for feature vector in the whole database
 * The matching works by distance calculation which uses the given threshold
 * Arguments: Collection, feature data, threshold
 ***/
char *search_vector_database(mongoc_collection_t *collection, float *data_vector, float threshold)
{
	const bson_t *fixdoc;
    char *iris_data = malloc(257);  
    char *str;                                           /* actual data */  
    bson_t *query = bson_new();
    mongoc_cursor_t *cursor = mongoc_collection_find(collection, MONGOC_QUERY_NONE,0,0,0,query,NULL,NULL);
    bool found=false;
    
    while(mongoc_cursor_next(cursor, &fixdoc)){          // iterate over all objects in the database
		str = bson_as_json(fixdoc,NULL);                 // convet object to string
		memcpy(iris_data,&str[61],256);
		iris_data[256] = '\0';
//		printf("%s\n",iris_data);                        //DEBUG

		char *iris_pointer = iris_data;

		float feature_vector[32];
		for(int n=0;n<32;n++)
		{
			char *offset = iris_pointer+8;
			feature_vector[n] = strtof(iris_pointer,&(offset));
			iris_pointer+=8;
		}
		/*comparison, 0.07 (7%) is the usual threshold value*/	
		if(distance_calculation(data_vector,feature_vector) < threshold)
		{
//			printf("Hurra scheissgeil\n");               //DEBUG
			found = true;
		}else{
//			printf("Ned so geil...\n");                  //DEBUG
		}	
		bson_free(str);
	}
	bson_destroy(query);
	mongoc_cursor_destroy(cursor);
	if(found){
		return("match");
	}else{
		return("no match");
	}
	/* should not be reached */
	return "No match!";
}

/***
 * calculates the distances between the source and destination iris feature vector
 * Arguments: float array1, float array2
 ***/
float distance_calculation(float *array1,float *array2)
{
    float dist, diff;
	for(int i=0;i<32;i++) 
	{
        diff = array1[i] - array2[i];
        dist += diff * diff;
    }
	return sqrt(dist);
}
