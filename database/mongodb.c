#include <stdio.h>
#include <bson.h>
#include <mongoc.h>

#include "database.h"

int main(int argc,char *argv[]) 
{
	/************** begin of dummy data *****************/
	float hector_vector[32];	/*sis is hector, hi wonts tu be fieded intu de dätabes*/
	
	short i=0;
	while(i<32)	/* hector vector fill */
	{	
		hector_vector[i]=0.123456;
		i++;
	}
	char *datastring;
	datastring = malloc(256);
	memset(datastring,'\0',256);
	short k=0;
    while(k<32)										/* process all 32 floating point variables */
    {
		char *number=malloc(8);						/* allocate memory */
		sprintf(number,"%-.6f",hector_vector[k]);	/* print the float as string */
		(void)strncat(datastring,number,8);			/* concatenate to one big datastring */
		k++;
	}
	
	printf("BEGIN:%s\n",datastring);
	float feature_vector[32];
	short n=0;
	while(n<32)
	{
		char *offset = datastring+8;
		feature_vector[n] = strtof(datastring,&(offset));
		datastring+=8;
		n++;
	}
	/************** end of dummy data *****************/
    mongoc_client_t *client;
    mongoc_collection_t *collection;
    bson_error_t error;
    bson_oid_t oid;
    bson_t *doc;
    /* search query */
    bson_t *query;
    /* search string */
    char *str;
	/*set up the environment*/
    mongoc_init ();
    client = mongoc_client_new ("mongodb://localhost:27017/");
    collection = mongoc_client_get_collection (client, "Iris", "Iris");

	/* mongodb insert */	
//	insert_data_database(collection, hector_vector, "Iris");
  /*mongodb search like*/
    printf("Result: %s\n",search_vector_database(collection, hector_vector));
    

	printf("******END SEARCH********");    
    
    /* mongodb delete*/
/*    doc = bson_new ();
    bson_oid_init (&oid, NULL);
    BSON_APPEND_OID (doc, "_id", &oid);
        
    if (!mongoc_collection_remove (collection, MONGOC_DELETE_SINGLE_REMOVE, doc, NULL, &error)) {
        printf ("Delete error %s\n", error.message);
    }else{
		printf("Delete success\n");
	}
    bson_destroy(doc);*/
    /* end */
    mongoc_collection_destroy (collection);
    mongoc_client_destroy (client);
  return 0;
}

