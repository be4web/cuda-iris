#include <stdio.h>
#include <bson.h>
#include <mongoc.h>

#include "database.h"

int main(int argc,char *argv[]) {
  
    mongoc_client_t *client;
    mongoc_collection_t *collection;
    mongoc_cursor_t *cursor;
    bson_error_t error;
    bson_oid_t oid;
    const bson_t *fixdoc;
    bson_t *doc;
    /* search query */
    bson_t *query;
    /* search string */
    char *str;
    
	/*set up the environment*/
    mongoc_init ();

    client = mongoc_client_new ("mongodb://localhost:27017/");
    collection = mongoc_client_get_collection (client, "test", "test");


	/* mongodb insert */
	
	insert_data_database( collection, "longstringthatshoulbesearchedFFFFFFFFFFFddajfnkejfnkjdsvnkjbnvkejbfkjbekfjbdskjvkj", "1234567890");
    
    /*mongodb search like*/
    
    query = bson_new();
    mongoc_collection_t *collection1;
    collection1 = mongoc_client_get_collection(client,"1234567890","dda");
    
    cursor = mongoc_collection_find(collection1, MONGOC_QUERY_NONE,0,0,0,query,NULL,NULL);
    
    /*move query cursor*/
    printf("Search all documents in database 'test'\n");
    while(mongoc_cursor_next(cursor, &fixdoc)){
		str = bson_as_json(fixdoc,NULL);
		printf("search string %s\n",str);
		bson_free(str);
	}
	bson_destroy(query);
	printf("******END SEARCH********");
    /* mongodb execute collStats command */
   
	exec_command_database(collection,"collStats","test");
    
    /* mongodb delete*/
    doc = bson_new ();
    bson_oid_init (&oid, NULL);
    BSON_APPEND_OID (doc, "ID", &oid);
        
    if (!mongoc_collection_remove (collection, MONGOC_DELETE_SINGLE_REMOVE, doc, NULL, &error)) {
        printf ("Delete error %s\n", error.message);
    }else{
		printf("Delete success\n");
	}
    
    bson_destroy(doc);
    /* end */
    
	mongoc_cursor_destroy(cursor);
    mongoc_collection_destroy (collection);
    mongoc_client_destroy (client);

  return 0;
}

