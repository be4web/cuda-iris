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
    collection = mongoc_client_get_collection (client, "Iris", "Iris");

	/* mongodb insert */	
/*	insert_data_database(collection, "ergjheghkearljfqöwkölskdmljvkldfhnbgkjhkdfjbhewkjhfklajdshkjbfksjadbfhjbgdskajdfsanvkjbnnewjrnq3wbrhiefhioguhhdfsufsaeiu3984t59hrigjdfkgh4398ru32983nq3wbrhiefhioguhhdfsufsaeiu3984t59nq3wbrhiefhioguhhdfsufsaeiu3984t59z24894t59hrigjdfk4t59hrigjdfk7fhjbg38rEZIO BATOCCI", "Iris");
	insert_data_database(collection, "fsdnvhergjheghkearljfqöwkölskdmljvkldfhnbgkjhkdfjbhewkjhfklajdshkjbfksjadbfhjbgdskajdfsanvkjbnnewjrnq3wbrhiefhioguhhdfsufsaeiu3984t59hrigjdfkgh4398ru32983z248975z39832059834957934734tz87erzhgihsdbhbjagshvchcv32432hvfjewv3243jdfsnkjbmkmbknndsjnskjsabckvnd", "Iris");
	insert_data_database(collection, "jbfksjadbffsdnvhergjhefsdnvhergjhehjbgdskajdfsanvkjbnnewjreghkearljfqöwkölskdmljvkldfhnbgkjhkdfjbhewkjhfk9hrigjdfkgh4398ru32983z248lajsufsaeiu3984t59z24894t59hrigdshkjbfksnq3wbrhiefhiogubjagshvchcv32432hvfjewv324359z24894t59hrigjdfkjdfsn34tz87erzhgihsfdu", "Iris");
	insert_data_database(collection, "33333333sdfaknmvlkndfgiojeiofjeworj3oit9z6u50htjuogijndfkjnfewbr32g777777777777efwb3iutrh4ezoghierkbvsdhbvhvbejhfvewhfvwkugtiuerhzt8094ut5509hujtrkldfnkljnvjqwwkshfsdhgiuofhpüödkhtrprhkfön,ölgh,nöghknpotzjmiotrjhioeruth483z82809392z4879389trgihushiduj388", "Iris");
  */  /*mongodb search like*/
    printf("Found %s\n",search_substring_database(collection,"ergjheghkearljfqöwkölskdmljvkldfhnbgkjhkdfjbhewkjhfklajdshkjbfksjadbfhjbgdskajdfsanvkjbnnewjrnq3wbrhiefhioguhhdfsufs"));

	printf("******END SEARCH********");    
    
    /* mongodb delete*/
    doc = bson_new ();
    bson_oid_init (&oid, NULL);
    BSON_APPEND_OID (doc, "_id", &oid);
        
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

