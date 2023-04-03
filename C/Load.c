#include "Load.h"

void txt_load_quant(FILE* fp, unsigned char* scale, unsigned char* shift, unsigned char* zero_point){
    // 参数一般不超过1024字符，因此设置1024字符的缓存区
    char buffer[1024];
    if(fgets(buffer, 1024, fp) == NULL){
        printf("Error in load quant, miss some quant params");
        exit(0);
    }
    if(buffer[strlen(buffer)-1] == '\n'){buffer[strlen(buffer)-1] = '\0';}

    int num_quant = 0;
    char* quant_param = strtok(buffer, ",");
    *scale = atoi(quant_param);
    quant_param = strtok(NULL, ",");
    *shift = atoi(quant_param);
    quant_param = strtok(NULL, ",");
    *zero_point = atoi(quant_param);
}

void txt_load_tag_check(FILE* fp, char check){
    char buffer[10];
    if(fgets(buffer, 10, fp) == NULL){
        printf("miss param error!\n");
        exit(1);
    } else {
        if(check == buffer[0]){
            return;
        } else {
            printf("wrong check error!\n");
            exit(1);
        }
    } 
}

void txt_load_uint8(unsigned char** list, FILE* fp, int num){
    char buffer[1024];
    *list = (unsigned char*)malloc(sizeof(unsigned char)*num);

    int count = 0;
    while(fgets(buffer, 1024, fp) != NULL){
        // 处理为一个完整的字符串
        if(buffer[strlen(buffer)-1] == '\n'){buffer[strlen(buffer)-1] = '\0';}
        
        char* param = strtok(buffer, ",");
        while (param != NULL) {
            if (count == num) {printf("Too many param, want %d\n", num);exit(0);}
            (*list)[count] = atoi(param);
            count++;
            param = strtok(NULL, ",");
        }
        if (count == num) {
            break;
        }
    }
}

void txt_load_int(int** list, FILE* fp, int num){
    char buffer[1024];
    *list = (int*)malloc(sizeof(int)*num);

    int count = 0;
    while(fgets(buffer, 1024, fp) != NULL){
        // 处理为一个完整的字符串
        if(buffer[strlen(buffer)-1] == '\n'){buffer[strlen(buffer)-1] = '\0';}
        
        char* param = strtok(buffer, ",");
        while (param != NULL) {
            if (count == num) {printf("Too many param, want %d\n", num);exit(0);}
            (*list)[count] = atoi(param);
            count++;
            param = strtok(NULL, ",");
        }
        if (count == num) {
            break;
        }
    }
}

void bin_load_uint8(unsigned char** list, FILE* fp, int num){
    char buffer[1024];
    *list = (unsigned char*)malloc(sizeof(unsigned char)*num);

    int i = 0;unsigned char temp;
    for (i = 0; i < num; i++) {
        fread(&temp, sizeof(unsigned char), 1, fp);
        (*list)[i] = temp;
    }
}

void bin_load_int(int** list, FILE* fp, int num){
    char buffer[1024];
    *list = (int*)malloc(sizeof(int)*num);

    int i = 0;int temp;
    for (i = 0; i < num; i++) {
        fread(&temp, sizeof(int), 1, fp);
        (*list)[i] = temp;
    }
}