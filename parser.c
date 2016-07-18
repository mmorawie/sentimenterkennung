#include <stdio.h>
#include <jni.h>
#include <string.h>
#include <stdlib.h>

JavaVM *jvm1;
JNIEnv *env1;

JNIEnv* create_vm(JavaVM **jvm)
{
    JNIEnv* env;
	JavaVMOption options;
    JavaVMInitArgs args;
    args.version = JNI_VERSION_1_6; args.nOptions = 1;
    options.optionString = 
"-Djava.class.path=.;stanford/stanford-parser.jar;stanford/ejml-0.23.jar;stanford/slf4j-api.jar;stanford/slf4j-simple.jar;stanford/stanford-parser-3.6.0-models.jar;./java";
	args.options = &options;
    args.ignoreUnrecognized = 0;
    int rv;
    rv = JNI_CreateJavaVM(jvm, (void**)&env, &args);
    return env;
}

int test() {
    JavaVM *jvm;
    JNIEnv *env;
    env = create_vm(&jvm);
    if(env == NULL) return 1;
    printf("--- 1 \n");
	char* buf = "aaaa";
    jclass parser_class;
    jmethodID parse_method;
	jmethodID test_method;
	parser_class = (*env)->FindClass(env, "Parser");
	printf("----------------- parser class \n");
	jclass hello_world_class = (*env)->FindClass(env, "helloWorld");
	printf("----------------- hello class \n");
	jmethodID square_method = (*env)->GetStaticMethodID(env, hello_world_class, "square", "(I)I");
	printf("----------------- square method \n");
	printf("%d squared is %d\n", 2,(*env)->CallStaticIntMethod(env, hello_world_class, square_method, 2));
    printf("----------------- square method run \n");
	
	test_method = (*env)->GetStaticMethodID(env, parser_class, "hello", "(I)V");
	printf("----------------- parser hello method \n");
	(*env)->CallStaticVoidMethod(env, parser_class, test_method, 1);
	printf("----------------- parser hello method run \n");
	parse_method = (*env)->GetStaticMethodID(env, parser_class, "process", "(Ljava/lang/String;)Ljava/lang/String;");
	printf("----------------- parser parse method \n");
	
	jstring jstrBuf = (*env)->NewStringUTF(env, buf);
	jstring returnString = (jstring) (*env)->CallStaticObjectMethod(env, parser_class, parse_method, jstrBuf);
	printf("----------------- parser parse method run \n");
	const char* sstr = (*env)->GetStringUTFChars(env,returnString, 0);
	printf(" %s <-- \n", sstr);
	(*env)->ReleaseStringUTFChars(env, returnString, sstr);
    return 0;
}

char* parse(char* buf){
	if(env1 == NULL) return "";
	
	jclass parser_class;
    jmethodID parse_method;
	
	parser_class = (*env1)->FindClass(env1, "Parser");
	parse_method = (*env1)->GetStaticMethodID(env1, parser_class, "process", "(Ljava/lang/String;)Ljava/lang/String;");
	
	jstring jstrBuf = (*env1)->NewStringUTF(env1, buf);
	jstring returnString = (jstring) (*env1)->CallStaticObjectMethod(env1, parser_class, parse_method, jstrBuf);
	
	const char* sstr = (*env1)->GetStringUTFChars(env1,returnString, 0);
	char* ret = strdup(sstr);
	(*env1)->ReleaseStringUTFChars(env1, returnString, sstr);
	
	//free(buf);
	return ret;
}

int startJVM(){
    env1 = create_vm(&jvm1);
	if(env1 == NULL){
		return -1;
	} else {
		return 0;
	}
}

int main(){
	return 0;
}
