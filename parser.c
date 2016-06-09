#include <stdio.h>
#include <jni.h>

JNIEnv* create_vm(JavaVM **jvm)
{
    JNIEnv* env;
	JavaVMOption options;
    JavaVMInitArgs args;
    args.version = JNI_VERSION_1_6; args.nOptions = 1;
    options.optionString = "-Djava.class.path=./";
    args.options = &options;
    args.ignoreUnrecognized = 0;
    int rv;
    rv = JNI_CreateJavaVM(jvm, (void**)&env, &args);
    return env;
}

void invoke_class(JNIEnv* env)
{
    jclass parser_class;
    jmethodID parse_method;
    jmethodID power_method;
    jint number=20;
    jint exponent=3;
    printf("----------------- 2 \n");
    hello_world_class = (*env)->FindClass(env, "helloWorld");
    printf("----------------- 3 \n");
    //main_method = (*env)->GetStaticMethodID(env, hello_world_class, "main", "([Ljava/lang/String;)V");
    square_method = (*env)->GetStaticMethodID(env, hello_world_class, "square", "(I)I");
    power_method = (*env)->GetStaticMethodID(env, hello_world_class, "power", "(II)I");
    printf("----------------- 4 \n");
    //(*env)->CallStaticVoidMethod(env, hello_world_class, square_method, 2);
    printf("%d squared is %d\n", number,
        (*env)->CallStaticIntMethod(env, hello_world_class, square_method, number));
    /*printf("%d raised to the %d power is %d\n", number, exponent,
        (*env)->CallStaticIntMethod(env, hello_world_class, power_method, number, exponent));*/
}

int hello(int argc, char **argv)
{
    JavaVM *jvm;
    JNIEnv *env;
    env = create_vm(&jvm);
    if(env == NULL)
        return 1;
    printf("----------------- 1 \n");
    invoke_class(env);
    return 0;
}
