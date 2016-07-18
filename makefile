

all:
	javac -cp "c:\A_Mikolaj\__proj\stanford-parser-full-2015-12-09\*;." -d "./java" Parser.java
	gcc -I "c:\Program Files (x86)\Java\jdk1.8.0_45\include" -I "c:\Program Files (x86)\Java\jdk1.8.0_45\include\win32" -L "c:\Program Files (x86)\Java\jdk1.8.0_45\lib" -m32 -c parser.c -ljvm
	gcc -I "c:\Program Files (x86)\Java\jdk1.8.0_45\include" -I "c:\Program Files (x86)\Java\jdk1.8.0_45\include\win32" -L "c:\Program Files (x86)\Java\jdk1.8.0_45\lib" -m32 -shared -o parser.dll parser.o -ljvm
	gcc -I "c:\Program Files (x86)\Java\jdk1.8.0_45\include" -I "c:\Program Files (x86)\Java\jdk1.8.0_45\include\win32" -L "c:\Program Files (x86)\Java\jdk1.8.0_45\lib" -m32 -o parser.exe parser.o -ljvm
	
