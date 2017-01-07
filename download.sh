

echo " 1/3 STANFORD PARSER DOWNLOAD "

mkdir stanford
cd stanford

wget http://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip
unzip stanford-parser-full-2015-12-09.zip -d .
mv stanford-parser-full-2015-12-09/* .
rm stanford-parser-full-2015-12-09
cd ..

echo " 2/3 SST DOWNLOAD "

mkdir sst
cd sst
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip stanfordSentimentTreebank.zip -d .
mv stanfordSentimentTreebank/* .
rm stanford-parser-full-2015-12-09.zip
cd ..

echo " 3/3 GLOVE DOWNLOAD "
mkdir glove
cd glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d .
cd ..
