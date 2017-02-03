

echo " 1/4 STANFORD PARSER DOWNLOAD "

mkdir stanford
cd stanford

wget http://nlp.stanford.edu/software/stanford-parser-full-2015-12-09.zip
unzip stanford-parser-full-2015-12-09.zip -d .
mv stanford-parser-full-2015-12-09/* .
rm stanford-parser-full-2015-12-09
cd ..

echo " 2/4 SST DOWNLOAD "

mkdir sst
cd sst
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip stanfordSentimentTreebank.zip -d .
mv stanfordSentimentTreebank/* .
rm stanford-parser-full-2015-12-09.zip
cd ..

echo " 3/4 GLOVE 1 DOWNLOAD "
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d .

echo " 4/4 GLOVE 2 DOWNLOAD "
wget http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip glove.42B.300d.zip -d .


