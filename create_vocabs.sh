#!/bin/bash

pushd . >/dev/null

cd `dirname $0`/parsed

echo "cd to `pwd`"
echo "set single token on each line"
cat orig.txt | tr " " "\n" >vocab.source.txt
cat norm.txt | tr " " "\n" >vocab.target.txt

echo "delete special symbols"
sed -i 's/"//g' vocab.target.txt vocab.source.txt
sed -i 's/!//g' vocab.target.txt vocab.source.txt
sed -i "s/\'//g" vocab.target.txt vocab.source.txt
sed -i "s/(//g" vocab.target.txt vocab.source.txt 
sed -i "s/)//g" vocab.target.txt vocab.source.txt 
sed -i "s/,//g" vocab.target.txt vocab.source.txt 

echo "delete empty lines"
sed -i '/^\s*$/d' vocab.target.txt vocab.source.txt

echo "remove duplicates"
awk '!seen[$0]++' vocab.target.txt >test
mv test vocab.target.txt 
awk '!seen[$0]++' vocab.source.txt >test
mv test vocab.source.txt

echo "Files `pwd`/vocab.source.txt and `pwd`/vocab.target.txt created"

popd >/dev/null
