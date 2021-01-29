#!/usr/bin/env bash

set -x

cd ./datasets

wget --no-check-certificate https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/cezanne2photo.zip
unzip cezanne2photo.zip
rm -f cezanne2photo.zip

wget --no-check-certificate https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip
unzip monet2photo.zip
rm -f monet2photo.zip

wget --no-check-certificate https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ukiyoe2photo.zip
unzip ukiyoe2photo.zip
rm -f ukiyoe2photo.zip

wget --no-check-certificate https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip
unzip vangogh2photo.zip
rm -f vangogh2photo.zip

mkdir -p photo2art/testA
mkdir -p photo2art/testB
mkdir -p photo2art/trainA
mkdir -p photo2art/trainB

for img in cezanne2photo/testB/*
do
    mv "$img" photo2art/testA/
done

for img in $(ls cezanne2photo/testA)
do
   mv 'cezanne2photo/testA/'$img 'photo2art/testB/cezanne_'$img
done

for img in $(ls monet2photo/testA)
do
   mv 'monet2photo/testA/'$img 'photo2art/testB/monet_'$img
done

for img in $(ls ukiyoe2photo/testA)
do
   mv 'ukiyoe2photo/testA/'$img 'photo2art/testB/ukiyoe_'$img
done

for img in $(ls vangogh2photo/testA)
do
   mv 'vangogh2photo/testA/'$img 'photo2art/testB/vangogh_'$img
done

for img in cezanne2photo/trainB/*
do
    mv "$img" photo2art/trainA/
done

for img in $(ls cezanne2photo/trainA)
do
   mv 'cezanne2photo/trainA/'$img 'photo2art/trainB/cezanne_'$img
done

for img in $(ls monet2photo/trainA)
do
   mv 'monet2photo/trainA/'$img 'photo2art/trainB/monet_'$img
done

for img in $(ls ukiyoe2photo/trainA)
do
   mv 'ukiyoe2photo/trainA/'$img 'photo2art/trainB/ukiyoe_'$img
done

for img in $(ls vangogh2photo/trainA)
do
   mv 'vangogh2photo/trainA/'$img 'photo2art/trainB/vangogh_'$img
done

rm -rf cezanne2photo
rm -rf monet2photo
rm -rf ukiyoe2photo
rm -rf vangogh2photo

cd ..
