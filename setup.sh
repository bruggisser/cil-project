#!/bin/bash

# Download the twitter data
DATASET="dataset"
if [ ! -f $DATASET.zip ]; then
  wget -O $DATASET.zip http://www.da.inf.ethz.ch/files/twitter-datasets.zip
fi
unzip $DATASET.zip
mv twitter-datasets/*.txt data/
rm -r twitter-datasets
rm $DATASET.zip

# Download glove
GLOVE="glove"
if [ ! -f $GLOVE.zip ]; then
  wget -O $GLOVE.zip https://nlp.stanford.edu/data/glove.twitter.27B.zip
fi
unzip $GLOVE.zip -d $GLOVE
mkdir data/glove
mv $GLOVE/*.txt data/glove/
rm $GLOVE.zip
rm -r $GLOVE

echo "Setup comet.ml? (y/n): "
read -r setup_comet

if [[ $setup_comet == "y" || $setup_comet == "Y" ]]; then
  # get Comet API variables
  echo "Enter your Comet.ml API: "
  read -r api_key
  echo "Enter your Comet.ml Project Name: "
  read -r project_name
  echo "Enter your Comet.ml Workspace: "
  read -r workspace
  echo '{"api_key":"'$api_key'", "project_name": "'$project_name'", "workspace": "'$workspace'"}' >> ./config/comet.json
fi
# run default setup script
source ./leonhard_project.sh
