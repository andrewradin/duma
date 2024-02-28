#!/bin/bash

DATE=$(unzip -l ${1} | grep ${2} | sed -e 's/.*\(....-..-..\).*/\1/')
echo ${DATE}
