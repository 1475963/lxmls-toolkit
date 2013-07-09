#!/usr/bin/env bash

# Export AWS authentication variables
export AWS_ACCESS_KEY_ID=AKIAJVYESLKCU75UVLPQ
export AWS_SECRET_ACCESS_KEY=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

# Run wordcount locally
python big_data/wordcount.py ../data/wikipedia/en_perline001.txt

# Run wordcount on Amazon
# IMPORTANT: you must export your AWS_ACCESS_KEY and AWS_SECRET_KEY variables before running this!
python big_data/wordcount.py -r emr ../data/wikipedia/en_perline001.txt --num-ec2-instances 2 --aws-region eu-west-1

# Run language detection locally
python big_data/kmers.py ../data/wikipedia/en_perline001.txt > output.en.txt
python big_data/kmers.py ../data/wikipedia/pt_perline01.txt > output.pt.txt
python big_data/postprocess.py
