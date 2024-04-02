# Dataset
## Cook County
This directory contains:
- The original CSV file, in zipped format, for Cook County Sentencing data, which can also be found [at this link](https://datacatalog.cookcountyil.gov/Courts/Sentencing/tg8v-tm6u/data)
- 30 TTL files representing the knowledge graph (structured data) of the Cook County Sentencing data

The CSV file can be ran alongside the script `../code/cook-county-materialization.py` to generate the 30 TTL files AND a single tab-delimited text file of the KG. The file size of this text file surpassed Github's 50MB limitations and, thus, could not be hosted at it's original state from this repository. 

For the purpose of sharing, the text file has been split between $x$ counts of files to bypass the 50 MB limitation; however, for the purpose of KGE data - the files should be re-collated into a singular file.