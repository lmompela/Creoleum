# Creoleum

Project Creole at UM - Kréyolum

- Data Collection : Martinican Creole & Haitian Creole data from newspapers and blog posts
- Data Storage and pre-processing : Organizing by source, date, and language spoken.
- Creation of a language classifier that can classify text depending on the percentage of creole content it has. From 0% to 100%. Personally I need it to be at least 50% for code-switching studies
- Cleaning up of the dataset from foreign characters (i.e. non utf-8 characters) mostly. Putting it one sentence per line.

#- Once the dataset is cleaned up, one sentence per line and with different levels of CS, finetune a language model using these different thresholds and test whether it can adequately reproduce CS sentences. Then test if it can interact and interpret correclty the command. Use it for an interpreter capable of using the same amount of code-switching as the person that's talking and adapat accordingly.
 
#Check if it's because of the Kréyolad that we see all these utf-8 characters. If so, just recollect everything about Jid, and ignore him from the articles collected out of Montray Kreyol. Or build a data checker that make sure that we don't copy the same article. 

NLTK package show all characters in document. 
