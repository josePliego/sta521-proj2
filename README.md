# Notes on Reproducibility
The project is structured as follows.

+ cache/ contains intermediate rds files created throughout the scripts
+ data/ contains the three raw text files with the original images
+ docs/ contains the project instructions and the original paper
+ graphs/ contains all the figures included in the final document
+ R/ contains all the R scripts used

Inside the R/ folder, scripts are numbered according to what section they correspond to, and letters are used to indicate how they should be run. For example, script 02a_preparation.R is the first script used in section 2 and should be run before 02b_importance.R. Each script has a header describing the file inputs it requires and the outputs it produces (excluding graphs).

The file sta521-proj2.Rmd contains the raw markdown writeup. No code is included in this file, all figures are imported from the graphs/ folder and all tables are written with the console outputs produced by each script. The file sta521-proj2.bib contains the bibliography used in the document.

Every script that relies on random computations explicitly sets at least one random seed, so sourcing each script produces the same final paper every time.

Finally, in order to avoid conflicts with R package versions, we use the renv package to create a reproducible snapshot of the versions used. This is specially important because we use the developer versions of the packages parsnip and discrim that are located in Github at the time of writing.

Because of file size limit when submitting to Gradescope, the folders data/ and cache/ were emptied before compressing the repository.
